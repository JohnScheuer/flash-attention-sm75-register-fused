// =============================================================================
// flash_fused_kernel.cu
//
// Flash Attention Register-Fused Kernel para SM75.
//
// Arquitetura do pipeline (por warp, por d_out tile):
//
//   [Registers] O_acc  ←── persiste por todos os KV tiles
//   [Registers] S_acc  ←── score tile (QK^T), descartado após P@V
//
//   Para cada KV tile:
//     1. Computar S_acc = Q @ K^T  (WMMA, Q/K carregados de smem)
//     2. warp_softmax_unnorm(S_acc)    → zero smem
//     3. online_update(O_acc, state)   → zero smem
//     4. apply_beta_to_S(S_acc, beta)  → zero smem
//     5. smem_P ← store(S_acc)        ← ÚNICO store intermediário
//     6. P_frag ← load(smem_P)
//     7. smem_V ← load V tile
//     8. O_acc += P_frag @ V_frag     (WMMA acumula diretamente em O_acc)
//
//   Após todos KV tiles:
//     9. finalize_O(O_acc, state)      → divide por l_final em registers
//    10. store O_acc → smem_scratch → global O
//
// Shared Memory por bloco (1 warp):
//   smem_Q:     16 × 16 × 2 bytes = 512 B   (half, tile de Q)
//   smem_K:     16 × 16 × 2 bytes = 512 B   (half, tile de K)
//   smem_P:     16 × 16 × 2 bytes = 512 B   (half, tile de P = softmax(QK^T))
//   smem_V:     16 × 16 × 2 bytes = 512 B   (half, tile de V)
//   ─────────────────────────────────────────────
//   Total:                          2048 B = 2 KB
//
//   (Para store final de O usamos smem_P reinterpretada como float: 1 KB)
//
// Ocupação estimada (SM75, 64 KB smem/SM, 65536 regs/SM):
//   Blocos por SM = min(64KB/2KB, 65536/(32*N_regs)) >> 32 blocos → >50% ocup.
// =============================================================================

#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>
#include <cstdio>

#include "wmma_layout.cuh"
#include "warp_softmax.cuh"
#include "online_state.cuh"
#include "flash_fused.cuh"

using namespace nvcuda;

// Tamanho dos tiles — fixos em 16 para combinar com WMMA m16n16k16
static constexpr int TM = 16;  // linhas de Q por tile (e de O)
static constexpr int TN = 16;  // linhas de K/V por tile
static constexpr int TK = 16;  // dimensão interna (chunk de d)

// =============================================================================
// Kernel principal
// =============================================================================
__global__ void __launch_bounds__(32, 8)
flash_fused_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int N,
    int d,
    float scale)
{
    // ── Identificação ─────────────────────────────────────────────────────────
    const int lane       = threadIdx.x & 31;
    const int q_tile_idx = blockIdx.x;           // qual tile de 16 linhas de Q
    const int q_base     = q_tile_idx * TM;      // linha inicial em Q (e em O)

    if (q_base >= N) return;

    const int num_kv_tiles = (N + TN - 1) / TN;
    const int num_d_tiles  = d / TK;             // d deve ser múltiplo de TK=16

    // ── Shared memory ─────────────────────────────────────────────────────────
    // Layout: [smem_Q | smem_K | smem_P | smem_V]  cada = TM*TK halfs = 256h
    extern __shared__ half smem[];
    half* smem_Q = smem;
    half* smem_K = smem_Q + TM * TK;
    half* smem_P = smem_K + TM * TK;
    half* smem_V = smem_P + TM * TK;

    // ── Loop sobre colunas de saída (tiles de d) ───────────────────────────────
    // Para cada d_out, calculamos a coluna correspondente de O.
    // S = Q @ K^T não depende de d_out, mas precisamos recomputá-lo por
    // d_out porque o online state e O_acc precisam ser reiniciados.
    //
    // OTIMIZAÇÃO: Para d pequeno (ex: 64), num_d_tiles=4, então recomputar
    // S 4 vezes é aceitável. Para d grande, use blocking 2D.

    for (int d_out = 0; d_out < num_d_tiles; d_out++) {

        // Inicializar acumulador de saída e estado online
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc;
        wmma::fill_fragment(O_acc, 0.0f);

        OnlineState online;
        online_state_init(online);

        // ── Loop sobre tiles de K/V ───────────────────────────────────────────
        for (int kv = 0; kv < num_kv_tiles; kv++) {
            const int kv_base = kv * TN;

            // ── Computar S = Q @ K^T  (somando sobre todos os chunks de d) ───
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
            wmma::fill_fragment(S_acc, 0.0f);

            for (int d_in = 0; d_in < num_d_tiles; d_in++) {
                const int d_base = d_in * TK;

                // Carregar Q[q_base:q_base+16, d_base:d_base+16]
                #pragma unroll
                for (int i = lane; i < TM * TK; i += 32) {
                    int r = i / TK;
                    int c = i % TK;
                    int gr = q_base + r;
                    int gc = d_base  + c;
                    smem_Q[i] = (gr < N && gc < d)
                                ? Q[gr * d + gc]
                                : __float2half(0.0f);
                }

                // Carregar K[kv_base:kv_base+16, d_base:d_base+16]
                #pragma unroll
                for (int i = lane; i < TN * TK; i += 32) {
                    int r = i / TK;
                    int c = i % TK;
                    int gr = kv_base + r;
                    int gc = d_base  + c;
                    smem_K[i] = (gr < N && gc < d)
                                ? K[gr * d + gc]
                                : __float2half(0.0f);
                }
                __syncwarp();

                // Fragmentos para Q (matrix_a, row_major) e K^T (matrix_b, col_major)
                // Carregar K como col_major sobre uma matriz row_major equivale
                // a transpor: K_stored[r][c] lido como col_major → K^T[c][r].
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Q_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kt_frag;

                wmma::load_matrix_sync(Q_frag,  smem_Q, TK);
                wmma::load_matrix_sync(Kt_frag, smem_K, TK);

                // S += Q_chunk @ K_chunk^T
                wmma::mma_sync(S_acc, Q_frag, Kt_frag, S_acc);
                __syncwarp();
            }

            // ── Softmax unnorm em registradores ────────────────────────────────
            RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);

            // ── Online rescaling de O_acc ──────────────────────────────────────
            BetaFactors beta = online_update(O_acc, online, ms);

            // ── Escalar S_acc por beta (por linha) ─────────────────────────────
            apply_beta_to_S(S_acc, beta);

            // ── S_acc → half → smem_P (ÚNICO store intermediário) ──────────────
            // Precisamos converter float→half antes de usar como matrix_a.
            // Fazemos isso escrevendo em smem com conversão explícita.
            {
                const int gid = lane >> 2;
                const int tig = lane & 3;

                int r_u = gid;
                int r_l = gid + 8;
                int c0  = tig * 2;
                int c1  = tig * 2 + 1;
                int c4  = tig * 2 + 8;
                int c5  = tig * 2 + 9;

                smem_P[r_u * 16 + c0] = __float2half(S_acc.x[0]);
                smem_P[r_u * 16 + c1] = __float2half(S_acc.x[1]);
                smem_P[r_u * 16 + c4] = __float2half(S_acc.x[4]);
                smem_P[r_u * 16 + c5] = __float2half(S_acc.x[5]);

                smem_P[r_l * 16 + c0] = __float2half(S_acc.x[2]);
                smem_P[r_l * 16 + c1] = __float2half(S_acc.x[3]);
                smem_P[r_l * 16 + c4] = __float2half(S_acc.x[6]);
                smem_P[r_l * 16 + c5] = __float2half(S_acc.x[7]);
            }
            __syncwarp();

            // ── Carregar P como matrix_a ───────────────────────────────────────
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> P_frag;
            wmma::load_matrix_sync(P_frag, smem_P, 16);

            // ── Carregar V[kv_base:kv_base+16, d_out*16:(d_out+1)*16] ─────────
            const int v_d_base = d_out * TK;
            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                int r = i / TK;
                int c = i % TK;
                int gr = kv_base  + r;
                int gc = v_d_base + c;
                smem_V[i] = (gr < N && gc < d)
                            ? V[gr * d + gc]
                            : __float2half(0.0f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> V_frag;
            wmma::load_matrix_sync(V_frag, smem_V, TK);

            // ── O_acc += P @ V  (acumula diretamente no O persistente) ─────────
            wmma::mma_sync(O_acc, P_frag, V_frag, O_acc);
            __syncwarp();

        } // fim loop KV tiles

        // ── Normalização final (divide O_acc por l_final) ─────────────────────
        finalize_O(O_acc, online);

        // ── Escrever O_acc → global O ─────────────────────────────────────────
        // Reusamos smem_P como scratch float (256 floats = 1 KB)
        float* smem_scratch = reinterpret_cast<float*>(smem_P);

        wmma::store_matrix_sync(smem_scratch, O_acc, 16, wmma::mem_row_major);
        __syncwarp();

        const int o_d_base = d_out * TK;
        #pragma unroll
        for (int i = lane; i < TM * TK; i += 32) {
            int r = i / TK;
            int c = i % TK;
            int gr = q_base  + r;
            int gc = o_d_base + c;
            if (gr < N && gc < d) {
                O[gr * d + gc] = __float2half(smem_scratch[i]);
            }
        }
        __syncwarp();

    } // fim loop d_out tiles
}

// =============================================================================
// Implementação da interface pública
// =============================================================================
void flash_fused_launch(const FlashFusedParams& p, cudaStream_t stream)
{
    const int num_q_tiles = (p.N + 15) / 16;

    // Shared memory: 4 tiles × 16×16 × sizeof(half) = 4 × 512 = 2048 bytes
    // Mas para o store final de O (float) precisamos de 16×16×4 = 1024 bytes.
    // smem_P é reusada para isso, então o máximo é 2048 bytes.
    // Para garantir, alocar max(2048, 1024 + 3*512) = 2048.
    const int smem_bytes = 4 * TM * TK * sizeof(half);

    // Definir atributo de smem se necessário (para kernels > 48KB, não é o caso aqui)
    cudaFuncSetAttribute(
        flash_fused_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);

    dim3 grid(num_q_tiles);
    dim3 block(32);  // 1 warp

    flash_fused_kernel<<<grid, block, smem_bytes, stream>>>(
        p.Q, p.K, p.V, p.O,
        p.N, p.d, p.scale);
}


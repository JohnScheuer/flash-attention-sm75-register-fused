#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cuda_runtime.h>

#include "wmma_layout.cuh"
#include "warp_softmax.cuh"
#include "online_step.cuh"
#include "flash_reuse_qk.cuh"

using namespace nvcuda;

static constexpr int TM = 16;
static constexpr int TN = 16;
static constexpr int TK = 16;
static constexpr int D  = 64;
static constexpr int D_TILES = D / TK;   // 4

// ============================================================================
// Scatter do accumulator S_acc -> tile FP16 em shared memory
// usando o layout confirmado do accumulator WMMA no SM75.
// ============================================================================
__device__ __forceinline__
void scatter_S_acc_to_half_tile(
    const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S_acc,
    half* smem_P)
{
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;

    const int r_u = gid;
    const int r_l = gid + 8;

    const int c0 = tig * 2;
    const int c1 = tig * 2 + 1;
    const int c4 = tig * 2 + 8;
    const int c5 = tig * 2 + 9;

    smem_P[r_u * 16 + c0] = __float2half(S_acc.x[0]);
    smem_P[r_u * 16 + c1] = __float2half(S_acc.x[1]);
    smem_P[r_u * 16 + c4] = __float2half(S_acc.x[4]);
    smem_P[r_u * 16 + c5] = __float2half(S_acc.x[5]);

    smem_P[r_l * 16 + c0] = __float2half(S_acc.x[2]);
    smem_P[r_l * 16 + c1] = __float2half(S_acc.x[3]);
    smem_P[r_l * 16 + c4] = __float2half(S_acc.x[6]);
    smem_P[r_l * 16 + c5] = __float2half(S_acc.x[7]);
}

// ============================================================================
// Kernel especializado para d = 64
//
// Grande mudança:
//   - S = QK^T é computado UMA vez por kv_tile
//   - P = softmax(S) é computado UMA vez por kv_tile
//   - O mesmo P é reutilizado para 4 tiles de saída:
//       O[:, 0:16], O[:,16:32], O[:,32:48], O[:,48:64]
//
// Isso remove a recomputação redundante de QK^T do kernel anterior.
// ============================================================================
__global__ void __launch_bounds__(32, 4)
flash_fused_reuse_qk_kernel_d64(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int N,
    float scale)
{
    const int lane   = threadIdx.x & 31;
    const int q_tile = blockIdx.x;
    const int q_base = q_tile * TM;

    if (q_base >= N) return;

    // Assumimos N padded/alinhado em múltiplo de 16.
    const int num_kv_tiles = N / TN;

    // Shared memory total:
    //   Q  = 16x16 half = 512 B
    //   K  = 16x16 half = 512 B
    //   P  = 16x16 half = 512 B
    //   V  = 16x16 half = 512 B
    // total = 2048 B
    extern __shared__ half smem[];
    half* smem_Q = smem;
    half* smem_K = smem_Q + TM * TK;
    half* smem_P = smem_K + TN * TK;
    half* smem_V = smem_P + TM * TN;

    // 4 acumuladores persistentes de saída:
    //   O_acc[0] -> cols  0..15
    //   O_acc[1] -> cols 16..31
    //   O_acc[2] -> cols 32..47
    //   O_acc[3] -> cols 48..63
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc[D_TILES];
    #pragma unroll
    for (int t = 0; t < D_TILES; t++) {
        wmma::fill_fragment(O_acc[t], 0.0f);
    }

    OnlineState state;
    online_state_init(state);

    // =========================================================================
    // Loop sobre KV tiles
    // =========================================================================
    for (int kv = 0; kv < num_kv_tiles; kv++) {
        const int kv_base = kv * TN;

        // ---------------------------------------------------------------------
        // 1) S_acc = Q @ K^T   (acumulado sobre os 4 chunks de d=64)
        // ---------------------------------------------------------------------
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
        wmma::fill_fragment(S_acc, 0.0f);

        #pragma unroll
        for (int d_in = 0; d_in < D_TILES; d_in++) {
            const int d_base = d_in * TK;

            // Load Q tile
            #pragma unroll
            for (int i = lane; i < TM * TK; i += 32) {
                int r = i / TK;
                int c = i % TK;
                int gr = q_base + r;
                int gc = d_base + c;
                smem_Q[i] = Q[gr * D + gc];
            }

            // Load K tile
            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                int r = i / TK;
                int c = i % TK;
                int gr = kv_base + r;
                int gc = d_base + c;
                smem_K[i] = K[gr * D + gc];
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kt_frag;

            wmma::load_matrix_sync(Q_frag,  smem_Q, TK);
            wmma::load_matrix_sync(Kt_frag, smem_K, TK);

            wmma::mma_sync(S_acc, Q_frag, Kt_frag, S_acc);
            __syncwarp();
        }

        // ---------------------------------------------------------------------
        // 2) Softmax unnorm em registradores
        // ---------------------------------------------------------------------
        RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);

        // ---------------------------------------------------------------------
        // 3) Atualiza estado online UMA vez
        // ---------------------------------------------------------------------
        OnlineStep step = online_step_update(state, ms);

        // ---------------------------------------------------------------------
        // 4) Escala tile atual por beta
        // ---------------------------------------------------------------------
        apply_beta_to_S(S_acc, step);

        // ---------------------------------------------------------------------
        // 5) Salva P em shared UMA vez e recarrega como matrix_a
        // ---------------------------------------------------------------------
        scatter_S_acc_to_half_tile(S_acc, smem_P);
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> P_frag;
        wmma::load_matrix_sync(P_frag, smem_P, 16);

        // ---------------------------------------------------------------------
        // 6) Reutiliza o mesmo P para TODOS os 4 d_out tiles
        // ---------------------------------------------------------------------
        #pragma unroll
        for (int d_out = 0; d_out < D_TILES; d_out++) {
            // Reescala O antigo por alpha antes de acumular contribuição nova
            apply_alpha_to_O(O_acc[d_out], step);

            // Load V tile correspondente a este d_out
            const int v_col_base = d_out * TK;

            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                int r = i / TK;
                int c = i % TK;
                int gr = kv_base + r;
                int gc = v_col_base + c;
                smem_V[i] = V[gr * D + gc];
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> V_frag;
            wmma::load_matrix_sync(V_frag, smem_V, TK);

            wmma::mma_sync(O_acc[d_out], P_frag, V_frag, O_acc[d_out]);
            __syncwarp();
        }
    }

    // =========================================================================
    // Finalização e store
    // =========================================================================
    #pragma unroll
    for (int d_out = 0; d_out < D_TILES; d_out++) {
        finalize_O(O_acc[d_out], state);

        // Reusa região [smem_P | smem_V] = 1024 B como scratch float[256]
        float* smem_scratch = reinterpret_cast<float*>(smem_P);
        wmma::store_matrix_sync(smem_scratch, O_acc[d_out], 16, wmma::mem_row_major);
        __syncwarp();

        const int out_col_base = d_out * TK;

        #pragma unroll
        for (int i = lane; i < TM * TK; i += 32) {
            int r = i / TK;
            int c = i % TK;
            int gr = q_base + r;
            int gc = out_col_base + c;
            O[gr * D + gc] = __float2half(smem_scratch[i]);
        }
        __syncwarp();
    }
}

// ============================================================================
// Host launcher
// ============================================================================
void flash_reuse_qk_launch(
    const FlashReuseQKParams& p,
    cudaStream_t stream)
{
    if (p.d != 64) {
        fprintf(stderr,
                "[flash_reuse_qk_launch] Esta versao requer d == 64. Recebido d=%d\n",
                p.d);
        return;
    }

    if ((p.N % 16) != 0) {
        fprintf(stderr,
                "[flash_reuse_qk_launch] Esta versao requer N multiplo de 16 (ou padded). Recebido N=%d\n",
                p.N);
        return;
    }

    const int num_q_tiles = p.N / 16;
    const int smem_bytes  = 4 * 16 * 16 * (int)sizeof(half); // 2048 B

    cudaFuncSetAttribute(
        flash_fused_reuse_qk_kernel_d64,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);

    dim3 grid(num_q_tiles);
    dim3 block(32);

    flash_fused_reuse_qk_kernel_d64<<<grid, block, smem_bytes, stream>>>(
        p.Q, p.K, p.V, p.O, p.N, p.scale);
}


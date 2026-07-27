#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "wmma_layout.cuh"
#include "warp_softmax.cuh"
#include "online_step.cuh"

using namespace nvcuda;

// ============================================================================
// Constantes
// ============================================================================
static constexpr int TM      = 16;
static constexpr int TN      = 16;
static constexpr int TK      = 16;
static constexpr int D_FIXED = 64;
static constexpr int D_TILES = D_FIXED / TK;   // 4

// ============================================================================
// Helpers
// ============================================================================
static inline void check_cuda(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error em %s: %s\n", w, cudaGetErrorString(e));
        std::exit(1);
    }
}
static inline float rand_val() {
    return (std::rand() / (float)RAND_MAX - 0.5f) * 0.5f;
}
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline void print_sep(int w = 110) {
    for (int i = 0; i < w; i++) std::putchar('-');
    std::putchar('\n');
}

// ============================================================================
// Scatter accumulator → smem half tile  (layout SM75 confirmado)
// ============================================================================
__device__ __forceinline__
void scatter_S_to_half(
    const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S,
    half* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2,  tig  = lane & 3;
    const int ru   = gid,        rl   = gid + 8;
    const int c0   = tig*2,      c1   = tig*2+1;
    const int c4   = tig*2+8,    c5   = tig*2+9;

    dst[ru*16+c0] = __float2half(S.x[0]);
    dst[ru*16+c1] = __float2half(S.x[1]);
    dst[ru*16+c4] = __float2half(S.x[4]);
    dst[ru*16+c5] = __float2half(S.x[5]);
    dst[rl*16+c0] = __float2half(S.x[2]);
    dst[rl*16+c1] = __float2half(S.x[3]);
    dst[rl*16+c4] = __float2half(S.x[6]);
    dst[rl*16+c5] = __float2half(S.x[7]);
}

// ============================================================================
// Shared memory layout para o kernel v2
//
// Ping-pong para K:
//   smem_K[2][TN * D_FIXED]  = 2 × 1024 halfs = 4096 B
//   buf 0 e buf 1 alternam: enquanto processa buf_cur, carrega buf_next
//
// Q: privado, carregado uma vez antes do loop KV
//   smem_Q[TM * D_FIXED] = 1024 halfs = 2048 B
//
// P: tile de P após softmax
//   smem_P[TM * TN] = 256 halfs = 512 B
//
// V: tile de V (um d_out por vez)
//   smem_V[TN * TK] = 256 halfs = 512 B
//
// Total: 4096 + 2048 + 512 + 512 = 7168 B ≈ 7 KB
// ============================================================================
static constexpr int SMEM_K_HALFS  = 2 * TN * D_FIXED;   // 2048 (ping-pong)
static constexpr int SMEM_Q_HALFS  = TM * D_FIXED;        // 1024
static constexpr int SMEM_P_HALFS  = TM * TN;             // 256
static constexpr int SMEM_V_HALFS  = TN * TK;             // 256
static constexpr int SMEM_TOTAL    = SMEM_K_HALFS
                                   + SMEM_Q_HALFS
                                   + SMEM_P_HALFS
                                   + SMEM_V_HALFS;         // 3584 halfs
static constexpr int SMEM_BYTES    = SMEM_TOTAL * (int)sizeof(half); // 7168 B

// ============================================================================
// Máscara causal: retorna true se o tile inteiro está acima da diagonal
// (pode ser descartado completamente) para atenção causal.
//
//   q_base  : linha inicial do tile de Q  (múltiplo de TM)
//   kv_base : linha inicial do tile de KV (múltiplo de TN)
//
// Se kv_base > q_base + TM - 1, todas as posições do tile têm j > i → zero.
// ============================================================================
__device__ __forceinline__
bool tile_fully_masked(int q_base, int kv_base) {
    return kv_base > q_base + TM - 1;
}

// Retorna true se o tile tem ALGUMA posição causal válida
// (kv_base <= q_base + TM - 1) mas NÃO é completamente válido
// (kv_base + TN - 1 > q_base), ou seja, está na diagonal.
__device__ __forceinline__
bool tile_is_diagonal(int q_base, int kv_base) {
    return (kv_base <= q_base + TM - 1) && (kv_base + TN - 1 > q_base);
}

// Aplica máscara causal ao accumulator S_acc em registradores.
// Zera posições onde j > i (usando o layout do SM75).
// q_base_row: linha de Q dentro da janela (0..TM-1 + q_base)
// kv_base_col: coluna de KV (= kv_base)
__device__ __forceinline__
void apply_causal_mask(
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S_acc,
    int q_base,
    int kv_base)
{
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;

    // Linhas deste thread no tile de saída
    const int row_u = q_base + gid;        // upper row (global)
    const int row_l = q_base + gid + 8;    // lower row (global)

    // Colunas: c0,c1 (left half), c4,c5 (right half)
    // Mapeamento: col_global = kv_base + col_local
    const int c0 = kv_base + tig*2;
    const int c1 = kv_base + tig*2 + 1;
    const int c4 = kv_base + tig*2 + 8;
    const int c5 = kv_base + tig*2 + 9;

    // upper row: x[0],x[1],x[4],x[5]
    if (c0 > row_u) S_acc.x[0] = -FLT_MAX * 0.5f;  // -inf para softmax
    if (c1 > row_u) S_acc.x[1] = -FLT_MAX * 0.5f;
    if (c4 > row_u) S_acc.x[4] = -FLT_MAX * 0.5f;
    if (c5 > row_u) S_acc.x[5] = -FLT_MAX * 0.5f;

    // lower row: x[2],x[3],x[6],x[7]
    if (c0 > row_l) S_acc.x[2] = -FLT_MAX * 0.5f;
    if (c1 > row_l) S_acc.x[3] = -FLT_MAX * 0.5f;
    if (c4 > row_l) S_acc.x[6] = -FLT_MAX * 0.5f;
    if (c5 > row_l) S_acc.x[7] = -FLT_MAX * 0.5f;
}

// ============================================================================
// Kernel v2: flatargs + causal + ping-pong K + acumulação FP16 opcional
//
// Flags de compilação:
//   CAUSAL    = 0 (não causal, padrão) ou 1 (causal)
//   USE_FP16_ACC = 0 (acumula em FP32, padrão) ou 1 (acumula em FP16)
//
// Grid: (N/TM, H, B)
// Block: 32 threads (1 warp)
// ============================================================================
template<int CAUSAL, int USE_FP16_ACC>
__global__ void __launch_bounds__(32, 4)
flash_v2_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int N,
    int H,
    float scale)
{
    const int q_tile  = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch   = blockIdx.z;
    const int lane    = threadIdx.x & 31;

    const int q_base = q_tile * TM;
    if (q_base >= N) return;

    const int head_stride = N * D_FIXED;
    const int base_off    = (batch * H + head_id) * head_stride;

    const half* __restrict__ Qh = Q + base_off;
    const half* __restrict__ Kh = K + base_off;
    const half* __restrict__ Vh = V + base_off;
    half*       __restrict__ Oh = O + base_off;

    // Para atenção causal: só processa kv_tiles com kv_base <= q_base
    // (tiles acima da diagonal são todos zero)
    const int num_kv = CAUSAL
                     ? (q_base / TN + 1)   // só tiles causalmente válidos
                     : (N / TN);

    // ── Shared memory ─────────────────────────────────────────────────────────
    extern __shared__ half smem[];
    half* smem_Kpp = smem;                          // ping-pong K [2][TN*D_FIXED]
    half* smem_Q   = smem_Kpp + SMEM_K_HALFS;      // Q [TM*D_FIXED]
    half* smem_P   = smem_Q   + SMEM_Q_HALFS;      // P [TM*TN]
    half* smem_V   = smem_P   + SMEM_P_HALFS;      // V [TN*TK]

    // ── Pré-carregar Q (uma vez, válido para todos os kv tiles) ───────────────
    #pragma unroll
    for (int d_in = 0; d_in < D_TILES; d_in++) {
        const int db = d_in * TK;
        #pragma unroll
        for (int i = lane; i < TM * TK; i += 32) {
            const int r = i / TK, c = i % TK;
            const int gr = q_base + r;
            smem_Q[r * D_FIXED + db + c] =
                (gr < N) ? Qh[gr * D_FIXED + db + c] : __float2half(0.f);
        }
    }
    __syncwarp();

    // ── Pré-carregar K para o buffer 0 (kv=0) ────────────────────────────────
    // Isso inicia o pipeline de ping-pong
    if (num_kv > 0) {
        const int kv_base0 = 0;
        #pragma unroll
        for (int i = lane; i < TN * D_FIXED; i += 32) {
            const int r = i / D_FIXED, c = i % D_FIXED;
            const int gr = kv_base0 + r;
            smem_Kpp[i] = (gr < N) ? Kh[gr * D_FIXED + c] : __float2half(0.f);
        }
        __syncwarp();
    }

    // ── Acumuladores de saída ─────────────────────────────────────────────────
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc[D_TILES];
    #pragma unroll
    for (int t = 0; t < D_TILES; t++)
        wmma::fill_fragment(O_acc[t], 0.0f);

    OnlineState state;
    online_state_init(state);

    // =========================================================================
    // Loop KV com ping-pong de K
    // =========================================================================
    for (int kv = 0; kv < num_kv; kv++) {
        const int kv_base  = kv * TN;
        const int cur_buf  = kv & 1;          // buffer atual
        const int next_buf = 1 - cur_buf;     // buffer próximo

        // ── Disparar load assíncrono do próximo K enquanto processa o atual ───
        // Em SM75 não há cp.async, então usamos loads escalares sobrepostos
        // com a computação WMMA (o compilador pode reordenar instruções).
        // O __syncwarp() antes do uso garante coerência.
        const int kv_next = kv + 1;
        if (kv_next < num_kv) {
            const int kv_base_next = kv_next * TN;
            const int off_next     = next_buf * TN * D_FIXED;
            #pragma unroll
            for (int i = lane; i < TN * D_FIXED; i += 32) {
                const int r = i / D_FIXED, c = i % D_FIXED;
                const int gr = kv_base_next + r;
                smem_Kpp[off_next + i] =
                    (gr < N) ? Kh[gr * D_FIXED + c] : __float2half(0.f);
            }
            // NÃO sincronizamos aqui — deixamos o load "voar" enquanto
            // executamos os WMMAs abaixo.
        }

        // ── Para causal: verificar se o tile está completamente mascarado ─────
        if (CAUSAL && tile_fully_masked(q_base, kv_base)) {
            // Tile acima da diagonal: pular completamente
            // Mas ainda precisamos sincronizar para o próximo load de K
            __syncwarp();
            continue;
        }

        // ── S = Q @ K^T ───────────────────────────────────────────────────────
        // Sincronizar para garantir que o buffer atual de K está pronto
        __syncwarp();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
        wmma::fill_fragment(S_acc, 0.0f);

        const int k_off = cur_buf * TN * D_FIXED;

        #pragma unroll
        for (int d_in = 0; d_in < D_TILES; d_in++) {
            const int db = d_in * TK;

            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Qf;
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> Kf;

            wmma::load_matrix_sync(Qf, smem_Q + db,           D_FIXED);
            wmma::load_matrix_sync(Kf, smem_Kpp + k_off + db, D_FIXED);
            wmma::mma_sync(S_acc, Qf, Kf, S_acc);
            __syncwarp();
        }

        // ── Máscara causal para tiles diagonais ───────────────────────────────
        if (CAUSAL && tile_is_diagonal(q_base, kv_base)) {
            apply_causal_mask(S_acc, q_base, kv_base);
        }

        // ── Softmax unnorm ────────────────────────────────────────────────────
        RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);

        // ── Online step ───────────────────────────────────────────────────────
        OnlineStep step = online_step_update(state, ms);

        // ── Beta em S_acc ─────────────────────────────────────────────────────
        apply_beta_to_S(S_acc, step);

        // ── P scatter → smem ──────────────────────────────────────────────────
        scatter_S_to_half(S_acc, smem_P);
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Pf;
        wmma::load_matrix_sync(Pf, smem_P, TN);

        // ── P @ V para os 4 tiles de saída ────────────────────────────────────
        #pragma unroll
        for (int d_out = 0; d_out < D_TILES; d_out++) {
            apply_alpha_to_O(O_acc[d_out], step);

            const int vc = d_out * TK;
            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                const int r = i / TK, c = i % TK;
                const int gr = kv_base + r;
                smem_V[i] = (gr < N) ? Vh[gr * D_FIXED + vc + c] : __float2half(0.f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> Vf;
            wmma::load_matrix_sync(Vf, smem_V, TK);
            wmma::mma_sync(O_acc[d_out], Pf, Vf, O_acc[d_out]);
            __syncwarp();
        }
    }

    // =========================================================================
    // Finaliza e escreve O
    // =========================================================================
    #pragma unroll
    for (int d_out = 0; d_out < D_TILES; d_out++) {
        finalize_O(O_acc[d_out], state);

        float* scratch = reinterpret_cast<float*>(smem_P);
        wmma::store_matrix_sync(scratch, O_acc[d_out], 16, wmma::mem_row_major);
        __syncwarp();

        const int oc = d_out * TK;
        #pragma unroll
        for (int i = lane; i < TM * TK; i += 32) {
            const int r = i / TK, c = i % TK;
            const int gr = q_base + r;
            if (gr < N)
                Oh[gr * D_FIXED + oc + c] = __float2half(scratch[i]);
        }
        __syncwarp();
    }
}

// ============================================================================
// Launcher
// ============================================================================
enum AttentionMode { FULL = 0, CAUSAL = 1 };

static void launch_v2(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int B, int H, int N,
    float scale,
    AttentionMode mode,
    cudaStream_t stream = 0)
{
    cudaFuncSetAttribute(flash_v2_kernel<0,0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(flash_v2_kernel<1,0>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);

    dim3 grid(N / TM, H, B);
    dim3 block(32);

    if (mode == CAUSAL)
        flash_v2_kernel<1,0><<<grid, block, SMEM_BYTES, stream>>>(
            Q, K, V, O, N, H, scale);
    else
        flash_v2_kernel<0,0><<<grid, block, SMEM_BYTES, stream>>>(
            Q, K, V, O, N, H, scale);
}

// ============================================================================
// CPU reference — full attention
// ============================================================================
static void cpu_ref_full(
    const half* Q, const half* K, const half* V, float* O_ref,
    int B, int H, int N, int d, float scale)
{
    float* S = new float[(size_t)N * N];
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int off = (b*H+h)*N*d;
            const half* q=Q+off; const half* k=K+off;
            const half* v=V+off; float* o=O_ref+off;

            for (int i=0;i<N;i++)
                for (int j=0;j<N;j++) {
                    float acc=0.f;
                    for (int x=0;x<d;x++)
                        acc+=__half2float(q[i*d+x])*__half2float(k[j*d+x]);
                    S[i*N+j]=acc*scale;
                }
            for (int i=0;i<N;i++){
                float mx=-FLT_MAX;
                for (int j=0;j<N;j++) mx=fmaxf(mx,S[i*N+j]);
                float sm=0.f;
                for (int j=0;j<N;j++){S[i*N+j]=expf(S[i*N+j]-mx);sm+=S[i*N+j];}
                for (int j=0;j<N;j++) S[i*N+j]/=sm;
            }
            for (int i=0;i<N;i++)
                for (int j=0;j<d;j++){
                    float acc=0.f;
                    for (int x=0;x<N;x++)
                        acc+=S[i*N+x]*__half2float(v[x*d+j]);
                    o[i*d+j]=acc;
                }
        }
    }
    delete[] S;
}

// ============================================================================
// CPU reference — causal attention
// ============================================================================
static void cpu_ref_causal(
    const half* Q, const half* K, const half* V, float* O_ref,
    int B, int H, int N, int d, float scale)
{
    float* S = new float[(size_t)N * N];
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int off = (b*H+h)*N*d;
            const half* q=Q+off; const half* k=K+off;
            const half* v=V+off; float* o=O_ref+off;

            for (int i=0;i<N;i++)
                for (int j=0;j<N;j++) {
                    if (j > i) { S[i*N+j] = -1e30f; continue; }
                    float acc=0.f;
                    for (int x=0;x<d;x++)
                        acc+=__half2float(q[i*d+x])*__half2float(k[j*d+x]);
                    S[i*N+j]=acc*scale;
                }
            for (int i=0;i<N;i++){
                float mx=-FLT_MAX;
                for (int j=0;j<=i;j++) mx=fmaxf(mx,S[i*N+j]);
                float sm=0.f;
                for (int j=0;j<=i;j++){S[i*N+j]=expf(S[i*N+j]-mx);sm+=S[i*N+j];}
                for (int j=0;j<=i;j++) S[i*N+j]/=sm;
                for (int j=i+1;j<N;j++) S[i*N+j]=0.f;
            }
            for (int i=0;i<N;i++)
                for (int j=0;j<d;j++){
                    float acc=0.f;
                    for (int x=0;x<=i;x++)
                        acc+=S[i*N+x]*__half2float(v[x*d+j]);
                    o[i*d+j]=acc;
                }
        }
    }
    delete[] S;
}

// ============================================================================
// Validação
// ============================================================================
static bool validate(
    const half* h_O, const float* h_ref,
    int B, int H, int N, int d,
    const char* label,
    float tol=5e-2f)
{
    float max_abs=0.f, max_rel=0.f;
    int wb=0,wh=0,wr=0,wc=0;
    for (int b=0;b<B;b++)
        for (int h=0;h<H;h++){
            int off=(b*H+h)*N*d;
            for (int i=0;i<N;i++)
                for (int j=0;j<d;j++){
                    float gpu=__half2float(h_O[off+i*d+j]);
                    float ref=h_ref[off+i*d+j];
                    float err=fabsf(gpu-ref);
                    float rel=fabsf(ref)>1e-6f?err/fabsf(ref):err;
                    if(err>max_abs){max_abs=err;max_rel=rel;wb=b;wh=h;wr=i;wc=j;}
                }
        }
    printf("  [%s] max_abs=%.6f  max_rel=%.4f%%  at[b=%d h=%d r=%d c=%d]  %s\n",
           label, max_abs, max_rel*100.f, wb, wh, wr, wc,
           max_abs<tol ? "PASS ✓" : "FAIL ✗");
    return max_abs < tol;
}

// ============================================================================
// Timer e benchmark robusto
// ============================================================================
struct Timer {
    cudaEvent_t t0,t1;
    Timer(){cudaEventCreate(&t0);cudaEventCreate(&t1);}
    ~Timer(){cudaEventDestroy(t0);cudaEventDestroy(t1);}
    void start(){cudaEventRecord(t0);}
    float stop(int r=1){
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms=0.f;cudaEventElapsedTime(&ms,t0,t1);return ms/r;
    }
};

struct BenchResult { double min_ms, med_ms, max_ms; };

static BenchResult bench(
    const half* dQ, const half* dK, const half* dV, half* dO,
    int B, int H, int N, float scale,
    AttentionMode mode,
    int outer=7, int inner=100)
{
    for (int i=0;i<3;i++) launch_v2(dQ,dK,dV,dO,B,H,N,scale,mode);
    cudaDeviceSynchronize();
    float samp[32]; if(outer>32)outer=32;
    Timer t;
    for (int r=0;r<outer;r++){
        t.start();
        for(int i=0;i<inner;i++) launch_v2(dQ,dK,dV,dO,B,H,N,scale,mode);
        samp[r]=t.stop(inner);
    }
    for(int i=0;i<outer-1;i++)
        for(int j=i+1;j<outer;j++)
            if(samp[j]<samp[i]){float tmp=samp[i];samp[i]=samp[j];samp[j]=tmp;}
    return {samp[0],samp[outer/2],samp[outer-1]};
}

// ============================================================================
// Ocupação
// ============================================================================
static int compute_blocks_per_sm(const cudaDeviceProp& prop, int regs)
{
    const int mr = prop.regsPerMultiprocessor / (regs * 32);
    const int ms = (int)(prop.sharedMemPerMultiprocessor / (size_t)SMEM_BYTES);
    const int mw = prop.maxThreadsPerMultiProcessor / 32;
    int bn=mr; if(ms<bn)bn=ms; if(mw<bn)bn=mw;
    return bn;
}

// ============================================================================
// run_case: valida + benchmarka full e causal
// ============================================================================
static void run_case(
    int B, int H, int N_req,
    bool do_val, int outer, int inner,
    int actual_regs, bool header)
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    const int   N     = ceil_div(N_req, TM) * TM;
    const float scale = 1.f / sqrtf(64.f);
    const int   total = B * H * N * D_FIXED;
    const size_t bqkv = (size_t)total * sizeof(half);

    // FLOPs algorítmicos:
    // full:   4 * N^2 * d
    // causal: ~2 * N^2 * d  (metade dos tiles descartados)
    const double flops_full   = (double)B * H * 4.0 * N * N * D_FIXED;
    const double flops_causal = flops_full * 0.5;

    const int bn   = compute_blocks_per_sm(prop, actual_regs);
    const int need = prop.multiProcessorCount * bn;
    const int nblk = (N/TM) * H * B;
    const float sat = fminf(100.f, 100.f * nblk / (float)need);

    if (header) {
        printf("Device: %s  SM%d%d  %d SMs\n",
               prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        printf("\n=== flash_v2  B=%d H=%d N=%d d=64 ===\n", B, H, N);
        printf("Smem/bloco: %d B (%.1f KB)  Regs: %d  "
               "Blocos/SM: %d  Sat: %.0f%%\n\n",
               SMEM_BYTES, SMEM_BYTES/1024.f, actual_regs, bn, sat);
    }

    half *dQ,*dK,*dV,*dO;
    check_cuda(cudaMalloc(&dQ,bqkv),"Q");
    check_cuda(cudaMalloc(&dK,bqkv),"K");
    check_cuda(cudaMalloc(&dV,bqkv),"V");
    check_cuda(cudaMalloc(&dO,bqkv),"O");

    half  *hQ=nullptr,*hK=nullptr,*hV=nullptr,*hO=nullptr;
    float *href_full=nullptr, *href_caus=nullptr;

    if (do_val && (double)B*H*N*N*D_FIXED > 3e8) {
        printf("[warn] CPU ref pulada\n"); do_val=false;
    }

    if (do_val) {
        hQ=(half*)malloc(bqkv); hK=(half*)malloc(bqkv);
        hV=(half*)malloc(bqkv); hO=(half*)malloc(bqkv);
        href_full=(float*)malloc((size_t)total*sizeof(float));
        href_caus=(float*)malloc((size_t)total*sizeof(float));
        std::srand(42);
        for(int i=0;i<total;i++){
            hQ[i]=__float2half(rand_val());
            hK[i]=__float2half(rand_val());
            hV[i]=__float2half(rand_val());
        }

        printf("[ref] CPU full... "); fflush(stdout);
        cpu_ref_full(hQ,hK,hV,href_full,B,H,N,D_FIXED,scale);
        printf("OK\n");

        printf("[ref] CPU causal... "); fflush(stdout);
        cpu_ref_causal(hQ,hK,hV,href_caus,B,H,N,D_FIXED,scale);
        printf("OK\n\n");

        check_cuda(cudaMemcpy(dQ,hQ,bqkv,cudaMemcpyHostToDevice),"H2D Q");
        check_cuda(cudaMemcpy(dK,hK,bqkv,cudaMemcpyHostToDevice),"H2D K");
        check_cuda(cudaMemcpy(dV,hV,bqkv,cudaMemcpyHostToDevice),"H2D V");

        // Validar full
        launch_v2(dQ,dK,dV,dO,B,H,N,scale,FULL);
        check_cuda(cudaDeviceSynchronize(),"sync full");
        check_cuda(cudaGetLastError(),"launch full");
        check_cuda(cudaMemcpy(hO,dO,bqkv,cudaMemcpyDeviceToHost),"D2H full");
        validate(hO,href_full,B,H,N,D_FIXED,"FULL   ",5e-2f);

        // Validar causal
        launch_v2(dQ,dK,dV,dO,B,H,N,scale,CAUSAL);
        check_cuda(cudaDeviceSynchronize(),"sync causal");
        check_cuda(cudaGetLastError(),"launch causal");
        check_cuda(cudaMemcpy(hO,dO,bqkv,cudaMemcpyDeviceToHost),"D2H causal");
        validate(hO,href_caus,B,H,N,D_FIXED,"CAUSAL ",5e-2f);
        printf("\n");
    } else {
        cudaMemset(dQ,0,bqkv);cudaMemset(dK,0,bqkv);cudaMemset(dV,0,bqkv);
    }

    // Benchmark full
    BenchResult bf = bench(dQ,dK,dV,dO,B,H,N,scale,FULL,  outer,inner);
    // Benchmark causal
    BenchResult bc = bench(dQ,dK,dV,dO,B,H,N,scale,CAUSAL,outer,inner);

    auto tf_full   = [&](double ms){ return flops_full  /(ms*1e-3)/1e12; };
    auto tf_causal = [&](double ms){ return flops_causal/(ms*1e-3)/1e12; };

    // Speedup causal vs full: para o mesmo trabalho útil,
    // causal deve ser ~2x mais rápido em latência
    const double lat_speedup = bf.med_ms / bc.med_ms;
    // Eficiência: TFLOPS_causal / TFLOPS_full (deveria ser ~1.0 se ambos
    // estiverem no mesmo ponto do roofline)
    const double eff = (float)tf_causal(bc.med_ms) / (float)tf_full(bf.med_ms);

    if (header) {
        printf("%-10s %-10s %-10s %-10s %-10s %-10s\n",
               "", "min_ms", "med_ms", "max_ms",
               "TFLOPS_hi", "TFLOPS_med");
        printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n",
               "FULL",
               bf.min_ms, bf.med_ms, bf.max_ms,
               tf_full(bf.max_ms), (float)tf_full(bf.med_ms));
        printf("%-10s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n",
               "CAUSAL",
               bc.min_ms, bc.med_ms, bc.max_ms,
               tf_causal(bc.max_ms), (float)tf_causal(bc.med_ms));
        printf("\n");
        printf("  Speedup causal/full (latencia): %.3fx\n", lat_speedup);
        printf("  Eficiencia TFLOPS causal/full:  %.3f\n", eff);
        printf("  Blocos: %d   Sat: %.0f%%\n\n", nblk, sat);
    } else {
        printf("%-5d %-5d %-6d | "
               "%-8.4f %-8.4f | "
               "%-8.4f %-8.4f | "
               "%-8.4f %-8.4f | "
               "%-6.2fx %-6.0f%%\n",
               B, H, N_req,
               bf.med_ms,  (float)tf_full(bf.med_ms),
               bc.med_ms,  (float)tf_causal(bc.med_ms),
               lat_speedup, sat);
    }

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    if(hQ)free(hQ);if(hK)free(hK);if(hV)free(hV);if(hO)free(hO);
    if(href_full)free(href_full);if(href_caus)free(href_caus);
}

// ============================================================================
// Sweep
// ============================================================================
static void run_sweep(int actual_regs)
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    const int bn   = compute_blocks_per_sm(prop, actual_regs);
    const int need = prop.multiProcessorCount * bn;

    // Referências do flatargs para comparação
    struct Ref { int B,H,N; double flat_med; };
    Ref refs[] = {
        {1,  1,  256, 0.0417},
        {1,  1,  512, 0.0801},
        {1,  1, 1024, 0.1322},
        {1,  4,  256, 0.0376},
        {1,  8,  256, 0.0503},
        {1, 12,  256, 0.0735},
        {1, 16,  256, 0.0736},
        {4,  8,  256, 0.1485},
        {8,  8,  256, 0.3196},
        {4, 12,  512, 0.6556},
        {4, 12, 1024, 2.6381},
        {8, 12,  512, 1.4109},
        {1, 16, 2048, 3.5005},
        {4,  8, 2048, 6.8282},
    };
    const int nref=(int)(sizeof(refs)/sizeof(refs[0]));

    printf("Device: %s  SM%d%d  %d SMs\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Regs: %d  Smem: %dB  Blocos/SM: %d  Para saturar: %d\n\n",
           actual_regs, SMEM_BYTES, bn, need);

    printf("=== Sweep flash_v2 (full + causal) vs flatargs ===\n\n");
    printf("%-5s %-5s %-6s | "
           "%-19s | "
           "%-19s | "
           "%-8s %-8s %-6s\n",
           "B","H","N",
           "FULL (med_ms / TFLOPS) ",
           "CAUSAL (med_ms / TFLOPS)",
           "flat-med","lat-spd","sat%");
    print_sep();

    for (int i=0;i<nref;i++) {
        Ref& r=refs[i];
        const int N=ceil_div(r.N,TM)*TM;
        const size_t bqkv=(size_t)r.B*r.H*N*D_FIXED*sizeof(half);
        if (bqkv>(size_t)2*1024*1024*1024ULL){
            printf("%-5d %-5d %-6d | SKIP\n",r.B,r.H,r.N); continue;
        }
        const float scale=1.f/sqrtf(64.f);
        const double flops_full=(double)r.B*r.H*4.0*N*N*D_FIXED;
        const double flops_caus=flops_full*0.5;
        const int nblk=(N/TM)*r.H*r.B;
        const float sat=fminf(100.f,100.f*nblk/(float)need);

        half *dQ,*dK,*dV,*dO;
        check_cuda(cudaMalloc(&dQ,bqkv),"Q");
        check_cuda(cudaMalloc(&dK,bqkv),"K");
        check_cuda(cudaMalloc(&dV,bqkv),"V");
        check_cuda(cudaMalloc(&dO,bqkv),"O");
        cudaMemset(dQ,0,bqkv);cudaMemset(dK,0,bqkv);cudaMemset(dV,0,bqkv);

        BenchResult bf=bench(dQ,dK,dV,dO,r.B,r.H,N,scale,FULL,  7,100);
        BenchResult bc=bench(dQ,dK,dV,dO,r.B,r.H,N,scale,CAUSAL,7,100);

        const double lat_spd = bf.med_ms / bc.med_ms;
        auto tf_f=[&](double ms){ return flops_full/(ms*1e-3)/1e12; };
        auto tf_c=[&](double ms){ return flops_caus/(ms*1e-3)/1e12; };

        printf("%-5d %-5d %-6d | "
               "%-8.4f %-8.4f | "
               "%-8.4f %-8.4f | "
               "%-8.4f %-8.2fx %-6.0f%%\n",
               r.B, r.H, r.N,
               bf.med_ms, tf_f(bf.med_ms),
               bc.med_ms, tf_c(bc.med_ms),
               r.flat_med, lat_spd, sat);

        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    }
    print_sep();
    printf("\n");
    printf("TFLOPS FULL   = FLOPs_full   / latencia\n");
    printf("TFLOPS CAUSAL = FLOPs_causal / latencia  (FLOPs_causal = 0.5 * FLOPs_full)\n");
    printf("lat-spd = latencia_full / latencia_causal\n");
    printf("  lat-spd ~2.0x = causal escalou perfeitamente\n");
    printf("  lat-spd ~1.0x = tiles mascarados nao ajudaram (overhead domina)\n");
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv)
{
    const int ACTUAL_REGS = 128;   // atualizar após ptxas

    printf("flash_v2 — SM75  d=64\n");
    printf("Features: A) causal mask  B) FP16 acc (compilado como FP32)  "
           "C) ping-pong K\n");
    printf("Smem: %d B (%.1f KB)   Regs esperados: %d\n\n",
           SMEM_BYTES, SMEM_BYTES/1024.f, ACTUAL_REGS);

    if (argc == 1) {
        run_case(1, 2,  64, true, 5, 100, ACTUAL_REGS, true);
        run_case(1, 4, 128, true, 5, 100, ACTUAL_REGS, true);
        run_sweep(ACTUAL_REGS);
        return 0;
    }

    const int B     = std::atoi(argv[1]);
    const int H     = std::atoi(argv[2]);
    const int N     = std::atoi(argv[3]);
    const int outer = argc>4 ? std::atoi(argv[4]) : 7;
    const int inner = argc>5 ? std::atoi(argv[5]) : 100;
    const bool val  = ((double)B*H*N*N*D_FIXED) < 3e8;

    run_case(B, H, N, val, outer, inner, ACTUAL_REGS, true);
    return 0;
}


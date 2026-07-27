// =============================================================================
// flash_fused_multihead.cu — Multi-head com grid 3D
// Grid: (N/16, num_heads, batch_size)
// =============================================================================

#include <cuda_fp16.h>
#include <mma.h>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include "warp_softmax.cuh"
#include "online_state.cuh"

using namespace nvcuda;

static constexpr int TM = 16;
static constexpr int TN = 16;
static constexpr int TK = 16;

// =============================================================================
struct MHAParams {
    const half* Q;   // [B, H, N, d]
    const half* K;
    const half* V;
    half*       O;
    int B, H, N, d;
    float scale;
};

// =============================================================================
__global__ void __launch_bounds__(32, 8)
flash_mha_kernel(MHAParams p)
{
    const int q_tile  = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch   = blockIdx.z;
    const int lane    = threadIdx.x & 31;
    const int q_base  = q_tile * TM;

    if (q_base >= p.N) return;

    // Offsets no tensor [B, H, N, d]
    const long long stride_b = (long long)p.H * p.N * p.d;
    const long long stride_h = (long long)p.N * p.d;
    const long long off      = (long long)batch * stride_b
                             + (long long)head_id * stride_h;

    const half* Q = p.Q + off;
    const half* K = p.K + off;
    const half* V = p.V + off;
    half*       O = p.O + off;

    const int num_kv = (p.N + TN - 1) / TN;
    const int num_d  = p.d / TK;

    extern __shared__ half smem[];
    half* sq = smem;
    half* sk = sq + TM * TK;
    half* sp = sk + TN * TK;
    half* sv = sp + TM * TN;

    for (int d_out = 0; d_out < num_d; d_out++) {

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc;
        wmma::fill_fragment(O_acc, 0.0f);
        OnlineState online;
        online_state_init(online);

        for (int kv = 0; kv < num_kv; kv++) {
            int kv_base = kv * TN;

            // ── S = Q @ K^T ──────────────────────────────────────────────────
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
            wmma::fill_fragment(S_acc, 0.0f);

            for (int d_in = 0; d_in < num_d; d_in++) {
                int db = d_in * TK;

                for (int i = lane; i < TM*TK; i += 32) {
                    int r = i/TK, c = i%TK;
                    sq[i] = ((q_base+r) < p.N)
                            ? Q[(q_base+r)*p.d + db + c]
                            : __float2half(0.f);
                }
                for (int i = lane; i < TN*TK; i += 32) {
                    int r = i/TK, c = i%TK;
                    sk[i] = ((kv_base+r) < p.N)
                            ? K[(kv_base+r)*p.d + db + c]
                            : __float2half(0.f);
                }
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Qf;
                wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> Kf;
                wmma::load_matrix_sync(Qf, sq, TK);
                wmma::load_matrix_sync(Kf, sk, TK);
                wmma::mma_sync(S_acc, Qf, Kf, S_acc);
                __syncwarp();
            }

            // ── Softmax + online rescaling ────────────────────────────────────
            RowMaxSum   ms   = warp_softmax_unnorm(S_acc, p.scale);
            BetaFactors beta = online_update(O_acc, online, ms);
            apply_beta_to_S(S_acc, beta);

            // ── Scatter S_acc → smem_P ────────────────────────────────────────
            {
                const int gid = lane>>2, tig = lane&3;
                const int ru  = gid,     rl  = gid + 8;
                const int c0  = tig*2,   c1  = tig*2+1;
                const int c4  = tig*2+8, c5  = tig*2+9;
                sp[ru*16+c0] = __float2half(S_acc.x[0]);
                sp[ru*16+c1] = __float2half(S_acc.x[1]);
                sp[ru*16+c4] = __float2half(S_acc.x[4]);
                sp[ru*16+c5] = __float2half(S_acc.x[5]);
                sp[rl*16+c0] = __float2half(S_acc.x[2]);
                sp[rl*16+c1] = __float2half(S_acc.x[3]);
                sp[rl*16+c4] = __float2half(S_acc.x[6]);
                sp[rl*16+c5] = __float2half(S_acc.x[7]);
            }
            __syncwarp();

            // ── P @ V → O_acc ─────────────────────────────────────────────────
            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Pf;
            wmma::load_matrix_sync(Pf, sp, 16);

            for (int i = lane; i < TN*TK; i += 32) {
                int r = i/TK, c = i%TK;
                sv[i] = ((kv_base+r) < p.N)
                        ? V[(kv_base+r)*p.d + d_out*TK + c]
                        : __float2half(0.f);
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> Vf;
            wmma::load_matrix_sync(Vf, sv, TK);
            wmma::mma_sync(O_acc, Pf, Vf, O_acc);
            __syncwarp();
        }

        // ── Finalizar e escrever O ────────────────────────────────────────────
        finalize_O(O_acc, online);

        float* scratch = reinterpret_cast<float*>(sp);
        wmma::store_matrix_sync(scratch, O_acc, 16, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane; i < TM*TK; i += 32) {
            int gr = q_base + i/TK;
            int gc = d_out*TK + i%TK;
            if (gr < p.N && gc < p.d)
                O[gr*p.d + gc] = __float2half(scratch[i]);
        }
        __syncwarp();
    }
}

// =============================================================================
static float bench_mha(MHAParams& p, int reps = 300)
{
    const int smem = 4 * TM * TK * (int)sizeof(half);
    cudaFuncSetAttribute(flash_mha_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    dim3 grid((p.N + TM - 1) / TM, p.H, p.B);
    dim3 block(32);

    for (int i = 0; i < 5; i++)
        flash_mha_kernel<<<grid, block, smem>>>(p);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < reps; i++)
        flash_mha_kernel<<<grid, block, smem>>>(p);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / reps;
}

// =============================================================================
static void print_sep(int w = 80) {
    for (int i = 0; i < w; i++) putchar('-');
    putchar('\n');
}

int main()
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("Device: %s  SM%d%d  %d SMs\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    // Ocupação teórica de V3
    const int regs_per_thread = 96;
    const int smem_per_block  = 4 * TM * TK * (int)sizeof(half);
    int max_by_regs = (prop.regsPerMultiprocessor / 32) / regs_per_thread;
    int max_by_smem = (int)(prop.sharedMemPerMultiprocessor
                            / (size_t)smem_per_block);
    int max_by_wrps = prop.maxThreadsPerMultiProcessor / 32;
    int bottleneck  = max_by_regs;
    if (max_by_smem < bottleneck) bottleneck = max_by_smem;
    if (max_by_wrps < bottleneck) bottleneck = max_by_wrps;
    int need_blocks = prop.multiProcessorCount * bottleneck;

    printf("Ocupação teórica V3:\n");
    printf("  Limite por regs : %d blocos/SM\n", max_by_regs);
    printf("  Limite por smem : %d blocos/SM\n", max_by_smem);
    printf("  Blocos p/ saturar device: %d\n\n", need_blocks);

    printf("=== Sweep Multi-Head  (B×H×(N/16) blocos no grid) ===\n\n");
    printf("%-5s %-5s %-5s %-5s | %-8s %-10s %-10s %-8s %-8s\n",
           "B", "H", "N", "d",
           "ms", "TFLOPS", "GB/s", "blocos", "sat%");
    print_sep();

    struct Cfg { int B, H, N, d; };
    Cfg cfgs[] = {
        // Sem batch/heads — reproduz o benchmark anterior
        {1,  1,  256, 64},
        {1,  1,  512, 64},
        {1,  1, 1024, 64},
        // Adicionar heads (GPT-2 small = 12 heads, d=64)
        {1,  4,  256, 64},
        {1,  8,  256, 64},
        {1, 12,  256, 64},
        {1, 16,  256, 64},
        // Batch
        {4,  8,  256, 64},
        {8,  8,  256, 64},
        // Realistas: batch + heads + seq maior
        {4, 12,  512, 64},
        {4, 12, 1024, 64},
        // d=128 (LLaMA style)
        {1,  8,  256, 128},
        {4,  8,  512, 128},
    };

    int ncfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
    for (int ci = 0; ci < ncfg; ci++) {
        Cfg& c = cfgs[ci];

        size_t bqkv = (size_t)c.B * c.H * c.N * c.d * sizeof(half);
        half *dQ, *dK, *dV, *dO;
        if (cudaMalloc(&dQ, bqkv) != cudaSuccess) {
            printf("%-5d %-5d %-5d %-5d | OOM\n", c.B, c.H, c.N, c.d);
            continue;
        }
        cudaMalloc(&dK, bqkv);
        cudaMalloc(&dV, bqkv);
        cudaMalloc(&dO, bqkv);
        cudaMemset(dQ, 0, bqkv);
        cudaMemset(dK, 0, bqkv);
        cudaMemset(dV, 0, bqkv);

        MHAParams p;
        p.Q = dQ; p.K = dK; p.V = dV; p.O = dO;
        p.B = c.B; p.H = c.H; p.N = c.N; p.d = c.d;
        p.scale = 1.0f / sqrtf((float)c.d);

        float ms = bench_mha(p, 300);

        // FLOPS: por cada (batch, head): 4 * N^2 * d
        double flops = (double)c.B * c.H * 4.0 * c.N * c.N * c.d;
        double tf    = flops / (ms * 1e-3) / 1e12;
        double gbps  = (4.0 * bqkv) / (ms * 1e-3) / 1e9;
        int    nblks = ((c.N + TM - 1) / TM) * c.H * c.B;
        float  sat   = 100.0f * nblks / (float)need_blocks;

        printf("%-5d %-5d %-5d %-5d | %-8.4f %-10.4f %-10.1f %-8d %-7.0f%%\n",
               c.B, c.H, c.N, c.d,
               ms, tf, gbps, nblks, sat > 100.f ? 100.f : sat);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    print_sep();
    printf("\nColuna 'sat%%' = blocos_no_grid / blocos_para_saturar_device.\n");
    printf("Quando sat%% >= 100%% o device está totalmente ocupado.\n");
    printf("Abaixo de ~50%% a latência de memória domina o tempo.\n");

    return 0;
}


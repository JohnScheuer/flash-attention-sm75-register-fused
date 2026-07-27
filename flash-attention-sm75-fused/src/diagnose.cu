// =============================================================================
// diagnose.cu — Isola cada gargalo do pipeline
// =============================================================================

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

#include "warp_softmax.cuh"
#include "online_state.cuh"

using namespace nvcuda;

static constexpr int TM = 16;
static constexpr int TN = 16;
static constexpr int TK = 16;

// =============================================================================
// V0: Bandwidth puro
// =============================================================================
__global__ void __launch_bounds__(32, 16)
kernel_v0_bandwidth(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int N, int d)
{
    const int lane   = threadIdx.x & 31;
    const int q_base = blockIdx.x * TM;
    const int num_d  = d / TK;

    extern __shared__ half smem[];
    half* sq = smem;
    half* sk = sq + TM * TK;
    half* sv = sk + TN * TK;

    for (int d_out = 0; d_out < num_d; d_out++) {
        for (int kv = 0; kv < (N / TN); kv++) {
            int kv_base = kv * TN;
            for (int d_in = 0; d_in < num_d; d_in++) {
                int db = d_in * TK;
                for (int i = lane; i < TM*TK; i += 32)
                    sq[i] = Q[(q_base + i/TK)*d + db + i%TK];
                for (int i = lane; i < TN*TK; i += 32)
                    sk[i] = K[(kv_base + i/TK)*d + db + i%TK];
                __syncwarp();
            }
            for (int i = lane; i < TN*TK; i += 32)
                sv[i] = V[(kv_base + i/TK)*d + d_out*TK + i%TK];
            __syncwarp();
        }
        for (int i = lane; i < TM*TK; i += 32)
            O[(q_base + i/TK)*d + d_out*TK + i%TK] = sq[i];
        __syncwarp();
    }
}

// =============================================================================
// V1: Bandwidth + WMMA QK^T
// =============================================================================
__global__ void __launch_bounds__(32, 8)
kernel_v1_wmma_qk(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int N, int d)
{
    const int lane   = threadIdx.x & 31;
    const int q_base = blockIdx.x * TM;
    const int num_d  = d / TK;
    const int num_kv = N / TN;

    extern __shared__ half smem[];
    half* sq = smem;
    half* sk = sq + TM * TK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;

    for (int d_out = 0; d_out < num_d; d_out++) {
        for (int kv = 0; kv < num_kv; kv++) {
            int kv_base = kv * TN;
            wmma::fill_fragment(S_acc, 0.0f);

            for (int d_in = 0; d_in < num_d; d_in++) {
                int db = d_in * TK;
                for (int i = lane; i < TM*TK; i += 32)
                    sq[i] = Q[(q_base + i/TK)*d + db + i%TK];
                for (int i = lane; i < TN*TK; i += 32)
                    sk[i] = K[(kv_base + i/TK)*d + db + i%TK];
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Qf;
                wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> Kf;
                wmma::load_matrix_sync(Qf, sq, TK);
                wmma::load_matrix_sync(Kf, sk, TK);
                wmma::mma_sync(S_acc, Qf, Kf, S_acc);
                __syncwarp();
            }
        }
    }

    // Dummy store: impede dead-code elimination
    if (lane == 0 && blockIdx.x == 0)
        O[0] = __float2half(S_acc.x[0]);
}

// =============================================================================
// V2: V1 + softmax em registradores
// =============================================================================
__global__ void __launch_bounds__(32, 8)
kernel_v2_with_softmax(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int N, int d, float scale)
{
    const int lane   = threadIdx.x & 31;
    const int q_base = blockIdx.x * TM;
    const int num_d  = d / TK;
    const int num_kv = N / TN;

    extern __shared__ half smem[];
    half* sq = smem;
    half* sk = sq + TM * TK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
    float sum_check = 0.0f;

    for (int d_out = 0; d_out < num_d; d_out++) {
        for (int kv = 0; kv < num_kv; kv++) {
            int kv_base = kv * TN;
            wmma::fill_fragment(S_acc, 0.0f);

            for (int d_in = 0; d_in < num_d; d_in++) {
                int db = d_in * TK;
                for (int i = lane; i < TM*TK; i += 32)
                    sq[i] = Q[(q_base + i/TK)*d + db + i%TK];
                for (int i = lane; i < TN*TK; i += 32)
                    sk[i] = K[(kv_base + i/TK)*d + db + i%TK];
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Qf;
                wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> Kf;
                wmma::load_matrix_sync(Qf, sq, TK);
                wmma::load_matrix_sync(Kf, sk, TK);
                wmma::mma_sync(S_acc, Qf, Kf, S_acc);
                __syncwarp();
            }

            RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);
            sum_check += ms.row_sum[0];
        }
    }

    if (lane == 0 && blockIdx.x == 0)
        O[0] = __float2half(sum_check);
}

// =============================================================================
// V3: Kernel completo
// =============================================================================
__global__ void __launch_bounds__(32, 8)
kernel_v3_full(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int N, int d, float scale)
{
    const int lane   = threadIdx.x & 31;
    const int q_base = blockIdx.x * TM;
    const int num_d  = d / TK;
    const int num_kv = N / TN;

    if (q_base >= N) return;

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
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
            wmma::fill_fragment(S_acc, 0.0f);

            for (int d_in = 0; d_in < num_d; d_in++) {
                int db = d_in * TK;
                for (int i = lane; i < TM*TK; i += 32)
                    sq[i] = Q[(q_base + i/TK)*d + db + i%TK];
                for (int i = lane; i < TN*TK; i += 32)
                    sk[i] = K[(kv_base + i/TK)*d + db + i%TK];
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Qf;
                wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> Kf;
                wmma::load_matrix_sync(Qf, sq, TK);
                wmma::load_matrix_sync(Kf, sk, TK);
                wmma::mma_sync(S_acc, Qf, Kf, S_acc);
                __syncwarp();
            }

            RowMaxSum ms  = warp_softmax_unnorm(S_acc, scale);
            BetaFactors b = online_update(O_acc, online, ms);
            apply_beta_to_S(S_acc, b);

            {
                const int gid = lane>>2, tig = lane&3;
                int ru = gid, rl = gid+8;
                int c0 = tig*2, c1 = tig*2+1, c4 = tig*2+8, c5 = tig*2+9;
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

            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Pf;
            wmma::load_matrix_sync(Pf, sp, 16);

            for (int i = lane; i < TN*TK; i += 32)
                sv[i] = V[(kv_base + i/TK)*d + d_out*TK + i%TK];
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> Vf;
            wmma::load_matrix_sync(Vf, sv, TK);
            wmma::mma_sync(O_acc, Pf, Vf, O_acc);
            __syncwarp();
        }

        finalize_O(O_acc, online);

        float* scratch = reinterpret_cast<float*>(sp);
        wmma::store_matrix_sync(scratch, O_acc, 16, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane; i < TM*TK; i += 32) {
            int gr = q_base + i/TK, gc = d_out*TK + i%TK;
            if (gr < N && gc < d)
                O[gr*d+gc] = __float2half(scratch[i]);
        }
        __syncwarp();
    }
}

// =============================================================================
// Timer
// =============================================================================
struct Timer {
    cudaEvent_t t0, t1;
    Timer()  { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    ~Timer() { cudaEventDestroy(t0); cudaEventDestroy(t1); }
    void  start()           { cudaEventRecord(t0); }
    float stop(int reps=1)  {
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms / reps;
    }
};

// =============================================================================
static void print_sep(int w = 78) {
    for (int i = 0; i < w; i++) putchar('-');
    putchar('\n');
}

// =============================================================================
int main(int argc, char** argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    int d = (argc > 2) ? atoi(argv[2]) : 64;
    N = ((N + 15) / 16) * 16;
    d = ((d + 15) / 16) * 16;

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("Device : %s  SM%d%d  %d SMs\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Peak BW: %.0f GB/s\n",
           prop.memoryBusWidth * (double)prop.memoryClockRate * 2.0 / 8.0 / 1e6);
    printf("Smem/SM: %zu KB\n\n",
           prop.sharedMemPerMultiprocessor / 1024);

    size_t bqkv = (size_t)N * d * sizeof(half);
    half *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bqkv); cudaMalloc(&dK, bqkv);
    cudaMalloc(&dV, bqkv); cudaMalloc(&dO, bqkv);
    cudaMemset(dQ, 0, bqkv); cudaMemset(dK, 0, bqkv); cudaMemset(dV, 0, bqkv);

    float  scale   = 1.0f / sqrtf((float)d);
    int    nblocks = N / TM;
    int    nreps   = 500;
    double flops   = 4.0 * N * N * d;        // 2×QKt + 2×PV
    double bytes   = 3.0 * (double)bqkv;     // lê Q,K,V

    int smem_v0 = 3 * TM * TK * (int)sizeof(half);
    int smem_v12= 2 * TM * TK * (int)sizeof(half);
    int smem_v3 = 4 * TM * TK * (int)sizeof(half);

    printf("=== Diagnóstico  N=%-4d  d=%-3d  blocos=%d ===\n\n",
           N, d, nblocks);

    printf("%-36s  %8s  %9s  %9s\n", "Versão", "ms", "TFLOPS", "GB/s");
    print_sep();

    Timer t;

#define BENCH(label, kern, smem, ...)                                       \
    do {                                                                     \
        for (int _i = 0; _i < 5; _i++)                                      \
            kern<<<nblocks, 32, smem>>>(__VA_ARGS__);                        \
        cudaDeviceSynchronize();                                             \
        t.start();                                                           \
        for (int _i = 0; _i < nreps; _i++)                                  \
            kern<<<nblocks, 32, smem>>>(__VA_ARGS__);                        \
        float _ms = t.stop(nreps);                                           \
        printf("%-36s  %8.4f  %9.4f  %9.1f\n",                             \
               label, _ms,                                                   \
               flops / (_ms * 1e-3) / 1e12,                                 \
               bytes / (_ms * 1e-3) / 1e9);                                  \
    } while(0)

    BENCH("V0: só memória (teto teórico)",
          kernel_v0_bandwidth, smem_v0,
          dQ, dK, dV, dO, N, d);

    BENCH("V1: + WMMA QK^T",
          kernel_v1_wmma_qk, smem_v12,
          dQ, dK, dV, dO, N, d);

    BENCH("V2: + softmax em registradores",
          kernel_v2_with_softmax, smem_v12,
          dQ, dK, dV, dO, N, d, scale);

    BENCH("V3: completo (+ PV + online)",
          kernel_v3_full, smem_v3,
          dQ, dK, dV, dO, N, d, scale);

#undef BENCH

    print_sep();

    // ── Diagnóstico de ocupação ───────────────────────────────────────────────
    printf("\n=== Ocupação  (V3: 96 regs/thread, smem=%d B/bloco) ===\n\n",
           smem_v3);

    int regs_per_thread  = 96;
    int max_by_regs = (prop.regsPerMultiprocessor / 32) / regs_per_thread;
    int max_by_smem = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem_v3);
    int max_by_wrps = prop.maxThreadsPerMultiProcessor / 32;
    int bottleneck  = max_by_regs;
    if (max_by_smem < bottleneck) bottleneck = max_by_smem;
    if (max_by_wrps < bottleneck) bottleneck = max_by_wrps;

    float occ = 100.0f * bottleneck / (float)max_by_wrps;

    printf("  Max blocos/SM por regs  : %2d  (%.0f%%)\n",
           max_by_regs, 100.0f * max_by_regs / max_by_wrps);
    printf("  Max blocos/SM por smem  : %2d  (%.0f%%)\n",
           max_by_smem, 100.0f * max_by_smem / max_by_wrps);
    printf("  Max blocos/SM por warps : %2d  (100%%)\n", max_by_wrps);
    printf("  Blocos ativos/SM        : %2d  (%.0f%% ocupação)\n",
           bottleneck, occ);
    printf("  SMs no device           : %2d\n", prop.multiProcessorCount);
    printf("  Blocos no grid (N=%d)  : %2d\n", N, nblocks);
    printf("  Blocos/SM médio         : %.2f\n",
           (float)nblocks / prop.multiProcessorCount);

    printf("\n  Diagnóstico:\n");

    if (nblocks < prop.multiProcessorCount)
        printf("  [!] GRID PEQUENO: só %d blocos para %d SMs.\n"
               "      %d SMs completamente ociosos!\n"
               "      Solução: adicionar dimensão batch/heads ao grid.\n",
               nblocks, prop.multiProcessorCount,
               prop.multiProcessorCount - nblocks);
    else if ((float)nblocks / prop.multiProcessorCount < (float)bottleneck * 0.5f)
        printf("  [!] BAIXA SATURAÇÃO: %.1f blocos/SM (máximo teórico=%d).\n"
               "      Aumente N ou adicione batch/heads.\n",
               (float)nblocks / prop.multiProcessorCount, bottleneck);
    else
        printf("  [OK] Grid adequado para saturar o device.\n");

    if (max_by_regs < 4)
        printf("  [!] REGISTER PRESSURE: %d regs/thread limita a %d blocos/SM.\n"
               "      Considere --maxrregcount=64 ou reformular o kernel.\n",
               regs_per_thread, max_by_regs);

    if (occ < 25.0f)
        printf("  [!] OCUPAÇÃO BAIXA: %.0f%%. Latência de memória não será\n"
               "      escondida. Kernel será memory-latency bound.\n", occ);

    printf("\n  Blocos necessários para saturar: >= %d\n",
           prop.multiProcessorCount * bottleneck);
    printf("  Com multi-head (H=8, B=4):       %d blocos\n",
           nblocks * 8 * 4);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}


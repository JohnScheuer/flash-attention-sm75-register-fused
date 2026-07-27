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

static constexpr int TM      = 16;
static constexpr int TN      = 16;
static constexpr int TK      = 16;
static constexpr int D_FIXED = 64;
static constexpr int D_TILES = D_FIXED / TK;   // 4

// ============================================================================
// Helpers
// ============================================================================
static inline void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error em %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

static inline float rand_val() {
    return (std::rand() / (float)RAND_MAX - 0.5f) * 0.5f;
}

static inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

static inline void print_sep(int w = 96) {
    for (int i = 0; i < w; i++) std::putchar('-');
    std::putchar('\n');
}

// ============================================================================
// Scatter accumulator -> tile half[16][16] em smem
// ============================================================================
__device__ __forceinline__
void scatter_S_to_half(
    const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S,
    half* __restrict__ smem_P)
{
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;

    const int ru = gid;
    const int rl = gid + 8;
    const int c0 = tig * 2;
    const int c1 = tig * 2 + 1;
    const int c4 = tig * 2 + 8;
    const int c5 = tig * 2 + 9;

    smem_P[ru * 16 + c0] = __float2half(S.x[0]);
    smem_P[ru * 16 + c1] = __float2half(S.x[1]);
    smem_P[ru * 16 + c4] = __float2half(S.x[4]);
    smem_P[ru * 16 + c5] = __float2half(S.x[5]);
    smem_P[rl * 16 + c0] = __float2half(S.x[2]);
    smem_P[rl * 16 + c1] = __float2half(S.x[3]);
    smem_P[rl * 16 + c4] = __float2half(S.x[6]);
    smem_P[rl * 16 + c5] = __float2half(S.x[7]);
}

// ============================================================================
// Kernel principal — flat args, int32 offsets, __restrict__ em tudo
//
// Mudanças vs flash_mha_reuse_qk:
//   1. Sem struct: todos os params são escalares ou ponteiros raw
//   2. Sem long long: offsets calculados em int32
//   3. __restrict__ explícito em todos os ponteiros
//   4. __launch_bounds__(32, 4) para guiar o alocador de regs
//
// Grid: (N/16, H, B)
// Block: 32 threads (1 warp)
// ============================================================================
__global__ void __launch_bounds__(32, 4)
flash_mha_flatargs_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int N,          // padded, múltiplo de 16
    int H,          // num heads
    float scale)    // 1/sqrt(d)
{
    // ── Identificação ─────────────────────────────────────────────────────────
    const int q_tile  = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch   = blockIdx.z;
    const int lane    = threadIdx.x & 31;

    const int q_base  = q_tile * TM;
    if (q_base >= N) return;

    // Offset para este (batch, head) em int32.
    // Máximo endereçável: B*H*N*d = 8*32*4096*64 = 67M elementos half
    // -> cabe confortavelmente em int32 (max ~2B).
    const int head_stride = N * D_FIXED;
    const int base_off    = (batch * H + head_id) * head_stride;

    const half* __restrict__ Qh = Q + base_off;
    const half* __restrict__ Kh = K + base_off;
    const half* __restrict__ Vh = V + base_off;
    half*       __restrict__ Oh = O + base_off;

    const int num_kv = N / TN;

    // smem: Q(512) | K(512) | P(512) | V(512) = 2048 B
    extern __shared__ half smem[];
    half* __restrict__ smem_Q = smem;
    half* __restrict__ smem_K = smem_Q + TM * TK;
    half* __restrict__ smem_P = smem_K + TN * TK;
    half* __restrict__ smem_V = smem_P + TM * TN;

    // 4 acumuladores de saída persistentes (um por coluna de saída d=0..63)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc[D_TILES];
    #pragma unroll
    for (int t = 0; t < D_TILES; t++)
        wmma::fill_fragment(O_acc[t], 0.0f);

    OnlineState state;
    online_state_init(state);

    // =========================================================================
    // Loop KV
    // =========================================================================
    for (int kv = 0; kv < num_kv; kv++) {
        const int kv_base = kv * TN;

        // ── 1) S = Q @ K^T  (4 chunks de d=16 acumulados) ────────────────────
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
        wmma::fill_fragment(S_acc, 0.0f);

        #pragma unroll
        for (int d_in = 0; d_in < D_TILES; d_in++) {
            const int db = d_in * TK;

            // Load Q tile [q_base : q_base+16, db : db+16]
            #pragma unroll
            for (int i = lane; i < TM * TK; i += 32) {
                smem_Q[i] = Qh[(q_base + i / TK) * D_FIXED + db + i % TK];
            }

            // Load K tile [kv_base : kv_base+16, db : db+16]
            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                smem_K[i] = Kh[(kv_base + i / TK) * D_FIXED + db + i % TK];
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Qf;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kf;
            wmma::load_matrix_sync(Qf, smem_Q, TK);
            wmma::load_matrix_sync(Kf, smem_K, TK);
            wmma::mma_sync(S_acc, Qf, Kf, S_acc);
            __syncwarp();
        }

        // ── 2) Softmax unnorm em registradores ────────────────────────────────
        RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);

        // ── 3) Online step: calcula alpha/beta, atualiza state ─────────────────
        OnlineStep step = online_step_update(state, ms);

        // ── 4) Beta em S_acc ───────────────────────────────────────────────────
        apply_beta_to_S(S_acc, step);

        // ── 5) P materializado UMA vez ─────────────────────────────────────────
        scatter_S_to_half(S_acc, smem_P);
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Pf;
        wmma::load_matrix_sync(Pf, smem_P, 16);

        // ── 6) Mesmo P reusado para os 4 tiles de saída ────────────────────────
        #pragma unroll
        for (int d_out = 0; d_out < D_TILES; d_out++) {
            // Reescala O antigo por alpha
            apply_alpha_to_O(O_acc[d_out], step);

            // Load V tile [kv_base : kv_base+16, d_out*16 : d_out*16+16]
            const int vc = d_out * TK;
            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                smem_V[i] = Vh[(kv_base + i / TK) * D_FIXED + vc + i % TK];
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> Vf;
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

        // reusa [smem_P | smem_V] = 1024 B como float scratch[256]
        float* scratch = reinterpret_cast<float*>(smem_P);
        wmma::store_matrix_sync(scratch, O_acc[d_out], 16, wmma::mem_row_major);
        __syncwarp();

        const int oc = d_out * TK;
        #pragma unroll
        for (int i = lane; i < TM * TK; i += 32) {
            Oh[(q_base + i / TK) * D_FIXED + oc + i % TK] =
                __float2half(scratch[i]);
        }
        __syncwarp();
    }
}

// ============================================================================
// Launcher
// ============================================================================
static void flatargs_launch(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int B, int H, int N,
    float scale,
    cudaStream_t stream = 0)
{
    const int smem_bytes = 4 * 16 * 16 * (int)sizeof(half);

    cudaFuncSetAttribute(
        flash_mha_flatargs_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);

    dim3 grid(N / 16, H, B);
    dim3 block(32);

    flash_mha_flatargs_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, N, H, scale);
}

// ============================================================================
// CPU reference
// ============================================================================
static void cpu_ref(
    const half* Q, const half* K, const half* V, float* O_ref,
    int B, int H, int N, int d, float scale)
{
    float* S = new float[N * N];

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int off = (b * H + h) * N * d;
            const half* q = Q + off;
            const half* k = K + off;
            const half* v = V + off;
            float*      o = O_ref + off;

            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) {
                    float acc = 0.f;
                    for (int x = 0; x < d; x++)
                        acc += __half2float(q[i*d+x]) * __half2float(k[j*d+x]);
                    S[i*N+j] = acc * scale;
                }

            for (int i = 0; i < N; i++) {
                float mx = -FLT_MAX;
                for (int j = 0; j < N; j++) mx = fmaxf(mx, S[i*N+j]);
                float sm = 0.f;
                for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-mx); sm += S[i*N+j]; }
                for (int j = 0; j < N; j++) S[i*N+j] /= sm;
            }

            for (int i = 0; i < N; i++)
                for (int j = 0; j < d; j++) {
                    float acc = 0.f;
                    for (int x = 0; x < N; x++)
                        acc += S[i*N+x] * __half2float(v[x*d+j]);
                    o[i*d+j] = acc;
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
    float tol = 5e-2f)
{
    float max_abs = 0.f, max_rel = 0.f;
    int wb = 0, wh = 0, wr = 0, wc = 0;

    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            int off = (b*H+h)*N*d;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < d; j++) {
                    float gpu = __half2float(h_O[off+i*d+j]);
                    float ref = h_ref[off+i*d+j];
                    float err = fabsf(gpu - ref);
                    float rel = fabsf(ref) > 1e-6f ? err/fabsf(ref) : err;
                    if (err > max_abs) {
                        max_abs = err; max_rel = rel;
                        wb=b; wh=h; wr=i; wc=j;
                    }
                }
        }

    printf("  Max abs error : %.6f  at [b=%d h=%d r=%d c=%d]\n",
           max_abs, wb, wh, wr, wc);
    printf("  Max rel error : %.4f%%\n", max_rel * 100.f);
    return max_abs < tol;
}

// ============================================================================
// Timer
// ============================================================================
struct Timer {
    cudaEvent_t t0, t1;
    Timer()  { cudaEventCreate(&t0); cudaEventCreate(&t1); }
    ~Timer() { cudaEventDestroy(t0); cudaEventDestroy(t1); }

    void start() { cudaEventRecord(t0); }

    // Retorna ms por iteração
    float stop(int reps = 1) {
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms / reps;
    }
};

// ============================================================================
// Benchmark robusto: min / median / max de K rodadas
// ============================================================================
struct BenchResult {
    double min_ms, med_ms, max_ms;
};

static BenchResult benchmark(
    const half* dQ, const half* dK, const half* dV, half* dO,
    int B, int H, int N, float scale,
    int outer = 7, int inner = 100)
{
    // warm-up
    for (int i = 0; i < 3; i++)
        flatargs_launch(dQ, dK, dV, dO, B, H, N, scale);
    cudaDeviceSynchronize();

    float samples[32];
    if (outer > 32) outer = 32;

    Timer t;
    for (int r = 0; r < outer; r++) {
        t.start();
        for (int i = 0; i < inner; i++)
            flatargs_launch(dQ, dK, dV, dO, B, H, N, scale);
        samples[r] = t.stop(inner);
    }

    // sort simples (outer <= 32)
    for (int i = 0; i < outer - 1; i++)
        for (int j = i + 1; j < outer; j++)
            if (samples[j] < samples[i]) {
                float tmp = samples[i];
                samples[i] = samples[j];
                samples[j] = tmp;
            }

    BenchResult res;
    res.min_ms = samples[0];
    res.med_ms = samples[outer / 2];
    res.max_ms = samples[outer - 1];
    return res;
}

// ============================================================================
// Ocupação real baseada no ptxas atual
// ============================================================================
static void print_occupancy(const cudaDeviceProp& prop, int regs_per_thread)
{
    const int smem_bytes    = 4 * 16 * 16 * (int)sizeof(half);
    const int max_by_regs   = prop.regsPerMultiprocessor / (regs_per_thread * 32);
    const int max_by_smem   = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem_bytes);
    const int max_by_warps  = prop.maxThreadsPerMultiProcessor / 32;
    int bottleneck = max_by_regs;
    if (max_by_smem  < bottleneck) bottleneck = max_by_smem;
    if (max_by_warps < bottleneck) bottleneck = max_by_warps;

    const float occ_pct = 100.f * bottleneck / (float)max_by_warps;
    const int need_blocks = prop.multiProcessorCount * bottleneck;

    printf("  Regs/thread         : %d\n", regs_per_thread);
    printf("  Max blocos/SM regs  : %d\n", max_by_regs);
    printf("  Max blocos/SM smem  : %d\n", max_by_smem);
    printf("  Max blocos/SM warps : %d\n", max_by_warps);
    printf("  Blocos ativos/SM    : %d  (%.0f%% ocupacao)\n", bottleneck, occ_pct);
    printf("  Blocos p/ saturar   : %d\n", need_blocks);
}

// ============================================================================
// Roda um caso com validação opcional e benchmark robusto
// ============================================================================
static void run_case(
    int B, int H, int N_req, int d,
    bool do_validate,
    int outer, int inner,
    int actual_regs,   // passado do ptxas
    bool print_header)
{
    if (d != 64) {
        printf("Esta versao suporta apenas d=64.\n");
        return;
    }

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    const int   N     = ceil_div(N_req, 16) * 16;
    const float scale = 1.f / sqrtf((float)d);
    const int   total = B * H * N * d;
    const size_t bqkv = (size_t)total * sizeof(half);
    const double flops = (double)B * H * 4.0 * N * N * d;

    if (print_header) {
        printf("Device: %s  SM%d%d  %d SMs\n",
               prop.name, prop.major, prop.minor,
               prop.multiProcessorCount);
        printf("\n=== flash_mha_flatargs ===\n");
        printf("B=%d H=%d N_req=%d N_pad=%d d=%d\n", B, H, N_req, N, d);
        printf("Grid = (%d, %d, %d) = %d blocos\n\n",
               N/16, H, B, (N/16)*H*B);
        print_occupancy(prop, actual_regs);
        printf("\n");
    }

    half  *dQ, *dK, *dV, *dO;
    check_cuda(cudaMalloc(&dQ, bqkv), "malloc Q");
    check_cuda(cudaMalloc(&dK, bqkv), "malloc K");
    check_cuda(cudaMalloc(&dV, bqkv), "malloc V");
    check_cuda(cudaMalloc(&dO, bqkv), "malloc O");

    half  *hQ = nullptr, *hK = nullptr, *hV = nullptr, *hO = nullptr;
    float *href = nullptr;

    const double cpu_work = (double)B * H * N * N * d;

    if (do_validate && cpu_work > 3e8) {
        printf("[warn] Caso muito grande para validacao CPU. Pulando.\n");
        do_validate = false;
    }

    if (do_validate) {
        hQ   = (half* )malloc(bqkv);
        hK   = (half* )malloc(bqkv);
        hV   = (half* )malloc(bqkv);
        hO   = (half* )malloc(bqkv);
        href = (float*)malloc((size_t)total * sizeof(float));

        std::srand(42);
        for (int i = 0; i < total; i++) {
            hQ[i] = __float2half(rand_val());
            hK[i] = __float2half(rand_val());
            hV[i] = __float2half(rand_val());
        }

        printf("[1/3] CPU reference... "); fflush(stdout);
        cpu_ref(hQ, hK, hV, href, B, H, N, d, scale);
        printf("OK\n");

        check_cuda(cudaMemcpy(dQ, hQ, bqkv, cudaMemcpyHostToDevice), "H2D Q");
        check_cuda(cudaMemcpy(dK, hK, bqkv, cudaMemcpyHostToDevice), "H2D K");
        check_cuda(cudaMemcpy(dV, hV, bqkv, cudaMemcpyHostToDevice), "H2D V");

        printf("[2/3] GPU kernel... "); fflush(stdout);
        flatargs_launch(dQ, dK, dV, dO, B, H, N, scale);
        check_cuda(cudaDeviceSynchronize(), "sync");
        check_cuda(cudaGetLastError(), "launch");
        printf("OK\n");

        check_cuda(cudaMemcpy(hO, dO, bqkv, cudaMemcpyDeviceToHost), "D2H O");

        printf("[3/3] Validacao...\n");
        bool ok = validate(hO, href, B, H, N, d, 5e-2f);
        printf("  Resultado: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");
    } else {
        check_cuda(cudaMemset(dQ, 0, bqkv), "memset Q");
        check_cuda(cudaMemset(dK, 0, bqkv), "memset K");
        check_cuda(cudaMemset(dV, 0, bqkv), "memset V");
    }

    BenchResult br = benchmark(dQ, dK, dV, dO, B, H, N, scale, outer, inner);

    const int smem_bytes    = 4 * 16 * 16 * (int)sizeof(half);
    const int max_by_regs   = prop.regsPerMultiprocessor / (actual_regs * 32);
    const int max_by_smem   = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem_bytes);
    const int max_by_warps  = prop.maxThreadsPerMultiProcessor / 32;
    int bottleneck = max_by_regs;
    if (max_by_smem  < bottleneck) bottleneck = max_by_smem;
    if (max_by_warps < bottleneck) bottleneck = max_by_warps;
    const int need_blocks = prop.multiProcessorCount * bottleneck;
    const int nblocks     = (N/16) * H * B;
    const float sat       = fminf(100.f, 100.f * nblocks / (float)need_blocks);

    auto tflops = [&](double ms) {
        return flops / (ms * 1e-3) / 1e12;
    };

    if (print_header) {
        printf("=== Benchmark (outer=%d inner=%d) ===\n", outer, inner);
        printf("  %-12s %-10s %-10s %-10s\n", "métrica", "min", "median", "max");
        printf("  %-12s %-10.4f %-10.4f %-10.4f\n",
               "ms",     br.min_ms, br.med_ms, br.max_ms);
        printf("  %-12s %-10.4f %-10.4f %-10.4f\n",
               "TFLOPS", tflops(br.max_ms), tflops(br.med_ms), tflops(br.min_ms));
        printf("  Blocos grid : %d   sat: %.0f%%\n\n", nblocks, sat);
    } else {
        printf("%-5d %-5d %-6d | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-7d %-7.0f%%\n",
               B, H, N_req,
               br.min_ms, br.med_ms, br.max_ms,
               tflops(br.max_ms), tflops(br.med_ms), tflops(br.min_ms),
               nblocks, sat);
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    if (hQ)   free(hQ);
    if (hK)   free(hK);
    if (hV)   free(hV);
    if (hO)   free(hO);
    if (href) free(href);
}

// ============================================================================
// Sweep principal
// ============================================================================
static void run_sweep(int actual_regs)
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    const int smem_bytes    = 4 * 16 * 16 * (int)sizeof(half);
    const int max_by_regs   = prop.regsPerMultiprocessor / (actual_regs * 32);
    const int max_by_smem   = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem_bytes);
    const int max_by_warps  = prop.maxThreadsPerMultiProcessor / 32;
    int bottleneck = max_by_regs;
    if (max_by_smem  < bottleneck) bottleneck = max_by_smem;
    if (max_by_warps < bottleneck) bottleneck = max_by_warps;
    const int need_blocks = prop.multiProcessorCount * bottleneck;

    printf("Device: %s  SM%d%d  %d SMs\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Regs/thread: %d  |  Blocos/SM: %d  |  Blocos para saturar: %d\n\n",
           actual_regs, bottleneck, need_blocks);

    printf("=== Sweep flash_mha_flatargs (d=64) ===\n");
    printf("Benchmark: 7 rodadas × 100 iteracoes, reporta min/median/max\n\n");

    printf("%-5s %-5s %-6s | "
           "%-27s | "
           "%-27s | "
           "%-7s %-7s\n",
           "B", "H", "N",
           "       latência (ms)       ",
           "       TFLOPS              ",
           "blocos", "sat%");
    printf("%-5s %-5s %-6s | "
           "%-8s %-8s %-8s | "
           "%-8s %-8s %-8s | "
           "%-7s %-7s\n",
           "", "", "",
           "min", "median", "max",
           "max", "median", "min",
           "", "");
    print_sep();

    struct Cfg { int B, H, N; };
    Cfg cfgs[] = {
        // Grid pequeno: mostra onde latência domina
        {1,  1,  256},
        {1,  1,  512},
        {1,  1, 1024},
        // Escalando heads
        {1,  4,  256},
        {1,  8,  256},
        {1, 12,  256},
        {1, 16,  256},
        // Escalando batch
        {4,  8,  256},
        {8,  8,  256},
        // Saturação plena
        {4, 12,  512},
        {4, 12, 1024},
        {8, 12,  512},
        // Sequências longas
        {1, 16, 2048},
        {4,  8, 2048},
    };

    const int ncfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
    for (int i = 0; i < ncfg; i++) {
        // Pular casos que excedam N padded confortavelmente
        // (cpu_ref pula automaticamente, mas vamos evitar OOM de device)
        const size_t bqkv = (size_t)cfgs[i].B * cfgs[i].H *
                            (size_t)ceil_div(cfgs[i].N,16)*16 * 64 * sizeof(half);
        if (bqkv > (size_t)2 * 1024 * 1024 * 1024ULL) {
            printf("%-5d %-5d %-6d | SKIP (>2GB)\n",
                   cfgs[i].B, cfgs[i].H, cfgs[i].N);
            continue;
        }
        run_case(cfgs[i].B, cfgs[i].H, cfgs[i].N, 64,
                 false,   // sem validacao no sweep
                 7, 100,  // outer=7, inner=100
                 actual_regs,
                 false);
    }

    print_sep();
    printf("\nNotas:\n");
    printf("  TFLOPS reportado = FLOPs algoritmicos / tempo\n");
    printf("  sat%%  = blocos_no_grid / blocos_para_saturar (regs=%d)\n",
           actual_regs);
    printf("  Para sat >= 100%%, o device esta totalmente ocupado.\n");
}

// ============================================================================
// Main
//
//   ./flash_mha_flatargs
//       -> sanity check + sweep completo
//
//   ./flash_mha_flatargs B H N [outer [inner]]
//       -> benchmark de um caso, com validacao se pequeno
// ============================================================================
int main(int argc, char** argv)
{
    // Valor real do ptxas — ATUALIZE após ver o log de compilação.
    // Será impresso pelo programa para você conferir no output.
    // Depois de compilar, pegue o numero de "Used X registers" e
    // substitua aqui + recompile para o sweep ficar correto.
    const int ACTUAL_REGS = 255;   // <-- atualizar se ptxas mudar

    printf("flash_mha_flatargs — SM75  d=64\n");
    printf("Regs/thread esperados: %d  (verifique o log ptxas)\n\n",
           ACTUAL_REGS);

    if (argc == 1) {
        // sanity check pequeno com validacao
        run_case(1, 2, 64, 64, true, 5, 100, ACTUAL_REGS, true);
        run_case(1, 4, 128, 64, true, 5, 100, ACTUAL_REGS, true);
        run_sweep(ACTUAL_REGS);
        return 0;
    }

    const int B     = (argc > 1) ? std::atoi(argv[1]) : 1;
    const int H     = (argc > 2) ? std::atoi(argv[2]) : 1;
    const int N     = (argc > 3) ? std::atoi(argv[3]) : 256;
    const int outer = (argc > 4) ? std::atoi(argv[4]) : 7;
    const int inner = (argc > 5) ? std::atoi(argv[5]) : 100;

    const bool do_val = ((double)B * H * N * N * 64) < 3e8;

    run_case(B, H, N, 64, do_val, outer, inner, ACTUAL_REGS, true);
    return 0;
}


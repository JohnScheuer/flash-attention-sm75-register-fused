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
static constexpr int D_TILES = D_FIXED / TK;   // 4 tiles de 16 colunas
static constexpr int PASS_TILES = 2;            // tiles vivos por pass

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
static inline void print_sep(int w = 100) {
    for (int i = 0; i < w; i++) std::putchar('-');
    std::putchar('\n');
}

// ============================================================================
// Scatter accumulator -> smem half tile
// ============================================================================
__device__ __forceinline__
void scatter_S_to_half(
    const wmma::fragment<wmma::accumulator, 16, 16, 16, float>& S,
    half* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;
    const int ru   = gid;
    const int rl   = gid + 8;
    const int c0   = tig * 2,     c1 = tig * 2 + 1;
    const int c4   = tig * 2 + 8, c5 = tig * 2 + 9;

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
// Kernel 2-tiles
//
// Diferença do kernel 4-tiles (flatargs):
//   - Mantém apenas 2 acumuladores O_acc[2] vivos ao mesmo tempo
//   - Faz 2 passes sobre os kv_tiles:
//       pass 0: acumula em O_acc[0..1]  (colunas  0..31)
//       pass 1: acumula em O_acc[0..1]  (colunas 32..63)
//   - QK^T é recomputado 2× por q_tile (vs 1× no 4-tiles e 4× no original)
//   - Esperamos que os regs caiam para ~130-170
//
// Grid: (N/16, H, B)
// Block: 32 threads (1 warp)
// ============================================================================
__global__ void __launch_bounds__(32, 4)
flash_mha_2tiles_kernel(
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

    const int num_kv = N / TN;

    // smem: Q(512) | K(512) | P(512) | V(512) = 2048 B
    extern __shared__ half smem[];
    half* __restrict__ smem_Q = smem;
    half* __restrict__ smem_K = smem_Q + TM * TK;
    half* __restrict__ smem_P = smem_K + TN * TK;
    half* __restrict__ smem_V = smem_P + TM * TN;

    // =========================================================================
    // 2 passes: cada pass processa PASS_TILES=2 colunas de saída
    // pass_id=0 -> d_out cols 0..31
    // pass_id=1 -> d_out cols 32..63
    // =========================================================================
    #pragma unroll
    for (int pass_id = 0; pass_id < D_TILES / PASS_TILES; pass_id++) {
        const int d_out_base = pass_id * PASS_TILES;

        // 2 acumuladores vivos neste pass
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc[PASS_TILES];
        #pragma unroll
        for (int t = 0; t < PASS_TILES; t++)
            wmma::fill_fragment(O_acc[t], 0.0f);

        OnlineState state;
        online_state_init(state);

        // ── Loop KV ──────────────────────────────────────────────────────────
        for (int kv = 0; kv < num_kv; kv++) {
            const int kv_base = kv * TN;

            // ── 1) S = Q @ K^T ────────────────────────────────────────────────
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
            wmma::fill_fragment(S_acc, 0.0f);

            #pragma unroll
            for (int d_in = 0; d_in < D_TILES; d_in++) {
                const int db = d_in * TK;

                #pragma unroll
                for (int i = lane; i < TM * TK; i += 32)
                    smem_Q[i] = Qh[(q_base + i/TK) * D_FIXED + db + i%TK];

                #pragma unroll
                for (int i = lane; i < TN * TK; i += 32)
                    smem_K[i] = Kh[(kv_base + i/TK) * D_FIXED + db + i%TK];
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Qf;
                wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> Kf;
                wmma::load_matrix_sync(Qf, smem_Q, TK);
                wmma::load_matrix_sync(Kf, smem_K, TK);
                wmma::mma_sync(S_acc, Qf, Kf, S_acc);
                __syncwarp();
            }

            // ── 2) Softmax unnorm ─────────────────────────────────────────────
            RowMaxSum ms = warp_softmax_unnorm(S_acc, scale);

            // ── 3) Online step (UMA vez por kv tile, compartilhado) ───────────
            OnlineStep step = online_step_update(state, ms);

            // ── 4) Beta em S_acc ──────────────────────────────────────────────
            apply_beta_to_S(S_acc, step);

            // ── 5) P materializado UMA vez ────────────────────────────────────
            scatter_S_to_half(S_acc, smem_P);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> Pf;
            wmma::load_matrix_sync(Pf, smem_P, 16);

            // ── 6) Reusa P para os 2 tiles deste pass ────────────────────────
            #pragma unroll
            for (int t = 0; t < PASS_TILES; t++) {
                apply_alpha_to_O(O_acc[t], step);

                const int vc = (d_out_base + t) * TK;
                #pragma unroll
                for (int i = lane; i < TN * TK; i += 32)
                    smem_V[i] = Vh[(kv_base + i/TK) * D_FIXED + vc + i%TK];
                __syncwarp();

                wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> Vf;
                wmma::load_matrix_sync(Vf, smem_V, TK);
                wmma::mma_sync(O_acc[t], Pf, Vf, O_acc[t]);
                __syncwarp();
            }
        }

        // ── Finaliza e escreve os 2 tiles deste pass ─────────────────────────
        #pragma unroll
        for (int t = 0; t < PASS_TILES; t++) {
            finalize_O(O_acc[t], state);

            float* scratch = reinterpret_cast<float*>(smem_P);
            wmma::store_matrix_sync(scratch, O_acc[t], 16, wmma::mem_row_major);
            __syncwarp();

            const int oc = (d_out_base + t) * TK;
            #pragma unroll
            for (int i = lane; i < TM * TK; i += 32)
                Oh[(q_base + i/TK) * D_FIXED + oc + i%TK] =
                    __float2half(scratch[i]);
            __syncwarp();
        }
    }
}

// ============================================================================
// Launcher
// ============================================================================
static void launch_2tiles(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half*       __restrict__ O,
    int B, int H, int N,
    float scale,
    cudaStream_t stream = 0)
{
    const int smem_bytes = 4 * 16 * 16 * (int)sizeof(half);
    cudaFuncSetAttribute(flash_mha_2tiles_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    dim3 grid(N / 16, H, B);
    dim3 block(32);
    flash_mha_2tiles_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K, V, O, N, H, scale);
}

// ============================================================================
// CPU reference
// ============================================================================
static void cpu_ref(
    const half* Q, const half* K, const half* V, float* O_ref,
    int B, int H, int N, int d, float scale)
{
    float* S = new float[(size_t)N * N];
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
                for (int j = 0; j < N; j++) {
                    S[i*N+j] = expf(S[i*N+j] - mx);
                    sm += S[i*N+j];
                }
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
    int B, int H, int N, int d, float tol = 5e-2f)
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
                        max_abs=err; max_rel=rel;
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
    void  start()          { cudaEventRecord(t0); }
    float stop(int reps=1) {
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms / reps;
    }
};

// ============================================================================
// Benchmark robusto
// ============================================================================
struct BenchResult { double min_ms, med_ms, max_ms; };

static BenchResult bench(
    const half* dQ, const half* dK, const half* dV, half* dO,
    int B, int H, int N, float scale,
    int outer = 7, int inner = 100)
{
    for (int i = 0; i < 3; i++)
        launch_2tiles(dQ, dK, dV, dO, B, H, N, scale);
    cudaDeviceSynchronize();

    float samp[32];
    if (outer > 32) outer = 32;
    Timer t;
    for (int r = 0; r < outer; r++) {
        t.start();
        for (int i = 0; i < inner; i++)
            launch_2tiles(dQ, dK, dV, dO, B, H, N, scale);
        samp[r] = t.stop(inner);
    }
    for (int i = 0; i < outer-1; i++)
        for (int j = i+1; j < outer; j++)
            if (samp[j] < samp[i]) { float tmp=samp[i]; samp[i]=samp[j]; samp[j]=tmp; }

    return { samp[0], samp[outer/2], samp[outer-1] };
}

// ============================================================================
// Ocupação
// ============================================================================
static void print_occ(const cudaDeviceProp& prop, int regs)
{
    const int smem  = 4 * 16 * 16 * (int)sizeof(half);
    const int mr    = prop.regsPerMultiprocessor / (regs * 32);
    const int ms    = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem);
    const int mw    = prop.maxThreadsPerMultiProcessor / 32;
    int bn = mr; if (ms < bn) bn = ms; if (mw < bn) bn = mw;
    printf("  Regs/thread       : %d\n", regs);
    printf("  Max blocos/SM     : %d  (regs=%d  smem=%d  warps=%d)\n", bn, mr, ms, mw);
    printf("  Ocupacao estimada : %.0f%%\n", 100.f * bn / mw);
    printf("  Blocos p/saturar  : %d\n", prop.multiProcessorCount * bn);
}

// ============================================================================
// run_case
// ============================================================================
static void run_case(
    int B, int H, int N_req,
    bool do_val, int outer, int inner,
    int actual_regs, bool header)
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    const int   N     = ceil_div(N_req, 16) * 16;
    const float scale = 1.f / sqrtf(64.f);
    const int   total = B * H * N * 64;
    const size_t bqkv = (size_t)total * sizeof(half);
    const double flops = (double)B * H * 4.0 * N * N * 64;

    if (header) {
        printf("Device: %s  SM%d%d  %d SMs\n",
               prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        printf("\n=== flash_mha_2tiles  B=%d H=%d N=%d d=64 ===\n", B, H, N);
        printf("Grid = (%d, %d, %d) = %d blocos\n\n",
               N/16, H, B, (N/16)*H*B);
        print_occ(prop, actual_regs);
        printf("\n");
    }

    half  *dQ, *dK, *dV, *dO;
    check_cuda(cudaMalloc(&dQ, bqkv), "Q");
    check_cuda(cudaMalloc(&dK, bqkv), "K");
    check_cuda(cudaMalloc(&dV, bqkv), "V");
    check_cuda(cudaMalloc(&dO, bqkv), "O");

    half  *hQ=nullptr, *hK=nullptr, *hV=nullptr, *hO=nullptr;
    float *href=nullptr;

    if (do_val && (double)B*H*N*N*64 > 3e8) {
        printf("[warn] CPU ref pulada (caso grande)\n");
        do_val = false;
    }

    if (do_val) {
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
        printf("[1/3] CPU ref... "); fflush(stdout);
        cpu_ref(hQ, hK, hV, href, B, H, N, 64, scale);
        printf("OK\n");

        check_cuda(cudaMemcpy(dQ, hQ, bqkv, cudaMemcpyHostToDevice), "H2D Q");
        check_cuda(cudaMemcpy(dK, hK, bqkv, cudaMemcpyHostToDevice), "H2D K");
        check_cuda(cudaMemcpy(dV, hV, bqkv, cudaMemcpyHostToDevice), "H2D V");

        printf("[2/3] GPU... "); fflush(stdout);
        launch_2tiles(dQ, dK, dV, dO, B, H, N, scale);
        check_cuda(cudaDeviceSynchronize(), "sync");
        check_cuda(cudaGetLastError(), "launch");
        printf("OK\n");

        check_cuda(cudaMemcpy(hO, dO, bqkv, cudaMemcpyDeviceToHost), "D2H O");
        printf("[3/3] Validacao...\n");
        bool ok = validate(hO, href, B, H, N, 64, 5e-2f);
        printf("  Resultado: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");
    } else {
        cudaMemset(dQ,0,bqkv); cudaMemset(dK,0,bqkv); cudaMemset(dV,0,bqkv);
    }

    BenchResult br = bench(dQ, dK, dV, dO, B, H, N, scale, outer, inner);

    const int smem      = 4 * 16 * 16 * (int)sizeof(half);
    const int mr        = prop.regsPerMultiprocessor / (actual_regs * 32);
    const int ms        = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem);
    const int mw        = prop.maxThreadsPerMultiProcessor / 32;
    int bn = mr; if (ms < bn) bn = ms; if (mw < bn) bn = mw;
    const int need      = prop.multiProcessorCount * bn;
    const int nblocks   = (N/16) * H * B;
    const float sat     = fminf(100.f, 100.f * nblocks / (float)need);

    auto tf = [&](double ms_val) {
        return flops / (ms_val * 1e-3) / 1e12;
    };

    if (header) {
        printf("=== Benchmark (outer=%d inner=%d) ===\n", outer, inner);
        printf("  %-10s %-10s %-10s %-10s\n", "", "min", "median", "max");
        printf("  %-10s %-10.4f %-10.4f %-10.4f\n",
               "ms", br.min_ms, br.med_ms, br.max_ms);
        printf("  %-10s %-10.4f %-10.4f %-10.4f\n",
               "TFLOPS", tf(br.max_ms), tf(br.med_ms), tf(br.min_ms));
        printf("  Blocos: %d   sat: %.0f%%\n\n", nblocks, sat);
    } else {
        printf("%-5d %-5d %-6d | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-7d %-6.0f%%\n",
               B, H, N_req,
               br.min_ms, br.med_ms, br.max_ms,
               tf(br.max_ms), tf(br.med_ms), tf(br.min_ms),
               nblocks, sat);
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    if (hQ) free(hQ); if (hK) free(hK);
    if (hV) free(hV); if (hO) free(hO);
    if (href) free(href);
}

// ============================================================================
// Sweep com comparação direta contra 4-tiles (flatargs)
// A comparação usa os números do sweep anterior gravados como constantes.
// ============================================================================
static void run_sweep(int actual_regs)
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);

    const int smem  = 4*16*16*(int)sizeof(half);
    const int mr    = prop.regsPerMultiprocessor / (actual_regs * 32);
    const int ms    = (int)(prop.sharedMemPerMultiprocessor / (size_t)smem);
    const int mw    = prop.maxThreadsPerMultiProcessor / 32;
    int bn = mr; if (ms < bn) bn = ms; if (mw < bn) bn = mw;
    const int need  = prop.multiProcessorCount * bn;

    printf("Device: %s  SM%d%d  %d SMs\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("Regs: %d  Blocos/SM: %d  Para saturar: %d\n\n",
           actual_regs, bn, need);

    printf("=== Sweep flash_mha_2tiles vs flash_mha_flatargs (d=64) ===\n");
    printf("Referencia flatargs (mediana) do sweep anterior:\n\n");

    // Resultados medianos do flatargs do sweep anterior
    // Ordem: B H N -> flatargs_med_ms
    struct Ref { int B, H, N; double flatargs_med_ms; };
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
    const int nref = (int)(sizeof(refs)/sizeof(refs[0]));

    printf("%-5s %-5s %-6s | "
           "%-27s | "
           "%-27s | "
           "%-8s %-8s %-8s\n",
           "B","H","N",
           "    2-tiles ms (min/med/max) ",
           "  2-tiles TFLOPS (hi/med/lo) ",
           "flat-med", "speedup", "sat%");
    print_sep();

    for (int i = 0; i < nref; i++) {
        Ref& r = refs[i];
        const size_t bqkv = (size_t)r.B * r.H *
                            (size_t)ceil_div(r.N,16)*16 * 64 * sizeof(half);
        if (bqkv > (size_t)2*1024*1024*1024ULL) {
            printf("%-5d %-5d %-6d | SKIP\n", r.B, r.H, r.N);
            continue;
        }

        const int   N     = ceil_div(r.N, 16) * 16;
        const float scale = 1.f / sqrtf(64.f);
        const double flops = (double)r.B * r.H * 4.0 * N * N * 64;

        half *dQ, *dK, *dV, *dO;
        check_cuda(cudaMalloc(&dQ, bqkv), "Q");
        check_cuda(cudaMalloc(&dK, bqkv), "K");
        check_cuda(cudaMalloc(&dV, bqkv), "V");
        check_cuda(cudaMalloc(&dO, bqkv), "O");
        cudaMemset(dQ,0,bqkv); cudaMemset(dK,0,bqkv); cudaMemset(dV,0,bqkv);

        BenchResult br = bench(dQ, dK, dV, dO, r.B, r.H, N, scale, 7, 100);

        const int nblocks = (N/16) * r.H * r.B;
        const float sat   = fminf(100.f, 100.f * nblocks / (float)need);

        auto tf = [&](double ms_val) {
            return flops / (ms_val * 1e-3) / 1e12;
        };

        double speedup = r.flatargs_med_ms / br.med_ms;

        printf("%-5d %-5d %-6d | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-8.4f %-8.4f %-8.4f | "
               "%-8.4f %-8.3fx %-6.0f%%\n",
               r.B, r.H, r.N,
               br.min_ms, br.med_ms, br.max_ms,
               tf(br.max_ms), tf(br.med_ms), tf(br.min_ms),
               r.flatargs_med_ms, speedup, sat);

        cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    }

    print_sep();
    printf("\nspeedup > 1.0x = 2-tiles e MAIS RAPIDO que flatargs 4-tiles\n");
    printf("speedup < 1.0x = 4-tiles ainda ganha (reuso compensa register pressure)\n");
}

// ============================================================================
// Main
//
//   ./flash_mha_2tiles
//       -> sanity + sweep comparativo
//
//   ./flash_mha_2tiles B H N [outer [inner]]
//       -> benchmark de um caso, com validacao se pequeno
// ============================================================================
int main(int argc, char** argv)
{
    // ATUALIZE apos ver o ptxas log desta compilacao
    const int ACTUAL_REGS = 255;   // placeholder; sera corrigido abaixo

    printf("flash_mha_2tiles — SM75  d=64\n");
    printf("PASS_TILES=%d  (recomputa QK 2x vs 1x no 4-tiles e 4x no original)\n",
           PASS_TILES);
    printf("Regs esperados: %d  (verifique ptxas log)\n\n", ACTUAL_REGS);

    if (argc == 1) {
        // sanity check com validacao
        run_case(1, 2, 64,  true, 5, 100, ACTUAL_REGS, true);
        run_case(1, 4, 128, true, 5, 100, ACTUAL_REGS, true);
        // sweep comparativo
        run_sweep(ACTUAL_REGS);
        return 0;
    }

    const int B     = std::atoi(argv[1]);
    const int H     = std::atoi(argv[2]);
    const int N     = std::atoi(argv[3]);
    const int outer = argc > 4 ? std::atoi(argv[4]) : 7;
    const int inner = argc > 5 ? std::atoi(argv[5]) : 100;
    const bool val  = ((double)B*H*N*N*64) < 3e8;

    run_case(B, H, N, val, outer, inner, ACTUAL_REGS, true);
    return 0;
}


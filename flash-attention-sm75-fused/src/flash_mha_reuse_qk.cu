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
#include "flash_mha_reuse_qk.cuh"

using namespace nvcuda;

static constexpr int TM = 16;
static constexpr int TN = 16;
static constexpr int TK = 16;
static constexpr int D_FIXED = 64;
static constexpr int D_TILES = D_FIXED / TK; // 4

// Atualize se o ptxas reportar outro valor para este target.
// Hoje, no seu build, a versão reuse-QK single-head ficou em 206 regs/thread.
// A multi-head tende a ficar na mesma ordem.
static constexpr int EST_REGS_PER_THREAD = 206;

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

static inline void print_sep(int w = 92) {
    for (int i = 0; i < w; i++) std::putchar('-');
    std::putchar('\n');
}

// ============================================================================
// Layout helper: scatter do accumulator para tile half[16][16] em shared
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
// Kernel MHA + reuse-QK, especializado para d=64
// Grid: (N/16, H, B)
// Block: 32 threads
// ============================================================================
__global__ void __launch_bounds__(32, 2)
flash_mha_reuse_qk_kernel_d64(FlashMHAReuseQKParams p)
{
    const int q_tile  = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch   = blockIdx.z;
    const int lane    = threadIdx.x & 31;

    const int q_base = q_tile * TM;
    if (q_base >= p.N) return;

    const long long stride_b = (long long)p.H * p.N * p.d;
    const long long stride_h = (long long)p.N * p.d;
    const long long off      = (long long)batch * stride_b
                             + (long long)head_id * stride_h;

    const half* Q = p.Q + off;
    const half* K = p.K + off;
    const half* V = p.V + off;
    half*       O = p.O + off;

    const int num_kv_tiles = p.N / TN;

    // smem layout:
    // Q  : 16x16 half = 512 B
    // K  : 16x16 half = 512 B
    // P  : 16x16 half = 512 B
    // V  : 16x16 half = 512 B
    // total = 2048 B
    extern __shared__ half smem[];
    half* smem_Q = smem;
    half* smem_K = smem_Q + TM * TK;
    half* smem_P = smem_K + TN * TK;
    half* smem_V = smem_P + TM * TN;

    // 4 tiles persistentes de saída: cols [0:16], [16:32], [32:48], [48:64]
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O_acc[D_TILES];
    #pragma unroll
    for (int t = 0; t < D_TILES; t++) {
        wmma::fill_fragment(O_acc[t], 0.0f);
    }

    OnlineState state;
    online_state_init(state);

    // ========================================================================
    // Loop nos KV tiles
    // ========================================================================
    for (int kv = 0; kv < num_kv_tiles; kv++) {
        const int kv_base = kv * TN;

        // --------------------------------------------------------------------
        // 1) S = Q @ K^T  (computado UMA vez para este kv tile)
        // --------------------------------------------------------------------
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;
        wmma::fill_fragment(S_acc, 0.0f);

        #pragma unroll
        for (int d_in = 0; d_in < D_TILES; d_in++) {
            const int d_base = d_in * TK;

            #pragma unroll
            for (int i = lane; i < TM * TK; i += 32) {
                const int r = i / TK;
                const int c = i % TK;
                smem_Q[i] = Q[(q_base + r) * D_FIXED + (d_base + c)];
            }

            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                const int r = i / TK;
                const int c = i % TK;
                smem_K[i] = K[(kv_base + r) * D_FIXED + (d_base + c)];
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Q_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> Kt_frag;

            wmma::load_matrix_sync(Q_frag,  smem_Q, TK);
            wmma::load_matrix_sync(Kt_frag, smem_K, TK);

            wmma::mma_sync(S_acc, Q_frag, Kt_frag, S_acc);
            __syncwarp();
        }

        // --------------------------------------------------------------------
        // 2) Softmax unnorm em registradores
        // --------------------------------------------------------------------
        RowMaxSum ms = warp_softmax_unnorm(S_acc, p.scale);

        // --------------------------------------------------------------------
        // 3) Atualiza online state UMA vez
        // --------------------------------------------------------------------
        OnlineStep step = online_step_update(state, ms);

        // --------------------------------------------------------------------
        // 4) Aplica beta ao tile atual de scores
        // --------------------------------------------------------------------
        apply_beta_to_S(S_acc, step);

        // --------------------------------------------------------------------
        // 5) Materializa P UMA vez
        // --------------------------------------------------------------------
        scatter_S_acc_to_half_tile(S_acc, smem_P);
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> P_frag;
        wmma::load_matrix_sync(P_frag, smem_P, 16);

        // --------------------------------------------------------------------
        // 6) Reusa o mesmo P para os 4 tiles de saída
        // --------------------------------------------------------------------
        #pragma unroll
        for (int d_out = 0; d_out < D_TILES; d_out++) {
            apply_alpha_to_O(O_acc[d_out], step);

            const int v_col_base = d_out * TK;

            #pragma unroll
            for (int i = lane; i < TN * TK; i += 32) {
                const int r = i / TK;
                const int c = i % TK;
                smem_V[i] = V[(kv_base + r) * D_FIXED + (v_col_base + c)];
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> V_frag;
            wmma::load_matrix_sync(V_frag, smem_V, TK);

            wmma::mma_sync(O_acc[d_out], P_frag, V_frag, O_acc[d_out]);
            __syncwarp();
        }
    }

    // ========================================================================
    // 7) Finaliza e escreve O
    // ========================================================================
    #pragma unroll
    for (int d_out = 0; d_out < D_TILES; d_out++) {
        finalize_O(O_acc[d_out], state);

        // usa [smem_P | smem_V] = 1024 B como scratch float[256]
        float* smem_scratch = reinterpret_cast<float*>(smem_P);
        wmma::store_matrix_sync(smem_scratch, O_acc[d_out], 16, wmma::mem_row_major);
        __syncwarp();

        const int out_col_base = d_out * TK;
        #pragma unroll
        for (int i = lane; i < TM * TK; i += 32) {
            const int r = i / TK;
            const int c = i % TK;
            O[(q_base + r) * D_FIXED + (out_col_base + c)] =
                __float2half(smem_scratch[i]);
        }
        __syncwarp();
    }
}

// ============================================================================
// Host launcher
// ============================================================================
void flash_mha_reuse_qk_launch(
    const FlashMHAReuseQKParams& p,
    cudaStream_t stream)
{
    if (p.d != 64) {
        fprintf(stderr,
                "[flash_mha_reuse_qk_launch] Esta versao requer d == 64. Recebido d=%d\n",
                p.d);
        return;
    }

    if ((p.N % 16) != 0) {
        fprintf(stderr,
                "[flash_mha_reuse_qk_launch] Esta versao requer N multiplo de 16. Recebido N=%d\n",
                p.N);
        return;
    }

    const int smem_bytes = 4 * 16 * 16 * (int)sizeof(half); // 2048 B

    cudaFuncSetAttribute(
        flash_mha_reuse_qk_kernel_d64,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes);

    dim3 grid(p.N / 16, p.H, p.B);
    dim3 block(32);

    flash_mha_reuse_qk_kernel_d64<<<grid, block, smem_bytes, stream>>>(p);
}

// ============================================================================
// CPU reference para validação pequena
// ============================================================================
static void attention_cpu_ref_mha(
    const half* Q,
    const half* K,
    const half* V,
    float* O_ref,
    int B,
    int H,
    int N,
    int d,
    float scale)
{
    const long long stride_b = (long long)H * N * d;
    const long long stride_h = (long long)N * d;

    float* S = new float[N * N];

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const long long off = (long long)b * stride_b + (long long)h * stride_h;
            const half* q = Q + off;
            const half* k = K + off;
            const half* v = V + off;
            float* o      = O_ref + off;

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float acc = 0.0f;
                    for (int x = 0; x < d; x++) {
                        acc += __half2float(q[i * d + x]) * __half2float(k[j * d + x]);
                    }
                    S[i * N + j] = acc * scale;
                }
            }

            for (int i = 0; i < N; i++) {
                float mx = -FLT_MAX;
                for (int j = 0; j < N; j++) {
                    mx = fmaxf(mx, S[i * N + j]);
                }

                float sm = 0.0f;
                for (int j = 0; j < N; j++) {
                    S[i * N + j] = expf(S[i * N + j] - mx);
                    sm += S[i * N + j];
                }

                for (int j = 0; j < N; j++) {
                    S[i * N + j] /= sm;
                }
            }

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < d; j++) {
                    float acc = 0.0f;
                    for (int x = 0; x < N; x++) {
                        acc += S[i * N + x] * __half2float(v[x * d + j]);
                    }
                    o[i * d + j] = acc;
                }
            }
        }
    }

    delete[] S;
}

// ============================================================================
// Validação
// ============================================================================
static bool validate_mha(
    const half* h_O,
    const float* h_ref,
    int B,
    int H,
    int N,
    int d,
    float tol = 5e-2f,
    bool verbose = true)
{
    const long long stride_b = (long long)H * N * d;
    const long long stride_h = (long long)N * d;

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int wb = 0, wh = 0, wr = 0, wc = 0;

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            const long long off = (long long)b * stride_b + (long long)h * stride_h;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < d; j++) {
                    float gpu = __half2float(h_O[off + i * d + j]);
                    float ref = h_ref[off + i * d + j];
                    float err = fabsf(gpu - ref);
                    float rel = (fabsf(ref) > 1e-6f) ? err / fabsf(ref) : err;

                    if (err > max_abs) {
                        max_abs = err;
                        max_rel = rel;
                        wb = b; wh = h; wr = i; wc = j;
                    }
                }
            }
        }
    }

    printf("  Max abs error: %.6f  at [b=%d h=%d r=%d c=%d]\n",
           max_abs, wb, wh, wr, wc);
    printf("  Max rel error: %.4f%%\n", max_rel * 100.0f);

    if (verbose) {
        const long long off = 0;
        printf("  Primeiras 2 linhas do head 0 batch 0 (gpu | ref):\n");
        for (int i = 0; i < 2 && i < N; i++) {
            printf("    row %d: ", i);
            for (int j = 0; j < 8 && j < d; j++) {
                printf("%.3f|%.3f ",
                       __half2float(h_O[off + i * d + j]),
                       h_ref[off + i * d + j]);
            }
            printf("...\n");
        }
    }

    return max_abs < tol;
}

// ============================================================================
// Benchmark
// ============================================================================
static double benchmark_mha(
    const FlashMHAReuseQKParams& p,
    int repeats)
{
    for (int i = 0; i < 5; i++) {
        flash_mha_reuse_qk_launch(p);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < repeats; i++) {
        flash_mha_reuse_qk_launch(p);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return (double)ms / repeats;
}

// ============================================================================
// Ocupação estimada
// ============================================================================
static int estimate_blocks_per_sm(const cudaDeviceProp& prop)
{
    const int max_by_regs  = prop.regsPerMultiprocessor / (EST_REGS_PER_THREAD * 32);
    const int max_by_smem  = (int)(prop.sharedMemPerMultiprocessor / (4 * 16 * 16 * sizeof(half)));
    const int max_by_warps = prop.maxThreadsPerMultiProcessor / 32;

    int bottleneck = max_by_regs;
    if (max_by_smem  < bottleneck) bottleneck = max_by_smem;
    if (max_by_warps < bottleneck) bottleneck = max_by_warps;
    return bottleneck;
}

// ============================================================================
// Executa um caso
// ============================================================================
static void run_case(
    int B,
    int H,
    int N_user,
    int d,
    int repeats,
    bool do_validate,
    bool pretty_header)
{
    if (d != 64) {
        printf("Este binario suporta apenas d=64. Recebido d=%d\n", d);
        return;
    }

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    const int N = ceil_div(N_user, 16) * 16;
    const float scale = 1.0f / sqrtf((float)d);
    const long long total_elems = (long long)B * H * N * d;
    const size_t bytes_qkv = (size_t)total_elems * sizeof(half);
    const size_t bytes_o   = bytes_qkv;

    if (pretty_header) {
        printf("Device: %s  SM%d%d  %d SMs\n",
               prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        printf("\n=== flash_mha_reuse_qk ===\n");
        printf("B=%d H=%d N_req=%d N_pad=%d d=%d repeats=%d\n",
               B, H, N_user, N, d, repeats);
        printf("scale=1/sqrt(%d)=%.4f\n", d, scale);
        printf("Grid = (%d, %d, %d) = %d blocos\n\n",
               N / 16, H, B, (N / 16) * H * B);
    }

    half *d_Q, *d_K, *d_V, *d_O;
    check_cuda(cudaMalloc(&d_Q, bytes_qkv), "cudaMalloc Q");
    check_cuda(cudaMalloc(&d_K, bytes_qkv), "cudaMalloc K");
    check_cuda(cudaMalloc(&d_V, bytes_qkv), "cudaMalloc V");
    check_cuda(cudaMalloc(&d_O, bytes_o),   "cudaMalloc O");

    half* h_Q = nullptr;
    half* h_K = nullptr;
    half* h_V = nullptr;
    half* h_O = nullptr;
    float* h_ref = nullptr;

    if (do_validate) {
        const double cpu_work = (double)B * H * N * N * d;
        if (cpu_work > 3.0e8) {
            printf("[warn] Validacao CPU pulada: caso muito grande.\n");
            do_validate = false;
        }
    }

    if (do_validate) {
        h_Q   = (half*) malloc(bytes_qkv);
        h_K   = (half*) malloc(bytes_qkv);
        h_V   = (half*) malloc(bytes_qkv);
        h_O   = (half*) malloc(bytes_o);
        h_ref = (float*)malloc((size_t)total_elems * sizeof(float));

        std::srand(42);
        for (long long i = 0; i < total_elems; i++) {
            h_Q[i] = __float2half(rand_val());
            h_K[i] = __float2half(rand_val());
            h_V[i] = __float2half(rand_val());
        }

        printf("[1/3] CPU reference... ");
        fflush(stdout);
        attention_cpu_ref_mha(h_Q, h_K, h_V, h_ref, B, H, N, d, scale);
        printf("OK\n");

        check_cuda(cudaMemcpy(d_Q, h_Q, bytes_qkv, cudaMemcpyHostToDevice), "memcpy Q");
        check_cuda(cudaMemcpy(d_K, h_K, bytes_qkv, cudaMemcpyHostToDevice), "memcpy K");
        check_cuda(cudaMemcpy(d_V, h_V, bytes_qkv, cudaMemcpyHostToDevice), "memcpy V");
    } else {
        check_cuda(cudaMemset(d_Q, 0, bytes_qkv), "memset Q");
        check_cuda(cudaMemset(d_K, 0, bytes_qkv), "memset K");
        check_cuda(cudaMemset(d_V, 0, bytes_qkv), "memset V");
    }

    FlashMHAReuseQKParams p;
    p.Q = d_Q;
    p.K = d_K;
    p.V = d_V;
    p.O = d_O;
    p.B = B;
    p.H = H;
    p.N = N;
    p.d = d;
    p.scale = scale;

    if (do_validate) {
        printf("[2/3] GPU kernel... ");
        fflush(stdout);
        flash_mha_reuse_qk_launch(p);
        check_cuda(cudaDeviceSynchronize(), "kernel sync");
        check_cuda(cudaGetLastError(), "kernel launch");
        printf("OK\n");

        check_cuda(cudaMemcpy(h_O, d_O, bytes_o, cudaMemcpyDeviceToHost), "memcpy O");

        printf("[3/3] Validacao...\n");
        bool ok = validate_mha(h_O, h_ref, B, H, N, d, 5e-2f, true);
        printf("  Resultado: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");
    }

    const double ms = benchmark_mha(p, repeats);
    const double flops = (double)B * H * 4.0 * N * N * d;
    const double tflops = flops / (ms * 1e-3) / 1e12;
    const double bw_alg = (3.0 * bytes_qkv + bytes_o) / (ms * 1e-3) / 1e9;

    const int nblocks = (N / 16) * H * B;
    const int blocks_per_sm_est = estimate_blocks_per_sm(prop);
    const int blocks_to_saturate = prop.multiProcessorCount * blocks_per_sm_est;
    double sat = 100.0 * nblocks / (double)blocks_to_saturate;
    if (sat > 100.0) sat = 100.0;

    if (pretty_header) {
        printf("=== Benchmark (%d iteracoes) ===\n", repeats);
        printf("  Latencia:        %.3f ms\n", ms);
        printf("  TFLOPS:          %.3f\n", tflops);
        printf("  Bandwidth alg.:  %.1f GB/s\n", bw_alg);
        printf("  Blocos grid:     %d\n", nblocks);
        printf("  Sat estimada:    %.0f%%\n", sat);
        printf("  Regs/thread est: %d\n", EST_REGS_PER_THREAD);
        printf("  Smem/bloco:      2 KB\n");
        printf("  Observacao: reuse-QK + grid 3D (B,H,N/16)\n\n");
    } else {
        printf("%-5d %-5d %-6d %-6d | %-8.4f %-10.4f %-10.1f %-8d %-7.0f%%\n",
               B, H, N_user, d,
               ms, tflops, bw_alg, nblocks, sat);
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    if (h_Q)   free(h_Q);
    if (h_K)   free(h_K);
    if (h_V)   free(h_V);
    if (h_O)   free(h_O);
    if (h_ref) free(h_ref);
}

// ============================================================================
// Sweep
// ============================================================================
static void run_sweep()
{
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    const int blocks_per_sm_est = estimate_blocks_per_sm(prop);
    const int blocks_to_saturate = prop.multiProcessorCount * blocks_per_sm_est;

    printf("Device: %s  SM%d%d  %d SMs\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    printf("Estimativa de ocupacao para este kernel:\n");
    printf("  Regs/thread est.:        %d\n", EST_REGS_PER_THREAD);
    printf("  Blocos ativos/SM est.:   %d\n", blocks_per_sm_est);
    printf("  Blocos para saturar GPU: %d\n\n", blocks_to_saturate);

    printf("=== Sweep flash_mha_reuse_qk (d=64 fixo) ===\n\n");
    printf("%-5s %-5s %-6s %-6s | %-8s %-10s %-10s %-8s %-8s\n",
           "B", "H", "N", "d", "ms", "TFLOPS", "GB/s", "blocos", "sat%");
    print_sep();

    struct Cfg { int B, H, N, d; };
    Cfg cfgs[] = {
        {1,  1,  256, 64},
        {1,  1,  512, 64},
        {1,  1, 1024, 64},
        {1,  4,  256, 64},
        {1,  8,  256, 64},
        {1, 12,  256, 64},
        {1, 16,  256, 64},
        {4,  8,  256, 64},
        {8,  8,  256, 64},
        {4, 12,  512, 64},
        {4, 12, 1024, 64},
    };

    const int ncfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
    for (int i = 0; i < ncfg; i++) {
        run_case(cfgs[i].B, cfgs[i].H, cfgs[i].N, cfgs[i].d,
                 300, false, false);
    }

    print_sep();
    printf("sat%% = blocos_no_grid / blocos_para_saturar_device (estimado)\n");
    printf("Kernel especializado: d=64, reuse-QK, multi-head grid 3D.\n");
}

// ============================================================================
// Main
//
// Uso:
//   ./flash_mha_reuse_qk
//       -> roda um sanity check pequeno + sweep
//
//   ./flash_mha_reuse_qk B H N d [repeats] [validate]
//       -> benchmarka um caso especifico
//       -> validate=1 faz CPU ref se o caso for pequeno
// ============================================================================
int main(int argc, char** argv)
{
    if (argc == 1) {
        // sanity check pequeno
        run_case(1, 2, 64, 64, 100, true, true);
        run_sweep();
        return 0;
    }

    int B = (argc > 1) ? std::atoi(argv[1]) : 1;
    int H = (argc > 2) ? std::atoi(argv[2]) : 1;
    int N = (argc > 3) ? std::atoi(argv[3]) : 256;
    int d = (argc > 4) ? std::atoi(argv[4]) : 64;
    int repeats  = (argc > 5) ? std::atoi(argv[5]) : 200;
    int validate = (argc > 6) ? std::atoi(argv[6]) : 0;

    run_case(B, H, N, d, repeats, validate != 0, true);
    return 0;
}


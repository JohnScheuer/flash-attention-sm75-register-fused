// =============================================================================
// main.cu
//
// Harness de validação e benchmark do Flash Attention Register-Fused.
//
// Uso:
//   ./flash_fused [N] [d] [repeats]
//   ./flash_fused 64  64  100
//   ./flash_fused 256 64  50
// =============================================================================

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#include "flash_fused.cuh"

// =============================================================================
// Atenção de referência na CPU (float32, numericamente estável)
// =============================================================================
void attention_cpu_ref(
    const half* Q, const half* K, const half* V,
    float* O,
    int N, int d, float scale)
{
    // S = Q @ K^T  [N x N]
    float* S = new float[N * N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++)
                s += __half2float(Q[i*d+k]) * __half2float(K[j*d+k]);
            S[i*N+j] = s * scale;
        }
    }

    // Softmax por linha
    for (int i = 0; i < N; i++) {
        float mx = -FLT_MAX;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, S[i*N+j]);
        float sm = 0.0f;
        for (int j = 0; j < N; j++) {
            S[i*N+j] = expf(S[i*N+j] - mx);
            sm += S[i*N+j];
        }
        for (int j = 0; j < N; j++) S[i*N+j] /= sm;
    }

    // O = S @ V  [N x d]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++)
                s += S[i*N+k] * __half2float(V[k*d+j]);
            O[i*d+j] = s;
        }
    }

    delete[] S;
}

// =============================================================================
// Utilitários
// =============================================================================
static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        exit(1);
    }
}

static float rand_half_val() {
    return (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
}

// =============================================================================
// Validação
// =============================================================================
bool validate(
    const half* h_O_gpu, const float* h_O_ref,
    int N, int d,
    float tol = 1e-2f,   // FP16 softmax tem erro maior que FP32
    bool verbose = false)
{
    float max_err = 0.0f;
    float max_rel = 0.0f;
    int   worst_r = 0, worst_c = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float gpu = __half2float(h_O_gpu[i*d+j]);
            float ref = h_O_ref[i*d+j];
            float err = fabsf(gpu - ref);
            float rel = (fabsf(ref) > 1e-6f) ? err / fabsf(ref) : err;

            if (err > max_err) {
                max_err = err; max_rel = rel;
                worst_r = i; worst_c = j;
            }
        }
    }

    printf("  Max abs error: %.6f  (at [%d][%d], gpu=%.6f ref=%.6f)\n",
           max_err, worst_r, worst_c,
           __half2float(h_O_gpu[worst_r*d+worst_c]),
           h_O_ref[worst_r*d+worst_c]);
    printf("  Max rel error: %.4f%%\n", max_rel * 100.0f);

    if (verbose) {
        printf("  Primeiras 4 linhas (gpu | ref):\n");
        for (int i = 0; i < 4 && i < N; i++) {
            printf("    row %d: ", i);
            for (int j = 0; j < 8 && j < d; j++) {
                printf("%.3f|%.3f ",
                       __half2float(h_O_gpu[i*d+j]),
                       h_O_ref[i*d+j]);
            }
            printf("...\n");
        }
    }

    return (max_err < tol);
}

// =============================================================================
// Benchmark
// =============================================================================
double benchmark_kernel(
    const FlashFusedParams& p,
    int repeats)
{
    // Warm-up
    for (int i = 0; i < 3; i++)
        flash_fused_launch(p);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < repeats; i++)
        flash_fused_launch(p);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return (double)ms / repeats;  // ms por iteração
}

// =============================================================================
int main(int argc, char** argv)
{
    int N       = (argc > 1) ? atoi(argv[1]) : 64;
    int d       = (argc > 2) ? atoi(argv[2]) : 64;
    int repeats = (argc > 3) ? atoi(argv[3]) : 100;

    // Arredondar para múltiplos de 16
    N = ((N + 15) / 16) * 16;
    d = ((d + 15) / 16) * 16;

    // Info do device
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  SM%d%d  %.1f GB\n\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);

    printf("=== Flash Attention Register-Fused ===\n");
    printf("N=%d  d=%d  scale=1/sqrt(%d)=%.4f\n\n",
           N, d, d, 1.0f/sqrtf((float)d));

    float scale = 1.0f / sqrtf((float)d);
    size_t bytes_qkv = (size_t)N * d * sizeof(half);
    size_t bytes_o   = bytes_qkv;

    // ── Alocar e inicializar host ──────────────────────────────────────────────
    half*  h_Q    = (half* )malloc(bytes_qkv);
    half*  h_K    = (half* )malloc(bytes_qkv);
    half*  h_V    = (half* )malloc(bytes_qkv);
    half*  h_O    = (half* )malloc(bytes_o  );
    float* h_Oref = (float*)malloc((size_t)N * d * sizeof(float));

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = __float2half(rand_half_val());
        h_K[i] = __float2half(rand_half_val());
        h_V[i] = __float2half(rand_half_val());
    }

    // ── Referência CPU ────────────────────────────────────────────────────────
    printf("[1/3] Computando referencia CPU... ");
    fflush(stdout);
    attention_cpu_ref(h_Q, h_K, h_V, h_Oref, N, d, scale);
    printf("OK\n");

    // ── Alocar device ─────────────────────────────────────────────────────────
    half *d_Q, *d_K, *d_V, *d_O;
    check_cuda(cudaMalloc(&d_Q, bytes_qkv), "malloc Q");
    check_cuda(cudaMalloc(&d_K, bytes_qkv), "malloc K");
    check_cuda(cudaMalloc(&d_V, bytes_qkv), "malloc V");
    check_cuda(cudaMalloc(&d_O, bytes_o  ), "malloc O");

    check_cuda(cudaMemcpy(d_Q, h_Q, bytes_qkv, cudaMemcpyHostToDevice), "memcpy Q");
    check_cuda(cudaMemcpy(d_K, h_K, bytes_qkv, cudaMemcpyHostToDevice), "memcpy K");
    check_cuda(cudaMemcpy(d_V, h_V, bytes_qkv, cudaMemcpyHostToDevice), "memcpy V");

    FlashFusedParams params{d_Q, d_K, d_V, d_O, N, d, scale};

    // ── Rodar kernel ──────────────────────────────────────────────────────────
    printf("[2/3] Rodando kernel GPU... ");
    fflush(stdout);
    flash_fused_launch(params);
    check_cuda(cudaDeviceSynchronize(), "sync");
    printf("OK\n");

    check_cuda(cudaMemcpy(h_O, d_O, bytes_o, cudaMemcpyDeviceToHost), "memcpy O");

    // ── Validação ─────────────────────────────────────────────────────────────
    printf("[3/3] Validando... \n");
    bool ok = validate(h_O, h_Oref, N, d, /*tol=*/5e-2f, /*verbose=*/true);
    printf("  Resultado: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

    // ── Benchmark ─────────────────────────────────────────────────────────────
    printf("=== Benchmark (%d iteracoes) ===\n", repeats);
    double ms = benchmark_kernel(params, repeats);

    // FLOPS: 2 * N * N * d (QK^T) + 2 * N * N * d (PV) = 4 * N^2 * d
    double flops  = 4.0 * N * N * d;
    double tflops = (flops / (ms * 1e-3)) / 1e12;
    double bw_gb  = (3.0 * bytes_qkv + bytes_o) / (ms * 1e-3) / 1e9;

    printf("  Latência:    %.3f ms\n",   ms);
    printf("  TFLOPS:      %.3f\n",      tflops);
    printf("  Bandwidth:   %.1f GB/s\n", bw_gb);
    printf("  Smem/bloco:  2 KB\n");
    printf("  Warps/bloco: 1\n\n");

    // ── Teste com múltiplos tamanhos ─────────────────────────────────────────
    if (N == 64 && d == 64) {
        printf("=== Sweep de tamanhos ===\n");
        printf("%-8s %-6s %-12s %-12s %-10s\n",
               "N", "d", "latência(ms)", "TFLOPS", "BW(GB/s)");

        int Ns[] = {64, 128, 256, 512};
        int ds[] = {32, 64, 128};

        for (int ni = 0; ni < 4; ni++) {
            for (int di = 0; di < 3; di++) {
                int Nt = Ns[ni], dt = ds[di];

                size_t bq = (size_t)Nt * dt * sizeof(half);
                half *dQ2, *dK2, *dV2, *dO2;
                if (cudaMalloc(&dQ2, bq) != cudaSuccess) continue;
                cudaMalloc(&dK2, bq);
                cudaMalloc(&dV2, bq);
                cudaMalloc(&dO2, bq);

                // Inicializar com lixo (não precisamos validar aqui)
                cudaMemset(dQ2, 0, bq);
                cudaMemset(dK2, 0, bq);
                cudaMemset(dV2, 0, bq);

                FlashFusedParams p2{dQ2, dK2, dV2, dO2, Nt, dt,
                                    1.0f/sqrtf((float)dt)};
                double ms2 = benchmark_kernel(p2, 50);
                double fl2 = 4.0 * Nt * Nt * dt;
                double tf2 = (fl2 / (ms2 * 1e-3)) / 1e12;
                double bw2 = (4.0 * bq) / (ms2 * 1e-3) / 1e9;

                printf("%-8d %-6d %-12.4f %-12.4f %-10.1f\n",
                       Nt, dt, ms2, tf2, bw2);

                cudaFree(dQ2); cudaFree(dK2);
                cudaFree(dV2); cudaFree(dO2);
            }
        }
    }

    // ── Limpeza ───────────────────────────────────────────────────────────────
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_Oref);

    return ok ? 0 : 1;
}


#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "flash_reuse_qk.cuh"

// ============================================================================
// CPU reference
// ============================================================================
static void attention_cpu_ref(
    const half* Q,
    const half* K,
    const half* V,
    float* O_ref,
    int N,
    int d,
    float scale)
{
    float* S = new float[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++) {
                acc += __half2float(Q[i * d + k]) * __half2float(K[j * d + k]);
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
            for (int k = 0; k < N; k++) {
                acc += S[i * N + k] * __half2float(V[k * d + j]);
            }
            O_ref[i * d + j] = acc;
        }
    }

    delete[] S;
}

// ============================================================================
// Helpers
// ============================================================================
static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error em %s: %s\n", where, cudaGetErrorString(e));
        exit(1);
    }
}

static float rand_val() {
    return (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
}

static bool validate(
    const half* h_O,
    const float* h_ref,
    int N,
    int d,
    float tol = 5e-2f,
    bool verbose = true)
{
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int wr = 0, wc = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            float gpu = __half2float(h_O[i * d + j]);
            float ref = h_ref[i * d + j];
            float err = fabsf(gpu - ref);
            float rel = (fabsf(ref) > 1e-6f) ? err / fabsf(ref) : err;

            if (err > max_abs) {
                max_abs = err;
                max_rel = rel;
                wr = i;
                wc = j;
            }
        }
    }

    printf("  Max abs error: %.6f  (at [%d][%d], gpu=%.6f ref=%.6f)\n",
           max_abs, wr, wc,
           __half2float(h_O[wr * d + wc]), h_ref[wr * d + wc]);
    printf("  Max rel error: %.4f%%\n", max_rel * 100.0f);

    if (verbose) {
        printf("  Primeiras 4 linhas (gpu | ref):\n");
        for (int i = 0; i < 4 && i < N; i++) {
            printf("    row %d: ", i);
            for (int j = 0; j < 8 && j < d; j++) {
                printf("%.3f|%.3f ",
                       __half2float(h_O[i * d + j]),
                       h_ref[i * d + j]);
            }
            printf("...\n");
        }
    }

    return max_abs < tol;
}

static double benchmark_kernel(
    const FlashReuseQKParams& p,
    int repeats)
{
    for (int i = 0; i < 5; i++) {
        flash_reuse_qk_launch(p);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    for (int i = 0; i < repeats; i++) {
        flash_reuse_qk_launch(p);
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
// Main
// ============================================================================
int main(int argc, char** argv)
{
    int N       = (argc > 1) ? atoi(argv[1]) : 256;
    int d       = (argc > 2) ? atoi(argv[2]) : 64;
    int repeats = (argc > 3) ? atoi(argv[3]) : 200;

    if (d != 64) {
        printf("Esta versao specialize é apenas para d=64. Recebido d=%d\n", d);
        return 1;
    }

    // Pad N para múltiplo de 16
    int N_orig = N;
    N = ((N + 15) / 16) * 16;

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("Device: %s  SM%d%d  %d SMs\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("\n=== Flash Attention Reuse-QK (d=64 especializado) ===\n");
    printf("N solicitado=%d  N padded=%d  d=%d  repeats=%d\n",
           N_orig, N, d, repeats);
    printf("scale=1/sqrt(%d)=%.4f\n", d, 1.0f / sqrtf((float)d));
    printf("Objetivo: eliminar recomputacao de QK^T para cada d_out.\n\n");

    const float scale = 1.0f / sqrtf((float)d);

    size_t bytes_qkv = (size_t)N * d * sizeof(half);
    size_t bytes_o   = bytes_qkv;

    half*  h_Q    = (half*) malloc(bytes_qkv);
    half*  h_K    = (half*) malloc(bytes_qkv);
    half*  h_V    = (half*) malloc(bytes_qkv);
    half*  h_O    = (half*) malloc(bytes_o);
    float* h_Oref = (float*)malloc((size_t)N * d * sizeof(float));

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = __float2half(rand_val());
        h_K[i] = __float2half(rand_val());
        h_V[i] = __float2half(rand_val());
    }

    printf("[1/3] CPU reference... ");
    fflush(stdout);
    attention_cpu_ref(h_Q, h_K, h_V, h_Oref, N, d, scale);
    printf("OK\n");

    half *d_Q, *d_K, *d_V, *d_O;
    check_cuda(cudaMalloc(&d_Q, bytes_qkv), "cudaMalloc Q");
    check_cuda(cudaMalloc(&d_K, bytes_qkv), "cudaMalloc K");
    check_cuda(cudaMalloc(&d_V, bytes_qkv), "cudaMalloc V");
    check_cuda(cudaMalloc(&d_O, bytes_o),   "cudaMalloc O");

    check_cuda(cudaMemcpy(d_Q, h_Q, bytes_qkv, cudaMemcpyHostToDevice), "memcpy Q");
    check_cuda(cudaMemcpy(d_K, h_K, bytes_qkv, cudaMemcpyHostToDevice), "memcpy K");
    check_cuda(cudaMemcpy(d_V, h_V, bytes_qkv, cudaMemcpyHostToDevice), "memcpy V");

    FlashReuseQKParams p;
    p.Q = d_Q;
    p.K = d_K;
    p.V = d_V;
    p.O = d_O;
    p.N = N;
    p.d = d;
    p.scale = scale;

    printf("[2/3] GPU kernel... ");
    fflush(stdout);
    flash_reuse_qk_launch(p);
    check_cuda(cudaDeviceSynchronize(), "kernel sync");
    check_cuda(cudaGetLastError(), "kernel launch");
    printf("OK\n");

    check_cuda(cudaMemcpy(h_O, d_O, bytes_o, cudaMemcpyDeviceToHost), "memcpy O");

    printf("[3/3] Validacao...\n");
    bool ok = validate(h_O, h_Oref, N, d, 5e-2f, true);
    printf("  Resultado: %s\n\n", ok ? "PASS ✓" : "FAIL ✗");

    printf("=== Benchmark (%d iteracoes) ===\n", repeats);
    double ms = benchmark_kernel(p, repeats);

    // Nesta versão, não há recomputação de QK por d_out.
    // FLOPs algorítmicos == FLOPs executados principais.
    double flops  = 4.0 * N * N * d;
    double tflops = (flops / (ms * 1e-3)) / 1e12;

    // Métrica simples de BW algorítmica
    double bw_gb  = (3.0 * bytes_qkv + bytes_o) / (ms * 1e-3) / 1e9;

    printf("  Latencia:    %.3f ms\n", ms);
    printf("  TFLOPS:      %.3f\n", tflops);
    printf("  Bandwidth:   %.1f GB/s\n", bw_gb);
    printf("  Smem/bloco:  2 KB\n");
    printf("  Warps/bloco: 1\n");
    printf("  Nota: esta versao remove a taxa 2.5x de recomputacao de QK (para d=64).\n\n");

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_Oref);

    return ok ? 0 : 1;
}


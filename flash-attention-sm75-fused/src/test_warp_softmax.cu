// =============================================================================
// test_warp_softmax.cu — corrigido
// =============================================================================

#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "wmma_layout.cuh"
#include "warp_softmax.cuh"

using namespace nvcuda;

// =============================================================================
// Kernel: carrega matriz conhecida no accumulator via mapeamento direto,
// aplica warp softmax, escreve resultado de volta.
// =============================================================================
__global__ void test_softmax_kernel(
    const float* __restrict__ input,   // [16][16] flat, row-major
    float* __restrict__ output,        // [16][16] flat, resultado
    float* __restrict__ out_max,       // [16]
    float* __restrict__ out_sum,       // [16]
    float scale,
    int   do_normalize)
{
    const int lane = threadIdx.x & 31;

    __shared__ float smem_in[16 * 16];
    __shared__ float smem_out[16 * 16];

    for (int i = lane; i < 256; i += 32)
        smem_in[i] = input[i];
    __syncwarp();

    // ── Carregar smem_in no accumulator usando o mapeamento conhecido ─────────
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_acc;

    {
        const int gid = lane >> 2;   // group_id
        const int tig = lane & 3;    // thread_in_group

        int r_u = gid;
        int r_l = gid + 8;
        int c0  = tig * 2;
        int c1  = tig * 2 + 1;
        int c4  = tig * 2 + 8;
        int c5  = tig * 2 + 9;

        S_acc.x[0] = smem_in[r_u * 16 + c0];
        S_acc.x[1] = smem_in[r_u * 16 + c1];
        S_acc.x[4] = smem_in[r_u * 16 + c4];
        S_acc.x[5] = smem_in[r_u * 16 + c5];

        S_acc.x[2] = smem_in[r_l * 16 + c0];
        S_acc.x[3] = smem_in[r_l * 16 + c1];
        S_acc.x[6] = smem_in[r_l * 16 + c4];
        S_acc.x[7] = smem_in[r_l * 16 + c5];
    }

    // ── Aplicar softmax ───────────────────────────────────────────────────────
    RowMaxSum ms;
    if (do_normalize)
        ms = warp_softmax_normalized(S_acc, scale);
    else
        ms = warp_softmax_unnorm(S_acc, scale);

    // ── Escrever resultado de volta em smem ───────────────────────────────────
    {
        const int gid = lane >> 2;
        const int tig = lane & 3;

        int r_u = gid;
        int r_l = gid + 8;
        int c0  = tig * 2;
        int c1  = tig * 2 + 1;
        int c4  = tig * 2 + 8;
        int c5  = tig * 2 + 9;

        smem_out[r_u * 16 + c0] = S_acc.x[0];
        smem_out[r_u * 16 + c1] = S_acc.x[1];
        smem_out[r_u * 16 + c4] = S_acc.x[4];
        smem_out[r_u * 16 + c5] = S_acc.x[5];

        smem_out[r_l * 16 + c0] = S_acc.x[2];
        smem_out[r_l * 16 + c1] = S_acc.x[3];
        smem_out[r_l * 16 + c4] = S_acc.x[6];
        smem_out[r_l * 16 + c5] = S_acc.x[7];
    }
    __syncwarp();

    // ── Exportar para global ──────────────────────────────────────────────────
    for (int i = lane; i < 256; i += 32)
        output[i] = smem_out[i];

    // Exportar max e sum (apenas thread_in_group == 0 escreve)
    if ((lane & 3) == 0) {
        int gid = lane >> 2;
        out_max[gid]     = ms.row_max[0];
        out_max[gid + 8] = ms.row_max[1];
        out_sum[gid]     = ms.row_sum[0];
        out_sum[gid + 8] = ms.row_sum[1];
    }
}

// =============================================================================
// Referencia CPU
// =============================================================================
void cpu_softmax_ref(
    const float* input,
    float* output,
    float* row_max,
    float* row_sum,
    float scale,
    int do_normalize)
{
    for (int r = 0; r < 16; r++) {
        float mx = -FLT_MAX;
        for (int c = 0; c < 16; c++) {
            float v = input[r * 16 + c] * scale;
            if (v > mx) mx = v;
        }
        row_max[r] = mx;

        float s = 0.0f;
        for (int c = 0; c < 16; c++) {
            float v = expf(input[r * 16 + c] * scale - mx);
            output[r * 16 + c] = v;
            s += v;
        }
        row_sum[r] = s;

        if (do_normalize) {
            for (int c = 0; c < 16; c++)
                output[r * 16 + c] /= s;
        }
    }
}

// =============================================================================
// Teste completo
// =============================================================================
static bool run_test(float scale, int do_normalize, const char* label)
{
    printf("\n--- %s (scale=%.4f, normalized=%d) ---\n",
           label, scale, do_normalize);

    // Gerar input
    float h_input[256];
    srand(1234);
    for (int i = 0; i < 256; i++)
        h_input[i] = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;

    // Referencia CPU
    float h_ref_out[256];
    float h_ref_max[16];
    float h_ref_sum[16];
    cpu_softmax_ref(h_input, h_ref_out, h_ref_max, h_ref_sum, scale, do_normalize);

    // GPU
    float *d_input, *d_output, *d_max, *d_sum;
    cudaMalloc(&d_input,  256 * sizeof(float));
    cudaMalloc(&d_output, 256 * sizeof(float));
    cudaMalloc(&d_max,     16 * sizeof(float));
    cudaMalloc(&d_sum,     16 * sizeof(float));

    cudaMemcpy(d_input, h_input, 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_max,  0, 16 * sizeof(float));
    cudaMemset(d_sum,  0, 16 * sizeof(float));

    test_softmax_kernel<<<1, 32>>>(
        d_input, d_output, d_max, d_sum, scale, do_normalize);
    cudaDeviceSynchronize();

    float h_gpu_out[256];
    float h_gpu_max[16];
    float h_gpu_sum[16];
    cudaMemcpy(h_gpu_out, d_output, 256 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_max, d_max,     16 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpu_sum, d_sum,     16 * sizeof(float), cudaMemcpyDeviceToHost);

    // Comparar
    float max_err_out = 0.0f;
    float max_err_max = 0.0f;
    float max_err_sum = 0.0f;
    int worst_r = 0, worst_c = 0;

    for (int r = 0; r < 16; r++) {
        float em = fabsf(h_gpu_max[r] - h_ref_max[r]);
        float es = fabsf(h_gpu_sum[r] - h_ref_sum[r]);
        if (em > max_err_max) max_err_max = em;
        if (es > max_err_sum) max_err_sum = es;

        for (int c = 0; c < 16; c++) {
            float e = fabsf(h_gpu_out[r * 16 + c] - h_ref_out[r * 16 + c]);
            if (e > max_err_out) {
                max_err_out = e;
                worst_r = r; worst_c = c;
            }
        }
    }

    printf("  max_err (saida):   %.8f  (pior: [%d][%d] gpu=%.6f ref=%.6f)\n",
           max_err_out, worst_r, worst_c,
           h_gpu_out[worst_r * 16 + worst_c],
           h_ref_out[worst_r * 16 + worst_c]);
    printf("  max_err (row_max): %.8f\n", max_err_max);
    printf("  max_err (row_sum): %.8f\n", max_err_sum);

    // Detalhe linha a linha se falhar
    bool pass = (max_err_out < 1e-4f) && (max_err_max < 1e-4f);

    if (!pass) {
        printf("  Detalhe por linha:\n");
        for (int r = 0; r < 16; r++) {
            float em = fabsf(h_gpu_max[r] - h_ref_max[r]);
            float es = fabsf(h_gpu_sum[r] - h_ref_sum[r]);
            if (em > 1e-4f || es > 1e-3f) {
                printf("    row %2d: max %.6f/%.6f  sum %.6f/%.6f\n",
                       r, h_gpu_max[r], h_ref_max[r],
                       h_gpu_sum[r], h_ref_sum[r]);
            }
        }
        printf("  Primeiras 2 linhas (gpu | ref):\n");
        for (int r = 0; r < 2; r++) {
            printf("    row %d:", r);
            for (int c = 0; c < 16; c++)
                printf(" %.3f|%.3f", h_gpu_out[r*16+c], h_ref_out[r*16+c]);
            printf("\n");
        }
    }

    printf("  Resultado: %s\n", pass ? "PASS ✓" : "FAIL ✗");

    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_max);   cudaFree(d_sum);

    return pass;
}

// =============================================================================
int main()
{
    printf("=== Teste Warp Register Softmax ===\n");

    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s  SM%d%d\n", prop.name, prop.major, prop.minor);

    bool all_pass = true;

    // Teste 1: normalizado, scale padrao
    all_pass &= run_test(1.0f / sqrtf(16.0f), 1, "Softmax Normalizado (scale=1/sqrt(16))");

    // Teste 2: normalizado, scale=1
    all_pass &= run_test(1.0f, 1, "Softmax Normalizado (scale=1.0)");

    // Teste 3: unnorm
    all_pass &= run_test(1.0f / sqrtf(64.0f), 0, "Softmax Unnorm (scale=1/sqrt(64))");

    printf("\n=== Resultado Geral: %s ===\n", all_pass ? "PASS ✓" : "FAIL ✗");

    return all_pass ? 0 : 1;
}


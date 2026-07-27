#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// ============================================================
// Naive attention: O = softmax(Q @ K^T / sqrt(d)) @ V
// Materializes full N×N attention matrix (O(N²) memory)
// FP16 input, FP32 accumulate
// ============================================================

// Kernel 1: S = Q @ K^T * scale
// S[b][h][i][j] = sum_k Q[b][h][i][k] * K[b][h][j][k] * scale
// Grid: (seq_len, seq_len, batch * heads)
// Each thread computes one element of S
__global__ void naive_qk_kernel(
    const half* __restrict__ Q,    // [batch, heads, seq_len, head_dim]
    const half* __restrict__ K,    // [batch, heads, seq_len, head_dim]
    float* __restrict__ S,         // [batch, heads, seq_len, seq_len]
    int batch, int heads, int seq_len, int head_dim,
    float scale, bool causal
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // query position
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // key position
    int bh = blockIdx.z;                              // batch * head index

    if (i >= seq_len || j >= seq_len) return;

    // Causal mask: positions where j > i are masked
    if (causal && j > i) {
        S[bh * seq_len * seq_len + i * seq_len + j] = -1e9f;
        return;
    }

    int qkv_offset = bh * seq_len * head_dim;

    float acc = 0.0f;
    for (int k = 0; k < head_dim; k++) {
        float q_val = __half2float(Q[qkv_offset + i * head_dim + k]);
        float k_val = __half2float(K[qkv_offset + j * head_dim + k]);
        acc += q_val * k_val;
    }

    S[bh * seq_len * seq_len + i * seq_len + j] = acc * scale;
}

// Kernel 2: P = softmax(S) row-wise
// For each row i: P[i][j] = exp(S[i][j] - max_j S[i][j]) / sum_j exp(S[i][j] - max_j)
// Grid: (seq_len, 1, batch * heads)
// Each thread handles one row
__global__ void naive_softmax_kernel(
    float* __restrict__ S,    // [batch*heads, seq_len, seq_len] in-place -> P
    int seq_len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // row index
    int bh = blockIdx.z;

    if (i >= seq_len) return;

    float* row = S + bh * seq_len * seq_len + i * seq_len;

    // Find max
    float row_max = -FLT_MAX;
    for (int j = 0; j < seq_len; j++) {
        row_max = fmaxf(row_max, row[j]);
    }

    // Compute exp and sum
    float row_sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        row[j] = expf(row[j] - row_max);
        row_sum += row[j];
    }

    // Normalize
    float inv_sum = 1.0f / row_sum;
    for (int j = 0; j < seq_len; j++) {
        row[j] *= inv_sum;
    }
}

// Kernel 3: O = P @ V
// O[b][h][i][k] = sum_j P[b][h][i][j] * V[b][h][j][k]
// Grid: (seq_len, head_dim, batch * heads)
// Each thread computes one element of O
__global__ void naive_pv_kernel(
    const float* __restrict__ P,   // [batch*heads, seq_len, seq_len]
    const half* __restrict__ V,    // [batch, heads, seq_len, head_dim]
    half* __restrict__ O,          // [batch, heads, seq_len, head_dim]
    int seq_len, int head_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // output row
    int k = blockIdx.y * blockDim.y + threadIdx.y;  // output col (head_dim)
    int bh = blockIdx.z;

    if (i >= seq_len || k >= head_dim) return;

    int v_offset = bh * seq_len * head_dim;
    const float* p_row = P + bh * seq_len * seq_len + i * seq_len;

    float acc = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        float v_val = __half2float(V[v_offset + j * head_dim + k]);
        acc += p_row[j] * v_val;
    }

    O[v_offset + i * head_dim + k] = __float2half(acc);
}

// ============================================================
// Host launcher
// ============================================================

extern "C" void launch_naive_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch, int heads, int seq_len, int head_dim,
    float scale, bool causal,
    float* workspace  // pre-allocated: batch * heads * seq_len * seq_len floats
) {
    int total_bh = batch * heads;

    // Kernel 1: Q @ K^T
    {
        dim3 block(16, 16);
        dim3 grid(
            (seq_len + block.x - 1) / block.x,
            (seq_len + block.y - 1) / block.y,
            total_bh
        );
        naive_qk_kernel<<<grid, block>>>(
            Q, K, workspace,
            batch, heads, seq_len, head_dim,
            scale, causal
        );
    }

    // Kernel 2: softmax
    {
        dim3 block(256);
        dim3 grid(
            (seq_len + block.x - 1) / block.x,
            1,
            total_bh
        );
        naive_softmax_kernel<<<grid, block>>>(workspace, seq_len);
    }

    // Kernel 3: P @ V
    {
        dim3 block(16, 16);
        dim3 grid(
            (seq_len + block.x - 1) / block.x,
            (head_dim + block.y - 1) / block.y,
            total_bh
        );
        naive_pv_kernel<<<grid, block>>>(
            workspace, V, O,
            seq_len, head_dim
        );
    }

    cudaDeviceSynchronize();
}

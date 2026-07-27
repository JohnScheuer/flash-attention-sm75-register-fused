#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <float.h>
#include <stdio.h>

using namespace nvcuda;

namespace {

// ============================================================
// Common constants
// ============================================================

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kThreads = 128;  // 4 warps

// ============================================================
// d=64 variants
// ============================================================

constexpr int kD64         = 64;
constexpr int kBrScalar64  = 32;
constexpr int kBcScalar64  = 32;
constexpr int kBrQK64      = 32;
constexpr int kBcQK64      = 32;

// NEW final d=64 kernel
constexpr int kBrFinal64   = 64;
constexpr int kBcFinal64   = 64;
constexpr int kSmemBytesD64Final = 40 * 1024;

// ============================================================
// d=128 variant
// ============================================================

constexpr int kD128        = 128;
constexpr int kBrFinal128  = 32;
constexpr int kBcFinal128  = 32;
constexpr int kSmemBytesD128Final = 40 * 1024;

// ============================================================
// v1 — Scalar d=64
// ============================================================

__global__ void flash_attention_forward_kernel_d64_scalar(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len,
    float scale,
    bool causal
) {
    const int tid    = threadIdx.x;
    const int q_tile = blockIdx.x;
    const int bh     = blockIdx.y;

    if (tid >= kBrScalar64) return;

    const int q_idx  = q_tile * kBrScalar64 + tid;
    const bool valid_q = (q_idx < seq_len);
    const int bh_off = bh * seq_len * kD64;
    const int nkv    = (seq_len + kBcScalar64 - 1) / kBcScalar64;

    __shared__ half smem_k[kBcScalar64 * kD64];
    __shared__ half smem_v[kBcScalar64 * kD64];

    float q[kD64];
    float o[kD64];
    float scores[kBcScalar64];

    float mi = -FLT_MAX;
    float li = 0.f;

    #pragma unroll
    for (int d = 0; d < kD64; d++) {
        q[d] = valid_q ? __half2float(Q[bh_off + q_idx * kD64 + d]) : 0.f;
        o[d] = 0.f;
    }

    const int q_tile_start = q_tile * kBrScalar64;
    const int q_tile_end   = min(q_tile_start + kBrScalar64 - 1, seq_len - 1);

    for (int kv = 0; kv < nkv; kv++) {
        const int k_start = kv * kBcScalar64;
        const int k_end   = min(k_start + kBcScalar64 - 1, seq_len - 1);

        if (causal && k_start > q_tile_end) break;

        const bool tile_fully_valid = (!causal) || (k_end <= q_tile_start);
        const bool tile_partial     = causal && !tile_fully_valid;

        const int kr = kv * kBcScalar64 + tid;
        #pragma unroll
        for (int d = 0; d < kD64; d++) {
            smem_k[tid * kD64 + d] = (kr < seq_len)
                ? K[bh_off + kr * kD64 + d] : __float2half(0.f);
            smem_v[tid * kD64 + d] = (kr < seq_len)
                ? V[bh_off + kr * kD64 + d] : __float2half(0.f);
        }
        __syncthreads();

        float mij = -FLT_MAX;

        if (valid_q) {
            #pragma unroll
            for (int c = 0; c < kBcScalar64; c++) {
                const int gc = k_start + c;
                if (gc >= seq_len) {
                    scores[c] = -FLT_MAX;
                    continue;
                }
                if (tile_partial && gc > q_idx) {
                    scores[c] = -FLT_MAX;
                    continue;
                }

                float acc = 0.f;
                #pragma unroll
                for (int d = 0; d < kD64; d++) {
                    acc += q[d] * __half2float(smem_k[c * kD64 + d]);
                }
                scores[c] = acc * scale;
                mij = fmaxf(mij, scores[c]);
            }
        }

        const float mn  = valid_q ? fmaxf(mi, mij) : mi;
        const float alp = (mi == -FLT_MAX) ? 0.f : __expf(mi - mn);

        if (valid_q) {
            #pragma unroll
            for (int d = 0; d < kD64; d++) o[d] *= alp;

            float lij = 0.f;
            #pragma unroll
            for (int c = 0; c < kBcScalar64; c++) {
                if (scores[c] == -FLT_MAX) continue;
                const float p = __expf(scores[c] - mn);
                lij += p;
                #pragma unroll
                for (int d = 0; d < kD64; d++) {
                    o[d] += p * __half2float(smem_v[c * kD64 + d]);
                }
            }
            li = li * alp + lij;
        }

        mi = mn;
        __syncthreads();
    }

    if (valid_q) {
        const float inv = (li > 0.f) ? (1.f / li) : 0.f;
        #pragma unroll
        for (int d = 0; d < kD64; d++) {
            O[bh_off + q_idx * kD64 + d] = __float2half(o[d] * inv);
        }
    }
}

// ============================================================
// v2 — WMMA QKᵀ only, d=64, Br=32, Bc=32
// ============================================================

__global__ void flash_attention_forward_kernel_d64_wmma_qk(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len,
    float scale,
    bool causal
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int q_tile  = blockIdx.x;
    const int bh      = blockIdx.y;

    const int bh_off = bh * seq_len * kD64;
    const int nkv    = (seq_len + kBcQK64 - 1) / kBcQK64;

    const int row_quad   = warp_id / 2;
    const int col_quad   = warp_id % 2;
    const int q_smem_row = row_quad * kWmmaN;
    const int k_smem_row = col_quad * kWmmaN;

    const bool is_sf  = (warp_id == 0 || warp_id == 2);
    const int my_qrow = q_tile * kBrQK64 + q_smem_row + lane_id;

    __shared__ half  smem_Q[kBrQK64 * kD64];
    __shared__ half  smem_K[kBcQK64 * kD64];
    __shared__ half  smem_V[kBcQK64 * kD64];
    __shared__ float smem_S[kBrQK64 * kBcQK64];

    {
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = 0; i < (kBrQK64 * kD64) / kThreads; i++) {
            const int idx = tid + i * kThreads;
            const int gr  = q_tile * kBrQK64 + idx / kD64;
            smem_Q[idx] = (gr < seq_len)
                ? Q[bh_off + gr * kD64 + (idx % kD64)]
                : __float2half(0.f);
        }
    }
    __syncthreads();

    float mi = -FLT_MAX;
    float li = 0.f;
    float o_acc[kD64];
    #pragma unroll
    for (int d = 0; d < kD64; d++) o_acc[d] = 0.f;

    const int q_tile_start = q_tile * kBrQK64;
    const int q_tile_end   = min(q_tile_start + kBrQK64 - 1, seq_len - 1);

    for (int kv = 0; kv < nkv; kv++) {
        const int kb      = kv * kBcQK64;
        const int k_start = kb;
        const int k_end   = min(kb + kBcQK64 - 1, seq_len - 1);

        if (causal && k_start > q_tile_end) break;

        const bool tile_fully_valid = (!causal) || (k_end <= q_tile_start);
        const bool tile_partial     = causal && !tile_fully_valid;

        {
            const int tid = threadIdx.x;
            #pragma unroll
            for (int i = 0; i < (kBcQK64 * kD64) / kThreads; i++) {
                const int idx = tid + i * kThreads;
                const int gr  = kb + idx / kD64;
                smem_K[idx] = (gr < seq_len)
                    ? K[bh_off + gr * kD64 + (idx % kD64)]
                    : __float2half(0.f);
                smem_V[idx] = (gr < seq_len)
                    ? V[bh_off + gr * kD64 + (idx % kD64)]
                    : __float2half(0.f);
            }
        }
        __syncthreads();

        {
            wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
            wmma::fill_fragment(acc, 0.f);

            #pragma unroll
            for (int ks = 0; ks < kD64 / kWmmaK; ks++) {
                wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> qa;
                wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> kb_f;

                wmma::load_matrix_sync(qa,  smem_Q + q_smem_row * kD64 + ks * kWmmaK, kD64);
                wmma::load_matrix_sync(kb_f, smem_K + k_smem_row * kD64 + ks * kWmmaK, kD64);
                wmma::mma_sync(acc, qa, kb_f, acc);
            }

            wmma::store_matrix_sync(
                smem_S + q_smem_row * kBcQK64 + k_smem_row,
                acc, kBcQK64, wmma::mem_row_major
            );
        }
        __syncthreads();

        if (is_sf && lane_id < kWmmaN && my_qrow < seq_len) {
            const int sr = q_smem_row + lane_id;
            float mij = -FLT_MAX;
            float row_sc[kBcQK64];

            #pragma unroll
            for (int c = 0; c < kBcQK64; c++) {
                const int gc = kb + c;
                if (gc >= seq_len) {
                    row_sc[c] = -FLT_MAX;
                    continue;
                }
                if (tile_partial && gc > my_qrow) {
                    row_sc[c] = -FLT_MAX;
                    continue;
                }
                row_sc[c] = smem_S[sr * kBcQK64 + c] * scale;
                mij = fmaxf(mij, row_sc[c]);
            }

            const float mn  = fmaxf(mi, mij);
            const float alp = (mi == -FLT_MAX) ? 0.f : __expf(mi - mn);

            #pragma unroll
            for (int d = 0; d < kD64; d++) o_acc[d] *= alp;

            float lij = 0.f;
            #pragma unroll
            for (int c = 0; c < kBcQK64; c++) {
                if (row_sc[c] == -FLT_MAX) continue;
                const float p = __expf(row_sc[c] - mn);
                lij += p;
                #pragma unroll
                for (int d = 0; d < kD64; d++) {
                    o_acc[d] += p * __half2float(smem_V[c * kD64 + d]);
                }
            }

            li = li * alp + lij;
            mi = mn;
        }
        __syncthreads();
    }

    if (is_sf && lane_id < kWmmaN && my_qrow < seq_len) {
        const float inv = (li > 0.f) ? (1.f / li) : 0.f;
        #pragma unroll
        for (int d = 0; d < kD64; d++) {
            O[bh_off + my_qrow * kD64 + d] = __float2half(o_acc[d] * inv);
        }
    }
}

// ============================================================
// vFinal d=64 — WMMA QKᵀ + PV, Br=64, Bc=64
//
// This is the new best-effort d=64 kernel.
//
// Shared memory layout (40 KB):
//   0      -  8192 : Q [64x64] half   = 8 KB   (persistent)
//   8192   - 16384 : K [64x64] half   = 8 KB   -> alias P [64x64] half
//   16384  - 24576 : V [64x64] half   = 8 KB
//   24576  - 40960 : S [64x64] float  = 16 KB  -> alias T [64x64] float
//
// Warps:
//   warp 0 owns rows  0..15
//   warp 1 owns rows 16..31
//   warp 2 owns rows 32..47
//   warp 3 owns rows 48..63
//
// All 4 warps are active in:
//   - QKᵀ
//   - softmax
//   - PV
//   - accumulation
// ============================================================

__global__ void flash_attention_forward_kernel_d64_wmma_qk_pv_br64_bc64(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len,
    float scale,
    bool causal
) {
    const int warp_id     = threadIdx.x / 32;
    const int lane_id     = threadIdx.x % 32;
    const int q_tile      = blockIdx.x;
    const int bh          = blockIdx.y;
    const int bh_off      = bh * seq_len * kD64;
    const int nkv         = (seq_len + kBcFinal64 - 1) / kBcFinal64;
    const int my_row_base = warp_id * kWmmaN;
    const int my_qrow     = q_tile * kBrFinal64 + my_row_base + lane_id;

    extern __shared__ unsigned char smem_raw[];
    half*  smem_Q = reinterpret_cast<half*> (smem_raw + 0);
    half*  smem_K = reinterpret_cast<half*> (smem_raw + 8192);
    half*  smem_V = reinterpret_cast<half*> (smem_raw + 16384);
    float* smem_S = reinterpret_cast<float*>(smem_raw + 24576);
    half*  smem_P = reinterpret_cast<half*> (smem_raw + 8192);   // alias K
    float* smem_T = reinterpret_cast<float*>(smem_raw + 24576);  // alias S

    // Load Q [64x64] once
    {
        const int tid = threadIdx.x;
        const int elems = kBrFinal64 * kD64;  // 4096
        #pragma unroll
        for (int i = 0; i < elems / kThreads; i++) {
            const int idx = tid + i * kThreads;
            const int gr  = q_tile * kBrFinal64 + idx / kD64;
            smem_Q[idx] = (gr < seq_len)
                ? Q[bh_off + gr * kD64 + (idx % kD64)]
                : __float2half(0.f);
        }
    }
    __syncthreads();

    float mi = -FLT_MAX;
    float li = 0.f;
    float o_acc[kD64];
    #pragma unroll
    for (int d = 0; d < kD64; d++) o_acc[d] = 0.f;

    const int q_tile_start = q_tile * kBrFinal64;
    const int q_tile_end   = min(q_tile_start + kBrFinal64 - 1, seq_len - 1);

    for (int kv = 0; kv < nkv; kv++) {
        const int kb      = kv * kBcFinal64;
        const int k_start = kb;
        const int k_end   = min(kb + kBcFinal64 - 1, seq_len - 1);

        if (causal && k_start > q_tile_end) break;

        const bool tile_fully_valid = (!causal) || (k_end <= q_tile_start);
        const bool tile_partial     = causal && !tile_fully_valid;

        // Load K,V [64x64]
        {
            const int tid = threadIdx.x;
            const int elems = kBcFinal64 * kD64;  // 4096
            #pragma unroll
            for (int i = 0; i < elems / kThreads; i++) {
                const int idx = tid + i * kThreads;
                const int gr  = kb + idx / kD64;
                smem_K[idx] = (gr < seq_len)
                    ? K[bh_off + gr * kD64 + (idx % kD64)]
                    : __float2half(0.f);
                smem_V[idx] = (gr < seq_len)
                    ? V[bh_off + gr * kD64 + (idx % kD64)]
                    : __float2half(0.f);
            }
        }
        __syncthreads();

        // QKᵀ: S[64x64]
        // Each warp computes one 16-row block across 4 col blocks
        {
            #pragma unroll
            for (int cb = 0; cb < kBcFinal64; cb += kWmmaN) {
                wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
                wmma::fill_fragment(acc, 0.f);

                #pragma unroll
                for (int ks = 0; ks < kD64 / kWmmaK; ks++) {
                    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> qa;
                    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> kb_f;

                    wmma::load_matrix_sync(
                        qa,
                        smem_Q + my_row_base * kD64 + ks * kWmmaK,
                        kD64
                    );
                    wmma::load_matrix_sync(
                        kb_f,
                        smem_K + cb * kD64 + ks * kWmmaK,
                        kD64
                    );
                    wmma::mma_sync(acc, qa, kb_f, acc);
                }

                wmma::store_matrix_sync(
                    smem_S + my_row_base * kBcFinal64 + cb,
                    acc,
                    kBcFinal64,
                    wmma::mem_row_major
                );
            }
        }
        __syncthreads();

        // Softmax over 64 cols
        float my_mn  = mi;
        float my_alp = 0.f;

        if (lane_id < kWmmaN && my_qrow < seq_len) {
            const int sr = my_row_base + lane_id;

            float mij = -FLT_MAX;
            if (tile_fully_valid) {
                #pragma unroll
                for (int c = 0; c < kBcFinal64; c++) {
                    if (kb + c >= seq_len) continue;
                    mij = fmaxf(mij, smem_S[sr * kBcFinal64 + c] * scale);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < kBcFinal64; c++) {
                    const int gc = kb + c;
                    if (gc >= seq_len || gc > my_qrow) continue;
                    mij = fmaxf(mij, smem_S[sr * kBcFinal64 + c] * scale);
                }
            }

            my_mn  = fmaxf(mi, mij);
            my_alp = (mi == -FLT_MAX) ? 0.f : __expf(mi - my_mn);

            float lij = 0.f;
            if (tile_fully_valid) {
                #pragma unroll
                for (int c = 0; c < kBcFinal64; c++) {
                    const int gc = kb + c;
                    float p = 0.f;
                    if (gc < seq_len) {
                        p = __expf(smem_S[sr * kBcFinal64 + c] * scale - my_mn);
                        lij += p;
                    }
                    smem_P[sr * kBcFinal64 + c] = __float2half(p);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < kBcFinal64; c++) {
                    const int gc = kb + c;
                    float p = 0.f;
                    if (gc < seq_len && gc <= my_qrow) {
                        p = __expf(smem_S[sr * kBcFinal64 + c] * scale - my_mn);
                        lij += p;
                    }
                    smem_P[sr * kBcFinal64 + c] = __float2half(p);
                }
            }

            li = li * my_alp + lij;
            mi = my_mn;
        }
        __syncthreads();

        // PV: T[64x64]
        // Each warp computes one 16-row block across 4 output col blocks
        {
            #pragma unroll
            for (int t_col = 0; t_col < kD64; t_col += kWmmaN) {
                wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> t_acc;
                wmma::fill_fragment(t_acc, 0.f);

                #pragma unroll
                for (int ks = 0; ks < kBcFinal64 / kWmmaK; ks++) {
                    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> p_frag;
                    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> v_frag;

                    wmma::load_matrix_sync(
                        p_frag,
                        smem_P + my_row_base * kBcFinal64 + ks * kWmmaK,
                        kBcFinal64
                    );
                    wmma::load_matrix_sync(
                        v_frag,
                        smem_V + ks * kWmmaK * kD64 + t_col,
                        kD64
                    );
                    wmma::mma_sync(t_acc, p_frag, v_frag, t_acc);
                }

                wmma::store_matrix_sync(
                    smem_T + my_row_base * kD64 + t_col,
                    t_acc,
                    kD64,
                    wmma::mem_row_major
                );
            }
        }
        __syncthreads();

        if (lane_id < kWmmaN && my_qrow < seq_len) {
            const int tr = my_row_base + lane_id;
            #pragma unroll
            for (int d = 0; d < kD64; d++) {
                o_acc[d] = o_acc[d] * my_alp + smem_T[tr * kD64 + d];
            }
        }
        __syncthreads();
    }

    if (lane_id < kWmmaN && my_qrow < seq_len) {
        const float inv = (li > 0.f) ? (1.f / li) : 0.f;
        #pragma unroll
        for (int d = 0; d < kD64; d++) {
            O[bh_off + my_qrow * kD64 + d] = __float2half(o_acc[d] * inv);
        }
    }
}

// ============================================================
// vFinal d=128 — WMMA QKᵀ + PV, Br=32, Bc=32, causal-pruned
// ============================================================

__global__ void flash_attention_forward_kernel_d128_wmma_qk_pv_br32(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len,
    float scale,
    bool causal
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int q_tile  = blockIdx.x;
    const int bh      = blockIdx.y;
    const int bh_off  = bh * seq_len * kD128;
    const int nkv     = (seq_len + kBcFinal128 - 1) / kBcFinal128;

    const int row_quad   = warp_id / 2;
    const int col_quad   = warp_id % 2;
    const int q_smem_row = row_quad * kWmmaN;
    const int k_smem_row = col_quad * kWmmaN;
    const int pv_p_row   = row_quad * kWmmaN;

    const bool is_sf  = (warp_id == 0 || warp_id == 2);
    const int my_qrow = q_tile * kBrFinal128 + q_smem_row + lane_id;

    extern __shared__ unsigned char smem_raw[];
    half*  smem_Q = reinterpret_cast<half*> (smem_raw + 0);
    half*  smem_K = reinterpret_cast<half*> (smem_raw + 8192);
    half*  smem_V = reinterpret_cast<half*> (smem_raw + 16384);
    float* smem_S = reinterpret_cast<float*>(smem_raw + 24576);
    half*  smem_P = reinterpret_cast<half*> (smem_raw + 8192);
    float* smem_T = reinterpret_cast<float*>(smem_raw + 24576);

    {
        const int tid = threadIdx.x;
        const int elems = kBrFinal128 * kD128; // 4096
        #pragma unroll
        for (int i = 0; i < elems / kThreads; i++) {
            const int idx = tid + i * kThreads;
            const int gr  = q_tile * kBrFinal128 + idx / kD128;
            smem_Q[idx] = (gr < seq_len)
                ? Q[bh_off + gr * kD128 + (idx % kD128)] : __float2half(0.f);
        }
    }
    __syncthreads();

    float mi = -FLT_MAX;
    float li = 0.f;
    float o_acc[kD128];
    #pragma unroll
    for (int d = 0; d < kD128; d++) o_acc[d] = 0.f;

    const int q_tile_start = q_tile * kBrFinal128;
    const int q_tile_end   = min(q_tile_start + kBrFinal128 - 1, seq_len - 1);

    for (int kv = 0; kv < nkv; kv++) {
        const int kb      = kv * kBcFinal128;
        const int k_start = kb;
        const int k_end   = min(kb + kBcFinal128 - 1, seq_len - 1);

        if (causal && k_start > q_tile_end) break;

        const bool tile_fully_valid = (!causal) || (k_end <= q_tile_start);
        const bool tile_partial     = causal && !tile_fully_valid;

        {
            const int tid = threadIdx.x;
            const int elems = kBcFinal128 * kD128; // 4096
            #pragma unroll
            for (int i = 0; i < elems / kThreads; i++) {
                const int idx = tid + i * kThreads;
                const int gr  = kb + idx / kD128;
                smem_K[idx] = (gr < seq_len)
                    ? K[bh_off + gr * kD128 + (idx % kD128)] : __float2half(0.f);
                smem_V[idx] = (gr < seq_len)
                    ? V[bh_off + gr * kD128 + (idx % kD128)] : __float2half(0.f);
            }
        }
        __syncthreads();

        // QKᵀ
        {
            wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc;
            wmma::fill_fragment(acc, 0.f);

            #pragma unroll
            for (int ks = 0; ks < kD128 / kWmmaK; ks++) {
                wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> qa;
                wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> kb_f;

                wmma::load_matrix_sync(
                    qa,
                    smem_Q + q_smem_row * kD128 + ks * kWmmaK,
                    kD128
                );
                wmma::load_matrix_sync(
                    kb_f,
                    smem_K + k_smem_row * kD128 + ks * kWmmaK,
                    kD128
                );
                wmma::mma_sync(acc, qa, kb_f, acc);
            }

            wmma::store_matrix_sync(
                smem_S + q_smem_row * kBcFinal128 + k_smem_row,
                acc, kBcFinal128, wmma::mem_row_major
            );
        }
        __syncthreads();

        float my_mn = mi;
        float my_alp = 0.f;

        if (is_sf && lane_id < kWmmaN && my_qrow < seq_len) {
            const int sr = q_smem_row + lane_id;

            float mij = -FLT_MAX;
            if (tile_fully_valid) {
                #pragma unroll
                for (int c = 0; c < kBcFinal128; c++) {
                    if (kb + c >= seq_len) continue;
                    mij = fmaxf(mij, smem_S[sr * kBcFinal128 + c] * scale);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < kBcFinal128; c++) {
                    const int gc = kb + c;
                    if (gc >= seq_len || gc > my_qrow) continue;
                    mij = fmaxf(mij, smem_S[sr * kBcFinal128 + c] * scale);
                }
            }

            my_mn  = fmaxf(mi, mij);
            my_alp = (mi == -FLT_MAX) ? 0.f : __expf(mi - my_mn);

            float lij = 0.f;
            if (tile_fully_valid) {
                #pragma unroll
                for (int c = 0; c < kBcFinal128; c++) {
                    const int gc = kb + c;
                    float p = 0.f;
                    if (gc < seq_len) {
                        p = __expf(smem_S[sr * kBcFinal128 + c] * scale - my_mn);
                        lij += p;
                    }
                    smem_P[sr * kBcFinal128 + c] = __float2half(p);
                }
            } else {
                #pragma unroll
                for (int c = 0; c < kBcFinal128; c++) {
                    const int gc = kb + c;
                    float p = 0.f;
                    if (gc < seq_len && gc <= my_qrow) {
                        p = __expf(smem_S[sr * kBcFinal128 + c] * scale - my_mn);
                        lij += p;
                    }
                    smem_P[sr * kBcFinal128 + c] = __float2half(p);
                }
            }

            li = li * my_alp + lij;
            mi = my_mn;
        }
        __syncthreads();

        // PV
        {
            #pragma unroll
            for (int t_col = 0; t_col < kD128; t_col += kWmmaN) {
                wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> t_acc;
                wmma::fill_fragment(t_acc, 0.f);

                #pragma unroll
                for (int ks = 0; ks < kBcFinal128 / kWmmaK; ks++) {
                    wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> p_frag;
                    wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> v_frag;

                    wmma::load_matrix_sync(
                        p_frag,
                        smem_P + pv_p_row * kBcFinal128 + ks * kWmmaK,
                        kBcFinal128
                    );
                    wmma::load_matrix_sync(
                        v_frag,
                        smem_V + ks * kWmmaK * kD128 + t_col,
                        kD128
                    );
                    wmma::mma_sync(t_acc, p_frag, v_frag, t_acc);
                }

                wmma::store_matrix_sync(
                    smem_T + pv_p_row * kD128 + t_col,
                    t_acc,
                    kD128,
                    wmma::mem_row_major
                );
            }
        }
        __syncthreads();

        if (is_sf && lane_id < kWmmaN && my_qrow < seq_len) {
            const int tr = q_smem_row + lane_id;
            #pragma unroll
            for (int d = 0; d < kD128; d++) {
                o_acc[d] = o_acc[d] * my_alp + smem_T[tr * kD128 + d];
            }
        }
        __syncthreads();
    }

    if (is_sf && lane_id < kWmmaN && my_qrow < seq_len) {
        const float inv = (li > 0.f) ? (1.f / li) : 0.f;
        #pragma unroll
        for (int d = 0; d < kD128; d++) {
            O[bh_off + my_qrow * kD128 + d] = __float2half(o_acc[d] * inv);
        }
    }
}

} // namespace

extern "C" void launch_flash_attention_forward(
    const half* Q, const half* K, const half* V, half* O,
    int batch, int heads, int seq_len, int head_dim,
    float scale, bool causal
) {
    if (head_dim == 64) {
        flash_attention_forward_kernel_d64_wmma_qk_pv_br64_bc64<<<
            dim3((seq_len + kBrFinal64 - 1) / kBrFinal64, batch * heads),
            dim3(kThreads),
            kSmemBytesD64Final
        >>>(Q, K, V, O, seq_len, scale, causal);
    } else if (head_dim == 128) {
        flash_attention_forward_kernel_d128_wmma_qk_pv_br32<<<
            dim3((seq_len + kBrFinal128 - 1) / kBrFinal128, batch * heads),
            dim3(kThreads),
            kSmemBytesD128Final
        >>>(Q, K, V, O, seq_len, scale, causal);
    } else {
        fprintf(stderr, "[flash_attn] head_dim=%d not supported\n", head_dim);
        return;
    }
    cudaDeviceSynchronize();
}

extern "C" void launch_flash_attention_forward_scalar(
    const half* Q, const half* K, const half* V, half* O,
    int batch, int heads, int seq_len, int head_dim,
    float scale, bool causal
) {
    if (head_dim != 64) {
        fprintf(stderr, "[flash_attn_scalar] head_dim=%d not supported\n", head_dim);
        return;
    }
    flash_attention_forward_kernel_d64_scalar<<<
        dim3((seq_len + kBrScalar64 - 1) / kBrScalar64, batch * heads),
        dim3(kBrScalar64)
    >>>(Q, K, V, O, seq_len, scale, causal);
    cudaDeviceSynchronize();
}

extern "C" void launch_flash_attention_forward_wmma_qk(
    const half* Q, const half* K, const half* V, half* O,
    int batch, int heads, int seq_len, int head_dim,
    float scale, bool causal
) {
    if (head_dim != 64) {
        fprintf(stderr, "[flash_attn_wmma_qk] head_dim=%d not supported\n", head_dim);
        return;
    }
    flash_attention_forward_kernel_d64_wmma_qk<<<
        dim3((seq_len + kBrQK64 - 1) / kBrQK64, batch * heads),
        dim3(kThreads)
    >>>(Q, K, V, O, seq_len, scale, causal);
    cudaDeviceSynchronize();
}

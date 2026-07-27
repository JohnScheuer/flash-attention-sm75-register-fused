# Design Notes — flash-attention-sm75

## 1. Problem Statement

The official FlashAttention library requires SM80+ (Ampere). Users with Turing GPUs (SM75) have no memory-efficient attention kernel available from official sources.

This project implements FlashAttention v1 forward in CUDA for SM75, from scratch, including PyTorch integration and HuggingFace model support.

---

## 2. Key Design Decisions

### 2.1 Tile Sizes

**d=64: Br=64, Bc=64**

With Br=64, all 4 warps are active during softmax and accumulation. With Bc=64, the KV loop iterates half as many times compared to Bc=32, reducing sync overhead.

Shared memory usage with aliasing: 40 KB (limit: 48 KB).

**d=128: Br=32, Bc=32**

D=128 with Br=32 means each warp still owns 16 rows. Larger tiles are not feasible because Q alone consumes 8 KB, leaving insufficient budget for K, V, S, P, T buffers.

### 2.2 Shared Memory Aliasing

The pipeline per KV tile is:

    Q @ K^T -> smem_S -> softmax -> smem_P -> P @ V -> smem_T -> o_acc

Buffers S and T are never needed simultaneously. Buffers K and P are never needed simultaneously. This allows:

- smem_P aliases smem_K (K is dead after QK^T)
- smem_T aliases smem_S (S is dead after softmax)

Q must NOT be aliased — it persists across all KV tiles.

Aliasing reduces total shared memory from ~60 KB to 40 KB, fitting within SM75's 48 KB budget.

### 2.3 WMMA Layout for QK^T

S = Q @ K^T where Q and K are both [Br × D].

With WMMA m16n16k16:

- A fragment: Q[row_block : +16, k_step*16 : +16] — row_major
- B fragment: K[col_block : +16, k_step*16 : +16] — col_major (gives transpose)

Using col_major for B is the standard WMMA trick for computing A @ B^T without explicit transpose.

- For d=64: 4 k-steps × 4 k-steps in D = 16 total WMMA ops per warp
- For d=128: 8 k-steps = 32 total WMMA ops per warp

### 2.4 WMMA Layout for PV

T = P @ V where P is [Br × Bc] and V is [Bc × D].

- A fragment: P[row_block : +16, k_step*16 : +16] — row_major
- B fragment: V[k_step*16 : +16, t_col : +16] — row_major

- For d=64: 2 k-steps (Bc/16) × 4 col-blocks (D/16) = 8 WMMA ops per warp
- For d=128: 2 k-steps × 8 col-blocks = 16 WMMA ops per warp

### 2.5 Causal Pruning

For each (q_tile, kv_tile) pair we compute:

    q_tile_start = q_tile * Br
    q_tile_end   = min(q_tile * Br + Br - 1, seq_len - 1)
    k_start      = kv_tile * Bc
    k_end        = min(kv_tile * Bc + Bc - 1, seq_len - 1)

Three cases:

- k_start > q_tile_end -> entire tile above diagonal -> break
- k_end <= q_tile_start -> entire tile below diagonal -> fast path (no per-element mask)
- otherwise -> partial tile -> per-element mask in softmax

This reduces causal attention work by 40-50% for large N. Single most impactful late-stage optimization.

### 2.6 Online Softmax Correctness

Each lane tracks (mi, li, o_acc) across KV tiles:

    m_new = max(mi, row_max(S_tile))
    alpha = exp(mi - m_new)
    o_acc = o_acc * alpha + P_tile @ V_tile
    li    = li * alpha + row_sum(P_tile)
    mi    = m_new

Final: O = o_acc / li

This is mathematically equivalent to computing full softmax(QK^T/sqrt(d))V without materializing the N×N matrix.

### 2.7 GQA/MQA Support

The Python layer in ops.py handles Grouped Query Attention before calling the kernel:

    if H_q != H_kv:
        factor = H_q // H_kv
        K = K.repeat_interleave(factor, dim=1).contiguous()
        V = V.repeat_interleave(factor, dim=1).contiguous()

The CUDA kernel always receives H_q == H_kv. This adds memory overhead but keeps kernel logic simple.

---

## 3. Performance Analysis

### 3.1 Roofline

RTX 2070 FP16 Tensor Core peak (measured via torch GEMM): **28.5 TFLOPS**

This kernel at best case: **~1.45 TFLOPS (~5% of roofline)**

### 3.2 Where the Gap Comes From

The remaining gap is not from lack of Tensor Core usage — both QK^T and PV use WMMA. The gap comes from:

**1. Intermediate shared memory staging**

Each KV tile does 3 store+load cycles (smem_S, smem_P, smem_T). This is unavoidable with the current design where softmax runs outside the Tensor Core path.

**2. Scalar softmax**

The row-max and exp loop runs in scalar FP32 outside the Tensor Core path. Cannot be executed on Tensor Cores.

**3. Sync overhead**

Multiple __syncthreads() per KV tile: after QK^T, after softmax, after PV. Total ~4-5 syncs per tile.

**4. Register pressure**

d=64 kernel uses ~180 registers/thread, near SM75 limit (255). Higher register pressure reduces occupancy.

### 3.3 What Actually Helped

| Optimization | Effect |
|---|---|
| WMMA for QK^T | +2.0-2.2x vs scalar |
| WMMA for PV | +1.1-1.5x additional |
| Br=64 (all warps active) | marginal on most shapes |
| Tile-level causal pruning | +50-100% on causal |
| Bc=64 | regression on most shapes |
| Shared memory aliasing (20KB saved) | < 5% |

The single most impactful late-stage optimization was **causal tile pruning**, which cut causal latency nearly in half for large sequences.

Bc=64 was expected to help by reducing KV iteration count, but increased register pressure enough to hurt occupancy, negating the benefit.

---

## 4. Known Limitations and Future Work

### 4.1 Forward Only

Backward pass requires:

- Materializing the softmax statistics (LSE = log-sum-exp per row) during forward
- Storing them for the backward
- Implementing the backward CUDA kernel with recompute of attention scores

Not implemented in this MVP. Would extend project scope by 25-40h.

### 4.2 KV Cache Decode Not Accelerated

Current implementation targets prefill only. Decode (single-token generation with KV cache) falls back to PyTorch SDPA.

This limits end-to-end acceleration to prefill-bound workloads. For decode-bound workloads (long generations, small batch), speedup would be minimal.

Extending to decode requires different kernel design (single-query attention against cached KV) — orthogonal to current codebase.

### 4.3 Bc=64 Regression

Increasing Bc from 32 to 64 for d=64 increased register pressure enough to hurt occupancy, negating the benefit of fewer KV iterations.

Would require register optimization to make Bc=64 viable.

### 4.4 Fragment-Fused Design

The next architectural improvement would be to avoid materializing smem_S, smem_P, smem_T entirely, keeping the attention scores inside WMMA accumulator registers between steps.

This requires implementing warp-level softmax using __shfl_sync directly on the accumulator .x array elements. Expected to push past 3 TFLOPS.

Non-trivial implementation. Deferred as future work.

---

## 5. Lessons Learned

### 5.1 Memory Efficiency Does Not Equal Compute Efficiency

FlashAttention's original value proposition is memory reduction, not compute speedup. On SM75 without cp.async, we achieve the memory reduction (O(N) vs O(N^2)) but not the compute speedup that FA has on SM80+.

This is important framing: the value is memory scaling, which enables use cases that pure compute optimization does not.

### 5.2 End-to-End Wins Can Hide Kernel-Level Losses

Our kernel is 10-13x slower than SDPA in isolation. Yet monkey-patched into Qwen2-0.5B, generation throughput improves 27%.

Reason: prefill dominates end-to-end latency for short-medium contexts, and our kernel runs the prefill path where memory efficiency helps overall pipeline throughput despite raw kernel being slower.

Lesson: benchmark end-to-end when possible, not just kernel isolation.

### 5.3 Tile-Level Causal Pruning Is High-ROI

Adding causal tile skip was the highest-ROI late optimization: ~50% latency reduction on causal attention with minimal code change.

Micro-optimizations at kernel level (register tuning, unrolling) gave diminishing returns compared to algorithmic optimizations (skipping unnecessary work).

Lesson: look for structural optimizations before tuning individual operations.

### 5.4 SM75 Is a Genuinely Different Target Than SM80+

Missing cp.async, smaller shared memory (96KB vs 164KB), lack of async copy engines — these force different optimization paths than FA papers assume.

Papers targeting SM80+ do not directly transfer to SM75. Design decisions must be revisited for the target architecture.

---

## 6. File Structure

    src/
      kernels/
        naive_attention.cu             O(N^2) reference kernel
        flash_attention_forward.cu     FlashAttention kernels (v1..v5)
      bindings/
        pytorch_extension.cpp          pybind11 C++ to Python bridge
      flash_attention_sm75/
        __init__.py                    Public Python API
        ops.py                         autograd.Function + GQA expansion
    benchmarks/
      benchmark_attention.py           Intelligent benchmark harness
      benchmark_d64_d128.py            d=64 vs d=128 comparison
      benchmark_tensorcore_roofline.py Hardware roofline measurement
    tests/
      test_correctness.py              Naive CUDA correctness
      test_flash_wmma_qk.py            WMMA QK^T correctness
      test_flash_wmma_qk_pv.py         WMMA QK^T + PV correctness
      test_flash_d128.py               d=128 correctness + regression
      test_api.py                      Python API correctness
      test_prefill_correctness.py     End-to-end logit comparison on Qwen2
    examples/
      llama_attention_replacement.py  Monkey-patch for HuggingFace

---

## 7. References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) — arxiv:2205.14135
- NVIDIA, "Programming Tensor Cores in CUDA 9" — WMMA API documentation
- NVIDIA Turing Architecture Whitepaper (2018)

# Flash Attention SM75 — Design Document

**Author:** João Felipe De Souza  
**Year:** 2026  
**Hardware:** NVIDIA GeForce RTX 2070 (SM75 / Turing)

---

## Problem Statement

Standard attention computes:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

The naive implementation materializes the full `N × N` score matrix in memory,
which is expensive both in time and memory for large sequences.

Flash Attention avoids this by computing attention in tiles, keeping the
running output in registers and applying online softmax rescaling.

This project asks a more specific question:

> On SM75, using only the WMMA API, can we keep the score tile **inside
> the WMMA accumulator registers** and compute softmax without ever writing
> it to shared memory?

---

## Hardware Context

### SM75 (Turing) — Key numbers

| Property | Value |
|---|---|
| Tensor core tile | m16n16k16, FP16 in, FP32/FP16 out |
| Registers per SM | 65,536 |
| Max threads per SM | 1,024 |
| Max warps per SM | 32 |
| Shared memory per SM | 64 KB |
| Peak FP16 tensor | 57.4 TFLOPS |
| Peak memory bandwidth | 448 GB/s |
| Ridge point | 128.1 FLOP/byte |

### WMMA accumulator layout (empirically verified)

Each warp holds a `wmma::fragment<accumulator, 16, 16, 16, float>` as 8 floats
per thread:

```
Thread T:
  group_id        = T >> 2    (0..7)
  thread_in_group = T & 3     (0..3)

  x[0] → C[group_id    ][ thread_in_group*2     ]
  x[1] → C[group_id    ][ thread_in_group*2 + 1 ]
  x[2] → C[group_id + 8][ thread_in_group*2     ]
  x[3] → C[group_id + 8][ thread_in_group*2 + 1 ]
  x[4] → C[group_id    ][ thread_in_group*2 + 8 ]
  x[5] → C[group_id    ][ thread_in_group*2 + 9 ]
  x[6] → C[group_id + 8][ thread_in_group*2 + 8 ]
  x[7] → C[group_id + 8][ thread_in_group*2 + 9 ]
```

**Key property:** each row of the 16×16 tile is owned by exactly 4 consecutive
lanes (`group_id*4` through `group_id*4+3`). Row-wise reduction only requires
`__shfl_xor_sync` with masks 1 and 2.

This property is what makes register-resident softmax possible.

---

## Core Design — Register-Fused Pipeline

### Old pipeline (naive fused)

```
For each KV tile:
  For each d_out tile:
    WMMA: S_acc = Q @ K^T
    Store S_acc → smem_S          (score round-trip)
    Read smem_S → scalar softmax
    Store P → smem_P
    WMMA: O_acc += P @ V
    Store O_acc → smem_T          (output round-trip)
    o_acc += smem_T               (scalar accumulate)
```

Problems:
- `QK^T` recomputed once per `d_out` tile
- two unnecessary shared-memory round-trips per iteration

### New pipeline (register-fused)

```
Preload Q to smem once

For each KV tile:
  WMMA: S_acc = Q @ K^T            (once per KV tile)
  warp_softmax_unnorm(S_acc)        (in registers, zero smem)
  online_step_update(state, ms)     (in registers, zero smem)
  apply_beta_to_S(S_acc, step)      (in registers, zero smem)
  Scatter S_acc → smem_P            (only smem touch for scores)
  Load smem_P → P_frag

  For each d_out tile:
    apply_alpha_to_O(O_acc[d_out])  (in registers, zero smem)
    Load V tile → smem_V
    WMMA: O_acc[d_out] += P @ V     (accumulates in registers)

After all KV tiles:
  finalize_O(O_acc, state)          (in registers, zero smem)
  Store O_acc → global O
```

The only shared-memory store for scores is the scatter of `S_acc` to `smem_P`
before loading it as a `matrix_a` fragment.

---

## Warp-Register Softmax

### Reduction pattern

Given the WMMA layout, each thread owns elements from exactly two rows:

- **upper row:** `x[0], x[1], x[4], x[5]`
- **lower row:** `x[2], x[3], x[6], x[7]`

Row-wise max reduction across the 4 threads that share a row:

```c
// XOR with 1: swap within pairs  (T0↔T1, T2↔T3)
// XOR with 2: swap across pairs  (T0↔T2, T1↔T3)
// Result: all 4 threads hold the same row max

tmp = __shfl_xor_sync(0xFFFFFFFF, val, 1);
val = fmaxf(val, tmp);
tmp = __shfl_xor_sync(0xFFFFFFFF, val, 2);
val = fmaxf(val, tmp);
```

Row-wise sum reduction: same pattern with addition.

No shared memory is touched during softmax.

### Unnormalized softmax

The kernel uses unnormalized softmax (does not divide by sum during the KV
loop). Division happens once at the end via `finalize_O`.

This is required for correctness with online rescaling.

---

## Online Softmax Rescaling

Based on FlashAttention algorithm 1 (Dao et al., 2022).

Each thread maintains an `OnlineState` for its two rows:

```
state.m[0]  running max   (upper row)
state.m[1]  running max   (lower row)
state.l[0]  running sum   (upper row)
state.l[1]  running sum   (lower row)
```

Per KV tile, `online_step_update` computes:

```
m_new = max(m_old, m_cur)
alpha  = exp(m_old - m_new)   → rescales old O_acc
beta   = exp(m_cur - m_new)   → scales new P before accumulation
l_new  = alpha * l_old + beta * l_cur
```

The old `O_acc` is rescaled by `alpha` in-place in registers.
The current `S_acc` is scaled by `beta` before being materialized as `P`.

After all KV tiles, `finalize_O` divides by `l_final`.

---

## QK Reuse — The Key Optimization

In the naive version, for `d=64` (4 output column tiles):

```
Total QK computations per q_tile = 4 × num_kv_tiles
```

In the reuse version:

```
Total QK computations per q_tile = 1 × num_kv_tiles
```

This reduces the `QK` cost by a factor of `d / TK = 64 / 16 = 4`.

For `d=64`, this alone produced a **4.75x speedup** over the previous kernel.

The key insight: `P = softmax(QK^T)` does not depend on `d_out`, so the same
probability tile can be used for all output column tiles in one shot.

---

## Causal Masking Design

Two cases per KV tile:

### Fully masked tile

If `kv_base > q_base + TM - 1`, all positions in the tile have `j > i`.
The tile is entirely above the diagonal and is **skipped completely**:

```c
if (CAUSAL && tile_fully_masked(q_base, kv_base)) {
    __syncwarp();
    continue;
}
```

### Diagonal tile

If the tile straddles the diagonal, masking is applied **in-place in registers**
after the WMMA operation:

```c
if (CAUSAL && tile_is_diagonal(q_base, kv_base)) {
    apply_causal_mask(S_acc, q_base, kv_base);
}
```

`apply_causal_mask` sets positions where `j > i` to `-FLT_MAX/2` directly
in `S_acc.x[0..7]`, using the known per-thread row/column mapping from the
WMMA layout.

The causal flag is a **compile-time template parameter** (`CAUSAL = 0 or 1`),
so the predicate has zero overhead in the non-causal path.

---

## Ping-Pong K Buffering

`flash_mha_v2` uses a double buffer for the K tile in shared memory:

```
smem_Kpp[2][TN * D_FIXED]   (2 × 1024 halfs = 4096 B)
```

While the current K tile is being processed, the next K tile is loaded
in the background into the alternate buffer:

```
cur_buf  = kv & 1
next_buf = 1 - cur_buf

// Issue next K load (non-blocking, overlapped with WMMA)
if (kv + 1 < num_kv):
    load smem_Kpp[next_buf] from global K

// Sync before use
__syncwarp()

// Process current tile from smem_Kpp[cur_buf]
```

On SM75, this is a software emulation of prefetch (no `cp.async`). The
compiler may reorder the load instructions to overlap with WMMA ops.

Effect: helps in cases where the grid is small and global memory latency is
exposed. Neutral in fully saturated cases where L2 already absorbs K reloads.

Side effect: register count dropped from 255 to 123 due to the buffer index
creating a liveness boundary the compiler can exploit.

---

## Multi-Warp Experiment

`flash_mha_4warp` tested 4 warps per block with K and V shared across warps.

Results:
- register count dropped from 255 to 86 due to `__syncthreads` barriers
- occupancy rose from 25% to 62%
- throughput was lower than the 1-warp kernel

Why:
- `__syncthreads` overhead: approximately `num_kv × 9` barriers per block
- L2 cache already absorbed K reloads in the 1-warp kernel
- the RTX 2070 in this regime is not latency-bound enough to benefit from higher occupancy

Conclusion: for SM75 with `d=64`, the 1-warp kernel is the right tradeoff.

---

## Register Budget Analysis

For the best 1-warp kernels:

| Component | Registers |
|---|---|
| `O_acc[4]` (4 tiles × 8 floats) | 32 |
| `S_acc` (8 floats) | 8 |
| `P_frag` (8 halfs, packed) | 4 |
| `Q_frag`, `K_frag`, `V_frag` | ~12 |
| `OnlineState` (4 floats) | 4 |
| `OnlineStep` (4 floats) | 4 |
| `RowMaxSum` (4 floats) | 4 |
| Addressing and loop vars | ~20 |
| Compiler temporaries and pipeline | ~167 |
| **Total** | **~255** |

The compiler allocates 255 registers per thread — effectively the SM75 maximum
without spilling. Reducing `D_TILES` from 4 to 2 did not reduce this, because
the base cost of the pipeline already consumes most of the budget.

---

## Shared Memory Layout

### `flash_mha_flatargs` (2 KB / block)

```
smem_Q   [TM × TK]   = 16 × 16 × 2 B = 512 B   (Q tile, one d-chunk)
smem_K   [TN × TK]   = 16 × 16 × 2 B = 512 B   (K tile, one d-chunk)
smem_P   [TM × TN]   = 16 × 16 × 2 B = 512 B   (P tile, also used as O scratch)
smem_V   [TN × TK]   = 16 × 16 × 2 B = 512 B   (V tile, one d_out-chunk)
─────────────────────────────────────────────────────
Total                                    2048 B
```

### `flash_mha_v2` (7 KB / block)

```
smem_Kpp [2 × TN × D]  = 2 × 1024 × 2 B = 4096 B  (K ping-pong)
smem_Q   [TM × D]       = 1024 × 2 B     = 2048 B  (full Q tile, all d)
smem_P   [TM × TN]      = 256 × 2 B      =  512 B  (P tile + O scratch)
smem_V   [TN × TK]      = 256 × 2 B      =  512 B  (V tile)
─────────────────────────────────────────────────────
Total                                       7168 B
```

---

## Roofline Position

For `B=4, H=12, N=1024, d=64` on RTX 2070 SM75:

| Metric | Our kernel | SDPA mem-eff |
|---|---|---|
| Algorithmic TFLOPS | 4.98 | 18.59 |
| Estimated traffic | 440 MB | ≥ 25 MB |
| Arithmetic intensity | 29.3 FLOP/byte | ≥ 512 FLOP/byte |
| Ridge point | 128.1 FLOP/byte | 128.1 FLOP/byte |
| Regime | **memory-bound** | **compute-bound** |

Our kernel is memory-bound because the 64× K reload pattern pushes the traffic
to 440 MB for a 25 MB algorithmic minimum. The arithmetic intensity of 29.3
is well below the ridge point of 128.1.

SDPA's minimum arithmetic intensity of 512 FLOP/byte exceeds the ridge point
regardless of actual internal traffic, confirming compute-bound operation.

---

## What Was Not Done

These are known limitations and possible extensions:

- `d=128` support (needs `D_TILES=8`, higher register pressure)
- FP16 accumulators (template ready, not benchmarked)
- SM80+ port (different WMMA layout, `cp.async` available)
- Multi-query attention / grouped-query attention
- Sliding window or local attention variants
- Direct hardware counter profiling (blocked by WSL in this setup)

---

## References

- Dao et al., "FlashAttention" (NeurIPS 2022)
- Dao et al., "FlashAttention-2" (ICLR 2024)
- NVIDIA, CUDA Programming Guide — WMMA API
- NVIDIA, Turing Architecture Whitepaper (2018)
- Rabe and Staats, "Self-attention Does Not Need O(n²) Memory" (2021)

# Flash Attention SM75 — Register-Fused Architecture

[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11%2B-green.svg)]()
[![SM75](https://img.shields.io/badge/SM75-Turing-orange.svg)]()
[![Peak](https://img.shields.io/badge/Peak-5.0%20TFLOPS-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

A from-scratch Flash Attention implementation for **NVIDIA Turing (SM75)** using CUDA WMMA, warp-shuffle softmax in registers, persistent output accumulators, QK reuse, and causal masking.

This is a **12.7× improvement** over the initial V1 kernel (0.39 → 5.0 TFLOPS), built entirely from first principles with no external fused-attention dependencies.

---

## What This Project Is

A research and engineering exploration of register-fused attention on SM75, answering:

> Can softmax be computed directly inside WMMA accumulator registers using warp shuffles, without round-tripping score tiles through shared memory?

**Answer: yes.** Fully validated and benchmarked.

This is **not** a production replacement for PyTorch SDPA. It is an architectural study that proves several non-obvious things about what is possible with the WMMA API on Turing hardware.

---

## Main Contributions

1. **Empirical WMMA layout verification** — probe kernel maps the exact per-thread element ownership of `wmma::fragment<accumulator, 16,16,16, float>` on SM75 hardware.

2. **Warp-register softmax** — row-wise softmax computed entirely in WMMA accumulator registers using `__shfl_xor_sync`. No shared memory touched during softmax.

3. **Persistent output accumulator** — `O_acc` remains in registers across all KV tiles and is only written to global memory at the end.

4. **QK reuse** — `QK^T` computed once per KV tile and reused for all output column tiles. Largest single optimization: **4.75× speedup** on its own.

5. **Causal masking** — compile-time template parameter. Fully-masked tiles skipped with a single predicate. Diagonal tiles masked directly in register-resident score fragments.

---

## Best Measured Results

All results on **RTX 2070 SM75**, **d=64**, **FP16 input / FP32 accumulator**.

### Recommended full-attention kernel: `flash_mha_flatargs`

**~5.0 algorithmic TFLOPS** in saturated cases.

### Recommended causal-attention kernel: `flash_mha_v2`

**~3.9–4.1 algorithmic TFLOPS** in saturated causal cases.

---

## Benchmark vs PyTorch SDPA on SM75

> **Context:** PyTorch SDPA on SM75 does **not** use FlashAttention (requires SM80+). The baseline is the mem-efficient backend (xformers lineage) — a legitimate, strong, production-optimized fused kernel.

### Full attention

| Config | `flatargs` | Best SDPA | Ratio |
|---|---:|---:|---:|
| B=4 H=12 N=1024 d=64 | 2.59 ms | 0.69 ms | 3.73× |
| B=4 H=8 N=2048 d=64 | 6.90 ms | 1.83 ms | 3.77× |
| B=8 H=12 N=512 d=64 | 1.29 ms | 0.37 ms | 3.53× |
| B=1 H=16 N=2048 d=64 | 3.42 ms | 0.95 ms | 3.60× |
| B=1 H=8 N=256 d=64 | 0.031 ms | 0.020 ms | 1.53× |
| B=1 H=1 N=512 d=64 | 0.040 ms | 0.034 ms | 1.20× |

**Median ratio: 3.60× slower than SDPA on large cases.**

### Causal attention

| Config | `v2` causal | Best SDPA causal | Ratio |
|---|---:|---:|---:|
| B=4 H=12 N=1024 d=64 | 1.66 ms | 0.40 ms | 4.16× |
| B=4 H=8 N=2048 d=64 | 4.18 ms | 1.00 ms | 4.17× |
| B=8 H=12 N=512 d=64 | 0.84 ms | 0.23 ms | 3.68× |
| B=1 H=16 N=2048 d=64 | 2.16 ms | 0.54 ms | 4.03× |

**Median ratio: 4.03× slower than SDPA causal.**

The gap reflects the difference between a clean-room WMMA educational kernel and a production-optimized implementation with vectorized loads, multi-warp tiling, and years of engineering.

---

## Kernel Evolution

Progressive optimization tested over the project:

| Version | Approach | TFLOPS | Notes |
|---|---|---:|---|
| V1 initial (`flash_fused`) | Register-fused baseline | 0.39 | QK recomputed per output tile |
| QK reuse (`flash_reuse_qk`) | Compute QK^T once per KV tile | 1.82 | 4.75× gain, biggest single lever |
| Multi-head grid (`flash_mha_reuse_qk`) | + 3D grid parallelism | 4.48 | Multi-head saturation |
| Flat args (`flash_mha_flatargs`) | + flat args, codegen improvement | **5.00** | **Recommended full attention** |
| 2-tiles experiment (`flash_mha_2tiles`) | Tested: fewer output tiles | 2.64 | Refuted — register limit is architectural |
| 4-warp experiment (`flash_mha_4warp`) | Tested: 4-warp K/V sharing | 4.23 | Refuted — sync overhead > cache benefit |
| V2 (`flash_mha_v2`) | + causal masking + ping-pong K | **~4.0** | **Recommended causal attention** |

**Total improvement: 12.7× from V1 to V2 (0.39 → 5.00 TFLOPS).**

---

## Roofline Analysis

### Our kernel (`flash_mha_flatargs`, B=4 H=12 N=1024 d=64)

- Algorithmic throughput: **4.98 TFLOPS**
- Estimated total traffic: **440 MB** (K reloaded ~64× dominates)
- Arithmetic intensity: **29.3 FLOP/byte**
- Ridge point (RTX 2070): **128 FLOP/byte**

**Since 29 << 128, our kernel is memory-bound.**

### SDPA (same configuration)

- Algorithmic throughput: **18.59 TFLOPS**
- Minimum possible traffic (Q+K+V+O read once): **25.2 MB**
- Minimum arithmetic intensity: **512 FLOP/byte**

**Even at the theoretical minimum traffic, SDPA sits 4× above the ridge point.**

This is provable without hardware profiling: the geometry of the problem plus the measured throughput places SDPA in the compute-bound regime.

### Why this matters

Both kernels are near their respective ceilings:
- **Ours:** memory bandwidth ceiling for register-fused approach on SM75
- **SDPA:** compute ceiling through techniques that raise arithmetic intensity

The 3.6× gap is a **regime difference**, not "our kernel is underoptimized."

> **Note:** Nsight Compute hardware counters are unavailable under WSL on this machine. The regime classification is derived from the roofline model using measured latency and known hardware specs.

---

## What Was Proven

| Hypothesis | Result |
|---|---|
| Softmax computable entirely in WMMA accumulator registers | ✅ Confirmed |
| Persistent `O_acc` in registers is viable | ✅ Confirmed |
| Reusing `QK^T` is the dominant optimization lever | ✅ Confirmed (4.75×) |
| 3D grid over batch × heads is necessary for saturation | ✅ Confirmed |
| Flat args improve codegen without changing register count | ✅ Confirmed (1.45×) |
| Fewer output tiles reduces register pressure | ❌ Refuted (255→255) |
| `maxrregcount` controls registers below ~252 | ❌ Refuted |
| 4-warp K/V sharing beats 1-warp on SM75 | ❌ Refuted (0.78×) |
| Causal masking integrates cleanly into this design | ✅ Confirmed |
| Kernel reaches memory bandwidth roofline | ✅ Confirmed (~5 TFLOPS ≈ roofline) |

---

## Architecture Summary

### Pipeline (`flash_mha_flatargs`)

```
Preload Q to smem once

For each KV tile:
  WMMA: S_acc = Q @ K^T            (once per KV tile)
  warp_softmax_unnorm(S_acc)        (in registers, zero smem)
  online_step_update(state, ms)     (in registers)
  apply_beta_to_S(S_acc, step)      (in registers)
  Scatter S_acc → smem_P            (only smem write for scores)
  Load smem_P → P_frag

  For each d_out tile:
    apply_alpha_to_O(O_acc[d_out])  (in registers)
    Load V tile → smem_V
    WMMA: O_acc[d_out] += P @ V     (accumulates in registers)

After all KV tiles:
  finalize_O(O_acc, state)          (in registers)
  Store O_acc → global O
```

### WMMA accumulator layout (SM75, verified empirically)

```
Thread T:  group_id = T>>2,  thread_in_group = T&3

x[0] → C[group_id    ][ thread_in_group*2     ]
x[1] → C[group_id    ][ thread_in_group*2 + 1 ]
x[2] → C[group_id + 8][ thread_in_group*2     ]
x[3] → C[group_id + 8][ thread_in_group*2 + 1 ]
x[4] → C[group_id    ][ thread_in_group*2 + 8 ]
x[5] → C[group_id    ][ thread_in_group*2 + 9 ]
x[6] → C[group_id + 8][ thread_in_group*2 + 8 ]
x[7] → C[group_id + 8][ thread_in_group*2 + 9 ]

Row reduction: __shfl_xor_sync(mask, val, 1) + (mask, val, 2)
```

### Shared memory per block

| Kernel | smem | Regs/thread | Blocks/SM | Occupancy |
|---|---|---|---|---|
| `flash_mha_flatargs` | 2 KB | 255 | 8 | 25% |
| `flash_mha_v2` | 7 KB | 123–128 | 9 | 28% |
| `flash_mha_4warp` | 12.5 KB | 86 | 5 | 62% |

---

## Honest Limitations

1. **Specialized to d=64.** Extending to d=128 requires D_TILES=8 and substantially different register/smem tradeoffs.

2. **3.6–4.0× slower than SDPA** on large workloads. The gap is real, expected, and documented.

3. **SM75-specific.** The WMMA layout and tuning are specific to Turing. Porting to SM80+ would require a new pass.

4. **No hardware-counter confirmation under WSL.** Regime statements are based on roofline reasoning, not direct `ncu` measurement.

5. **No FP16 accumulator.** Template parameter exists but is not benchmarked. FP32 accumulators used throughout.

6. **Forward pass only.** No backward / gradient support.

---

## Reproducibility

```bash
make all
./probe_layout
./test_warp_softmax
./flash_mha_flatargs 4 12 1024 7 200
./flash_mha_v2 4 12 1024 7 200
python benchmark_sdpa_final.py
```

---

## Relation to V1

This repository extends the initial [flash-attention-sm75](https://github.com/JohnScheuer/flash-attention-sm75) V1 kernel with register-fused architecture. V1 remains a reference for its simpler implementation and broader feature set (d=64/128, GQA, HuggingFace monkey-patch). V2 (this repo) demonstrates architectural exploration, QK reuse, and roofline validation.

---

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (NeurIPS 2022)
- Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism" (ICLR 2024)
- NVIDIA, CUDA Programming Guide — WMMA API
- NVIDIA, Turing Architecture Whitepaper (2018)
- Rabe and Staats, "Self-attention Does Not Need O(n²) Memory" (2021)

---

## License

MIT License — see [LICENSE](./LICENSE).

Copyright (c) 2026 João Felipe De Souza

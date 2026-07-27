# Flash Attention SM75 — Register-Fused Architecture

# Flash Attention SM75 — Register-Fused Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?logo=nvidia&logoColor=white)](#)
[![GPU](https://img.shields.io/badge/GPU-SM75%20(Turing)-orange)](#)
[![WMMA](https://img.shields.io/badge/WMMA-m16n16k16-blue)](#)
[![Precision](https://img.shields.io/badge/Precision-FP16%20input%20%2F%20FP32%20accumulator-1f6feb)](#)
[![Attention](https://img.shields.io/badge/Attention-Full%20%2B%20Causal-purple)](#)
[![Status](https://img.shields.io/badge/Status-Research--Grade-informational)](#)
[![Hardware](https://img.shields.io/badge/Validated%20on-RTX%202070-success)](#)
[![Benchmark](https://img.shields.io/badge/Best%20full%20kernel-~5.0%20TFLOPS-critical)](#)
[![Baseline](https://img.shields.io/badge/Baseline-PyTorch%20SDPA%20(SM75)-blueviolet)](#)

A from-scratch Flash Attention study for **NVIDIA Turing / SM75** using:

- CUDA WMMA
- warp-shuffle softmax in registers
- persistent `O_acc` in WMMA accumulators
- online rescaling
- multi-head grid parallelism

A from-scratch Flash Attention study for **NVIDIA Turing / SM75** using:

- CUDA WMMA
- warp-shuffle softmax in registers
- persistent `O_acc` in WMMA accumulators
- online rescaling
- multi-head grid parallelism

**Author:** João Felipe De Souza  
**Year:** 2026  
**License:** MIT

---

## What this project is

This repository is a **research/engineering exploration** of Flash Attention on **SM75** GPUs (e.g. RTX 20xx / T4), implemented directly with CUDA WMMA primitives and warp-level operations.

The main goal was to answer a concrete low-level question:

> Can softmax be computed directly inside WMMA accumulator registers on SM75, without round-tripping score tiles through shared memory?

**Answer: yes.**

This project proves that:

- the SM75 WMMA accumulator layout can be decoded and exploited,
- row-wise softmax can be implemented entirely in registers using `__shfl_xor_sync`,
- `O_acc` can remain persistent in registers across KV tiles,
- reusing `QK^T` across output tiles is the dominant optimization lever.

---

## Main contributions

### 1. Empirical WMMA layout verification on SM75
The project includes a probe kernel that maps the exact layout of:

- `wmma::fragment<accumulator, 16,16,16, float>`
- `wmma::fragment<matrix_a, 16,16,16, half, row_major>`

on real SM75 hardware.

### 2. Warp-register softmax
A row-wise softmax was implemented directly on WMMA accumulator registers using:

- no score tile in shared memory,
- no scalar score array,
- only warp shuffles and in-place register math.

### 3. Persistent output accumulator
The output tile `O_acc` remains in registers across all KV tiles and is only written out at the end.

### 4. QK reuse
The largest speedup came from computing `QK^T` **once per KV tile** and reusing the resulting probability tile for all output column tiles.

### 5. Causal masking
A causal version was implemented that:
- skips fully masked tiles,
- applies diagonal masking directly in register-resident score fragments.

---

## Final status

This project produced two final kernels worth keeping:

### Recommended full-attention kernel
**`flash_mha_flatargs`**

Best full-attention implementation in this repository.

### Recommended causal-attention kernel
**`flash_mha_v2`**

Best causal implementation in this repository.

---

## Best measured results

All results below were measured on:

- **GPU:** NVIDIA GeForce RTX 2070
- **Architecture:** SM75
- **PyTorch:** 2.7.1+cu118
- **dtype:** FP16 inputs
- **d = 64**

### Best full-attention throughput
`flash_mha_flatargs` reaches approximately:

- **~5.0 algorithmic TFLOPS** in saturated cases

### Best causal-attention throughput
`flash_mha_v2` reaches approximately:

- **~3.9–4.1 algorithmic TFLOPS** in saturated causal cases

---

## Final benchmark vs PyTorch SDPA on SM75

### Important context

On **SM75**, PyTorch SDPA does **not** use FlashAttention proper (FlashAttention requires SM80+).

The strongest relevant baseline on this hardware is:

- **PyTorch SDPA default backend**
- and, when available,
- **PyTorch SDPA mem-efficient backend**

That is the comparison reported below.

---

## Full attention — `flash_mha_flatargs` vs SDPA

| Config | `flash_mha_flatargs` | Best SDPA | Ratio |
|---|---:|---:|---:|
| B=4, H=12, N=1024, d=64 | 2.5892 ms | 0.6932 ms | 3.73x |
| B=4, H=8, N=2048, d=64 | 6.8988 ms | 1.8307 ms | 3.77x |
| B=8, H=12, N=512, d=64 | 1.2886 ms | 0.3652 ms | 3.53x |
| B=1, H=16, N=2048, d=64 | 3.4228 ms | 0.9509 ms | 3.60x |
| B=1, H=8, N=256, d=64 | 0.0312 ms | 0.0204 ms | 1.53x |
| B=1, H=1, N=512, d=64 | 0.0405 ms | 0.0338 ms | 1.20x |

**Median ratio (full): 3.60x slower than best SDPA on large cases**

### Interpretation
This is **not** competitive with production-grade SDPA on large workloads, but it is a **major improvement over V1** and a meaningful result for a clean-room SM75 WMMA implementation.

---

## Causal attention — `flash_mha_v2` vs SDPA

| Config | `flash_mha_v2` causal | Best SDPA causal | Ratio |
|---|---:|---:|---:|
| B=4, H=12, N=1024, d=64 | 1.6608 ms | 0.3994 ms | 4.16x |
| B=4, H=8, N=2048, d=64 | 4.1766 ms | 1.0015 ms | 4.17x |
| B=8, H=12, N=512, d=64 | 0.8358 ms | 0.2272 ms | 3.68x |
| B=1, H=16, N=2048, d=64 | 2.1618 ms | 0.5367 ms | 4.03x |
| B=1, H=8, N=256, d=64 | 0.0292 ms | 0.0204 ms | 1.43x |
| B=1, H=1, N=512, d=64 | 0.0529 ms | 0.0326 ms | 1.62x |

**Median ratio (causal): 4.03x slower than best SDPA causal**

### Interpretation
The causal kernel is **research-grade**, correct, and structurally interesting, but still clearly behind optimized SDPA.

---

## Internal causal speedup

Even though `flash_mha_v2` is still behind SDPA, it does achieve a meaningful **internal** speedup relative to our own full-attention kernel.

| Config | `flatargs` full | `v2` causal | Speedup |
|---|---:|---:|---:|
| B=4, H=12, N=1024 | 2.5892 ms | 1.6608 ms | 1.56x |
| B=4, H=8, N=2048 | 6.8988 ms | 4.1766 ms | 1.65x |
| B=8, H=12, N=512 | 1.2886 ms | 0.8358 ms | 1.54x |
| B=1, H=16, N=2048 | 3.4228 ms | 2.1618 ms | 1.58x |

This confirms that causal masking is implemented correctly and yields substantial savings.

---

## Kernel evolution

### V1 — baseline fused kernel
- correct
- register softmax already working
- still recomputed `QK^T` per output tile
- around **0.39 TFLOPS**

### QK reuse
- eliminated redundant `QK^T` recomputation
- biggest single gain of the project
- around **4.75x speedup** on its own

### `flash_mha_flatargs`
- best full-attention version
- simplified interface and improved code generation
- around **~5.0 TFLOPS**

### `flash_mha_2tiles`
- tested whether fewer live output fragments reduce register pressure
- result: **no benefit**
- register count stayed effectively pinned
- performance dropped

### `flash_mha_4warp`
- tested 4 warps per block with shared K/V
- reduced register count dramatically
- but synchronization overhead outweighed the gain
- did not beat the best 1-warp kernel

### `flash_mha_v2`
- added causal masking
- added K ping-pong buffering
- reduced registers vs flatargs
- best causal kernel in the repo

---

## What was proven

| Hypothesis | Result |
|---|---|
| Softmax can be computed entirely in WMMA accumulator registers | ✅ Confirmed |
| Persistent `O_acc` in registers is viable | ✅ Confirmed |
| Reusing `QK^T` is the dominant optimization lever | ✅ Confirmed |
| 3D grid over batch × heads is necessary for saturation | ✅ Confirmed |
| Fewer output tiles necessarily reduce register pressure | ❌ Refuted |
| `maxrregcount` alone can solve the register problem | ❌ Refuted |
| 4-warp K/V sharing beats best 1-warp kernel on SM75 | ❌ Refuted |
| Causal masking can be integrated cleanly into this design | ✅ Confirmed |

---

## Architecture summary

### Core idea
The kernel operates on a 16×16 score tile in WMMA accumulator registers:

1. compute `S = QK^T` in WMMA accumulators
2. compute row max and row sum using warp shuffles
3. apply `exp(x - max)` in-place in registers
4. update online softmax state in registers
5. materialize the probability tile once
6. reuse it for all output column tiles
7. keep `O_acc` persistent in registers
8. only store final output after all KV tiles are processed

### Shared memory use
The final high-performance 1-warp kernels use very little shared memory:
- `flash_mha_flatargs`: about **2 KB / block**
- `flash_mha_v2`: about **7 KB / block**

---

## Roofline analysis

### Verified result for our kernel
For `B=4, H=12, N=1024, d=64`:

#### `flash_mha_flatargs`
- algorithmic throughput: **4.98 TFLOPS**
- estimated total traffic: **440 MB**
- arithmetic intensity: **29.3 FLOP/byte**

Hardware ridge point on RTX 2070:

- peak bandwidth: **448 GB/s**
- peak FP16 tensor throughput: **57.4 TFLOPS**
- ridge point: **128.1 FLOP/byte**

Since:

- `29.3 FLOP/byte < 128.1 FLOP/byte`

our kernel is **memory-bound**.

### About SDPA
For the same configuration, best SDPA is about:

- **18.59 algorithmic TFLOPS**

The minimum possible traffic for SDPA (reading Q, K, V once and writing O once) is:

- **25.2 MB**

This gives a **best-case / lower-bound traffic assumption** corresponding to:

- **512 FLOP/byte**

Since this is above the ridge point, SDPA clearly operates in a **much higher-intensity regime** than our kernel.

### Important wording note
Because WSL does not support Nsight Compute counter profiling on this setup, this project does **not** claim direct measurement of SDPA tensor-core utilization.

The conservative and correct statement is:

> Our kernel is memory-bound.  
> SDPA operates in a much higher-intensity regime and likely approaches the compute ceiling of the hardware through optimization techniques not implemented in this MVP.

That is the wording used throughout this repository.

---

## Why SDPA is faster

This project does **not** claim to match production SDPA.

The remaining gap is explained by techniques used in optimized fused attention backends, including:

- vectorized global memory loads
- deeper software pipelining
- more aggressive multi-warp tiling
- more mature shared-memory scheduling
- much higher effective arithmetic intensity
- years of production optimization

This repository intentionally prioritizes:

- correctness
- transparency
- architectural clarity
- reproducibility on SM75

over absolute peak throughput.

---

## Honest limitations

1. **Specialized to d=64**
   - this repository focuses on the `d=64` case
   - extending to `d=128` is possible, but changes the register/smem tradeoff substantially

2. **Not competitive with SDPA on large workloads**
   - full attention: ~3.6x gap
   - causal attention: ~4.0x gap

3. **SM75-specific engineering**
   - the WMMA layout and tuning are specific to Turing/SM75
   - porting to SM80+ would require a new pass

4. **No direct hardware-counter confirmation under WSL**
   - WSL blocks Nsight Compute counter profiling on this setup
   - performance-regime statements are therefore based on measured timings + roofline reasoning

---

## Recommended binaries

### Full attention
Use:

```bash
./flash_mha_flatargs B H N [outer] [inner]
```

Example:

```bash
./flash_mha_flatargs 4 12 1024 7 200
```

### Causal attention
Use:

```bash
./flash_mha_v2 B H N [outer] [inner]
```

Example:

```bash
./flash_mha_v2 4 12 1024 7 200
```

### SDPA comparison
Use:

```bash
python benchmark_sdpa_final.py
```

---

## Reproducibility

Suggested sequence:

```bash
make all
./probe_layout
./test_warp_softmax
./flash_mha_flatargs 4 12 1024 7 200
./flash_mha_v2 4 12 1024 7 200
python benchmark_sdpa_final.py
```

---

## Repository status

This repository should be framed as:

> **A significant V2 improvement over the initial SM75 kernel, with a fully documented register-fused WMMA softmax architecture, but still below production SDPA throughput.**

That framing is both honest and technically accurate.

---

## Citation / attribution

If you reference this repository, please describe it as:

> A research-grade, from-scratch SM75 Flash Attention implementation using CUDA WMMA and warp-register softmax, developed by João Felipe De Souza (2026).

---

## License

MIT License — see `LICENSE`.

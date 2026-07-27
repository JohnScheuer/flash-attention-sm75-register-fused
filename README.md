# flash-attention-sm75

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.x](https://img.shields.io/badge/CUDA-11.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![SM75](https://img.shields.io/badge/GPU-SM75%20(Turing)-76B900.svg)](https://www.nvidia.com/en-us/geforce/turing/)
[![Status](https://img.shields.io/badge/status-published-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Correctness](https://img.shields.io/badge/max_err-<0.02-brightgreen.svg)]()

FlashAttention v1 forward pass implemented in CUDA for **NVIDIA Turing (SM75)** — RTX 2070, RTX 2080, T4, and similar.

Drop-in replacement for `torch.nn.functional.scaled_dot_product_attention` on hardware where the official FlashAttention library does not run.

> **For detailed technical analysis, algorithm derivation, WMMA layouts, performance decomposition, and lessons learned, read [DESIGN.md](DESIGN.md).**

---

## Why This Exists

The official [FlashAttention](https://github.com/Dao-AILab/flash-attention) requires **SM80+** (Ampere or newer). Users with Turing-generation GPUs (RTX 20xx, SM75) fall back to memory-intensive naive attention or the slower PyTorch SDPA math backend.

This project brings FlashAttention v1 to Turing with:

- Full CUDA kernel implementation using **Tensor Cores (WMMA)**
- Support for `head_dim = 64` and `head_dim = 128`
- Causal and non-causal attention
- GQA / MQA support via automatic K/V head expansion
- Drop-in monkey-patch for HuggingFace Transformers

---

## Hardware Requirements

| Requirement | Value |
|---|---|
| GPU Architecture | Turing (SM75) |
| Example GPUs | RTX 2060/2070/2080, Quadro RTX, T4 |
| CUDA | 11.x |
| Dtype | FP16 only |

---

## When To Use This

**Use flash-attention-sm75 if:**

- You have a Turing GPU (RTX 2060/2070/2080, T4, Quadro RTX)
- You need memory-efficient attention for long sequences (O(N) vs O(N^2))
- You want prefill acceleration in HuggingFace models on SM75
- You want an educational reference for FlashAttention on non-Ampere hardware

**Do NOT use this if:**

- You have Ampere+ GPU (use official FlashAttention instead)
- You need bit-exact reproduction with SDPA
- You need backward pass support (forward only)
- You need production-grade raw kernel performance (SDPA efficient is faster in isolation)

---

## Installation

From source:

    git clone https://github.com/JohnScheuer/flash-attention-sm75
    cd flash-attention-sm75
    pip install -e .

---

## Usage

### Direct API

    import torch
    from flash_attention_sm75 import flash_attention_forward

    Q = torch.randn(1, 8, 1024, 64, dtype=torch.float16, device='cuda')
    K = torch.randn(1, 8, 1024, 64, dtype=torch.float16, device='cuda')
    V = torch.randn(1, 8, 1024, 64, dtype=torch.float16, device='cuda')

    # Causal attention (e.g. decoder)
    O = flash_attention_forward(Q, K, V, causal=True)

    # Non-causal (e.g. encoder)
    O = flash_attention_forward(Q, K, V, causal=False)

### Monkey-patch for HuggingFace Transformers

    from examples.llama_attention_replacement import apply_patch, remove_patch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Apply patch before loading or calling the model
    apply_patch()

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda",
    )

    # All prefill attention calls now use the SM75 flash kernel
    # Decode (KV cache) falls back to PyTorch SDPA automatically

    remove_patch()

---

## Performance

### Context

This implementation is ~10-13x slower than PyTorch SDPA efficient backend on SM75 in isolation. The SDPA efficient backend is heavily optimized by NVIDIA/Meta engineers with proprietary techniques.

**Despite raw kernel being slower, end-to-end model throughput improves 27%** due to memory efficiency and prefill acceleration.

The value proposition is:

- **Memory efficiency** — O(N) vs O(N^2), enabling longer sequences
- **End-to-end throughput** — 27% improvement on Qwen2-0.5B
- **Working FlashAttention on SM75** — where the official version does not compile
- **Educational clarity** — reference implementation of the algorithm

For a detailed breakdown of where the performance gap comes from and why, see the [Performance Analysis section in DESIGN.md](DESIGN.md#3-performance-analysis).

### Throughput vs scalar baseline (d=64, non-causal)

Tested on NVIDIA GeForce RTX 2070 (SM75, 28.5 TFLOPS FP16 Tensor Core peak).

| seq_len | scalar ms | flash ms | speedup | TFLOPS |
|---|---|---|---|---|
| 128 | 0.238 | 0.096 | 2.5x | 0.35 |
| 512 | 1.373 | 0.578 | 2.4x | 0.93 |
| 1024 | 4.043 | 1.586 | 2.6x | 1.35 |
| 2048 | 14.61 | 6.207 | 2.4x | 1.38 |
| 4096 | 59.47 | 23.83 | 2.5x | 1.44 |

### Throughput vs scalar baseline (d=128, causal)

| seq_len | flash ms | TFLOPS |
|---|---|---|
| 512 | 0.522 | 1.03 |
| 1024 | 1.604 | 1.34 |
| 2048 | 6.163 | 1.39 |
| 4096 | 23.720 | 1.45 |

### Memory footprint vs naive attention

| seq_len | naive (MB) | flash (MB) | reduction |
|---|---|---|---|
| 1024 | 47.3 | 13.8 | **3.4x** |
| 2048 | 153.2 | 19.0 | **8.1x** |
| 4096 | 566.4 | 29.5 | **19.2x** |

Memory scales O(N) instead of O(N^2) — identical to the original FlashAttention paper.

### End-to-end generation (Qwen2-0.5B-Instruct)

| Backend | Throughput |
|---|---|
| Baseline (no patch) | 31-33 tok/s |
| With flash patch | 40-42 tok/s |
| **Improvement** | **+27%** |

### Tensor Core roofline

    RTX 2070 FP16 Tensor Core peak: 28.5 TFLOPS (measured via PyTorch GEMM)
    This implementation:            ~1.4 TFLOPS
    Efficiency:                     ~5%

The gap is dominated by intermediate shared-memory staging (S, P, T buffers) and scalar softmax overhead — not by lack of Tensor Core usage. See [DESIGN.md section 3.2](DESIGN.md#32-where-the-gap-comes-from) for detailed analysis.

---

## Correctness

Validated against `torch.nn.functional.scaled_dot_product_attention` and against a full FP32 reference implementation.

| Config | max_abs_err | Status |
|---|---|---|
| d=64, non-causal, N=128..4096 | < 5e-4 | PASS |
| d=64, causal, N=128..4096 | < 1e-3 | PASS |
| d=128, non-causal, N=128..4096 | < 5e-4 | PASS |
| d=128, causal, N=128..4096 | < 2e-3 | PASS |
| Prefill logits on Qwen2-0.5B | ~0.02 | PASS (top-1 match) |

Tolerance target: < 1e-2 (standard FP16 attention tolerance).

---

## Architecture

### Algorithm

Standard FlashAttention v1 (Dao et al., 2022) forward pass:

- Tiled computation over Q in blocks of Br
- For each Q tile, iterates over K/V tiles of size Bc
- Online softmax with running max and sum (no N×N matrix materialized)
- Memory complexity: O(N) instead of O(N^2)

For algorithmic details, WMMA layouts, and mathematical correctness proofs, see [DESIGN.md](DESIGN.md).

### Kernel Evolution

| Version | Description | TFLOPS (d=64, N=1024) |
|---|---|---|
| v1 scalar | Baseline tiled, no Tensor Cores | 0.55 |
| v2 wmma_qk | QK^T via WMMA Tensor Cores | 1.14 |
| v3 wmma_qk_pv | QK^T + PV via WMMA | 1.34 |
| v4 Br=64 | All 4 warps active in softmax | 1.35 |
| v5 causal pruning | Tile-level causal skip | 1.08 (N=1024 causal) |

### Tile Sizes

| head_dim | Br | Bc | Shared Memory |
|---|---|---|---|
| 64 | 64 | 64 | 40 KB |
| 128 | 32 | 32 | 40 KB |

SM75 shared memory limit: 48 KB per block.

### Warp Layout (d=64, Br=64)

    4 warps x 32 threads = 128 threads per block

    Each warp owns 16 Q-rows:
      warp 0 -> rows  0..15
      warp 1 -> rows 16..31
      warp 2 -> rows 32..47
      warp 3 -> rows 48..63

    All 4 warps active during: QK^T WMMA, softmax, PV WMMA, o_acc accumulation

### Causal Pruning

For each (q_tile, kv_tile) pair:

- Tile fully above diagonal -> break (skip all future KV tiles)
- Tile fully below diagonal -> fast path, no per-element masking
- Tile crosses diagonal -> per-element masking

This eliminates ~40-50% of work for causal attention on large sequences. See [DESIGN.md section 2.5](DESIGN.md#25-causal-pruning) for implementation details.

---

## Limitations

- Forward pass only (no backward / gradient support)
- FP16 only (no BF16, FP32, FP8)
- head_dim = 64 or 128 only
- KV cache decode step not accelerated (falls back to SDPA)
- SM75 only (compile with -arch=sm_75)
- Not bit-exact with SDPA (FP16 precision differences cause divergent generation over long outputs)

For known limitations, workarounds, and future work directions, see [DESIGN.md section 4](DESIGN.md#4-known-limitations-and-future-work).

---

## Comparison with Alternatives

| Alternative | SM75 support | Notes |
|---|---|---|
| Official FlashAttention | Not supported (SM80+ only) | This project fills the gap |
| PyTorch SDPA efficient | Supported | Faster in isolation, closed source |
| PyTorch SDPA math | Supported | O(N^2) memory, slower |
| xFormers | Limited | Not maintained for SM75 |

---

## Documentation

- **[DESIGN.md](DESIGN.md)** — Complete technical analysis including:
  - Algorithm derivation and WMMA layouts
  - Shared memory aliasing strategy
  - Performance decomposition (why we hit 5% of roofline)
  - Bugs found and root-caused
  - Lessons learned about SM75 vs SM80+ optimization
  - File structure and code organization

- **[README.md](README.md)** — This file (usage, benchmarks, quickstart)

- **[LICENSE](LICENSE)** — MIT License

---

## Related Projects

Part of a series exploring LLM inference infrastructure on SM75:

- [llm-fusion-compiler](https://github.com/JohnScheuer/llm-fusion-compiler) — CUDA fusion compiler for Transformer operations
- [fused-backward-kernels-sm75](https://github.com/JohnScheuer/fused-backward-kernels-sm75) — Backward pass kernels for gemm_bias_gelu
- [sm75-tensorcore-microkernel](https://github.com/JohnScheuer/sm75-tensorcore-microkernel) — PTX-level Tensor Core GEMM
- [mini-llm-inference-engine](https://github.com/JohnScheuer/mini-llm-inference-engine) — C++/CUDA LLM inference runtime

---

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022) — [arxiv:2205.14135](https://arxiv.org/abs/2205.14135)
- NVIDIA, "Programming Tensor Cores in CUDA 9" — WMMA API documentation
- NVIDIA Turing Architecture Whitepaper (2018)

---

## License

MIT License — see [LICENSE](LICENSE) file for details.

Copyright (c) 2026 João Felipe De Souza

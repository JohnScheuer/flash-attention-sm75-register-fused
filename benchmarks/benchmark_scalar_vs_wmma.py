"""
Comparação direta: flash_scalar vs flash_wmma_qk
Foco em d=64, shapes representativos.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math, csv
from pathlib import Path
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from reference.attention_ref import attention_reference
from reference.correctness import check_correctness

import flash_attn_sm75_cuda as _C

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

WARMUP = 5
ITERS  = 30

def bench(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record(); fn(); end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times)//2]


def sdpa_ctx(kind):
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        if kind == "math":
            return sdpa_kernel(backends=[SDPBackend.MATH])
        return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
    except Exception:
        if kind == "math":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False)
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=False, enable_mem_efficient=True)


def run(batch, heads, seq_len, causal):
    head_dim = 64
    scale = 1.0 / math.sqrt(head_dim)
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    if causal: flops //= 2

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    ref = attention_reference(Q, K, V, causal=causal, scale=scale)

    results = {}

    # sdpa_efficient
    with sdpa_ctx("efficient"):
        fn = lambda: F.scaled_dot_product_attention(Q,K,V,is_causal=causal,scale=scale)
        ms = bench(fn)
        out = fn()
    results["sdpa_efficient"] = {
        "ms": ms, "tflops": flops/(ms*1e9),
        "err": check_correctness(out, ref)["max_abs_err"]
    }

    # flash_scalar
    fn = lambda: _C.flash_attention_forward_scalar(Q, K, V, causal, scale)
    ms  = bench(fn)
    out = fn()
    results["flash_scalar"] = {
        "ms": ms, "tflops": flops/(ms*1e9),
        "err": check_correctness(out, ref)["max_abs_err"]
    }

    # flash_wmma_qk
    fn = lambda: _C.flash_attention_forward(Q, K, V, causal, scale)
    ms  = bench(fn)
    out = fn()
    results["flash_wmma_qk"] = {
        "ms": ms, "tflops": flops/(ms*1e9),
        "err": check_correctness(out, ref)["max_abs_err"]
    }

    return results


def main():
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    configs = [
        (1,  8,  128,  False),
        (1,  8,  256,  False),
        (1,  8,  512,  False),
        (1,  8,  1024, False),
        (1,  8,  2048, False),
        (1,  8,  4096, False),
        (1,  8,  128,  True),
        (1,  8,  512,  True),
        (1,  8,  1024, True),
        (1,  8,  2048, True),
        (1, 32,  512,  False),
        (1, 32, 1024,  False),
        (4,  8,  512,  False),
        (4,  8, 1024,  False),
    ]

    rows = []
    print(f"{'B':>2} {'H':>3} {'N':>5} {'csl':>4}  "
          f"{'sdpa_eff ms':>12} {'scalar ms':>10} {'wmma_qk ms':>11}  "
          f"{'scalar TFLOPS':>13} {'wmma TFLOPS':>11}  "
          f"{'speedup vs scalar':>17}")
    print("-" * 105)

    for b, h, n, c in configs:
        r = run(b, h, n, c)
        se  = r["sdpa_efficient"]
        sc  = r["flash_scalar"]
        wm  = r["flash_wmma_qk"]
        speedup = sc["ms"] / wm["ms"]

        print(f"{b:>2} {h:>3} {n:>5} {str(c):>4}  "
              f"{se['ms']:>12.3f} {sc['ms']:>10.3f} {wm['ms']:>11.3f}  "
              f"{sc['tflops']:>13.3f} {wm['tflops']:>11.3f}  "
              f"{speedup:>17.2f}x")

        rows.append({
            "batch": b, "heads": h, "seq_len": n, "causal": c,
            "sdpa_eff_ms": se["ms"],   "sdpa_eff_tflops": se["tflops"],
            "scalar_ms":   sc["ms"],   "scalar_tflops":   sc["tflops"],
            "wmma_qk_ms":  wm["ms"],   "wmma_qk_tflops":  wm["tflops"],
            "speedup_wmma_vs_scalar": speedup,
            "wmma_err": wm["err"],
        })

    csv_path = RESULTS_DIR / "benchmark_scalar_vs_wmma.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()

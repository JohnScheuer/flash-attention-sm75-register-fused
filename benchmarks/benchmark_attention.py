"""
Intelligent benchmark harness for attention backends.

Backends:
- sdpa_math
- sdpa_efficient
- naive_cuda
- flash_mvp  (our FlashAttention forward MVP, d=64 only)

Outputs:
- results/benchmark_phase1.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import csv
import math
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from reference.attention_ref import attention_reference
from reference.correctness import check_correctness

try:
    import flash_attn_sm75_cuda as _C
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda"

SEQ_LENS   = [128, 256, 512, 1024, 2048, 4096, 8192]
HEAD_DIMS  = [64, 128]
NUM_HEADS  = [8, 16, 32]
BATCHES    = [1, 4]
CAUSALS    = [False, True]

MAX_REF_ATTENTION_MB      = 768
MAX_NAIVE_WORKSPACE_MB    = 2500
MAX_SDPA_MATH_EST_MB      = 5000


def sdpa_ctx(kind: str):
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        if kind == "math":
            return sdpa_kernel(backends=[SDPBackend.MATH])
        elif kind == "efficient":
            return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
        else:
            return nullcontext()
    except Exception:
        if kind == "math":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            )
        elif kind == "efficient":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            )
        else:
            return nullcontext()


def attn_elements(batch, heads, seq_len):
    return batch * heads * seq_len * seq_len


def attention_flops(batch, heads, seq_len, head_dim, causal=False):
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    if causal:
        flops = flops // 2
    return flops


def estimate_ref_attention_mb(batch, heads, seq_len):
    return attn_elements(batch, heads, seq_len) * 4 / 1e6


def estimate_naive_workspace_mb(batch, heads, seq_len):
    return attn_elements(batch, heads, seq_len) * 4 / 1e6


def estimate_sdpa_math_mb(batch, heads, seq_len):
    return attn_elements(batch, heads, seq_len) * 8 / 1e6


def benchmark_policy(backend, batch, heads, seq_len, head_dim):
    score = batch * heads * seq_len * seq_len * head_dim
    naive_mb = estimate_naive_workspace_mb(batch, heads, seq_len)
    math_mb = estimate_sdpa_math_mb(batch, heads, seq_len)

    if backend == "sdpa_efficient":
        if score <= 3e8:
            return {"mode": "timed", "warmup": 5, "iters": 30, "reason": "small"}
        elif score <= 2e9:
            return {"mode": "timed", "warmup": 3, "iters": 10, "reason": "medium"}
        else:
            return {"mode": "timed", "warmup": 2, "iters": 5, "reason": "large"}

    if backend == "sdpa_math":
        if math_mb > MAX_SDPA_MATH_EST_MB:
            return {"mode": "skip", "warmup": 0, "iters": 0, "reason": "expected_oom_math"}
        if score <= 3e8:
            return {"mode": "timed", "warmup": 5, "iters": 20, "reason": "small"}
        elif score <= 2e9:
            return {"mode": "timed", "warmup": 2, "iters": 6, "reason": "medium"}
        else:
            return {"mode": "timed", "warmup": 1, "iters": 2, "reason": "large"}

    if backend == "naive_cuda":
        if not HAS_CUDA_EXT:
            return {"mode": "skip", "warmup": 0, "iters": 0, "reason": "ext_missing"}
        if naive_mb > MAX_NAIVE_WORKSPACE_MB:
            return {"mode": "skip", "warmup": 0, "iters": 0, "reason": "workspace_too_large"}
        if seq_len <= 512:
            return {"mode": "timed", "warmup": 3, "iters": 10, "reason": "timed_small"}
        if seq_len <= 1024 and batch * heads <= 16:
            return {"mode": "timed", "warmup": 1, "iters": 3, "reason": "timed_medium"}
        return {"mode": "mem_only", "warmup": 0, "iters": 0, "reason": "mem_only_large"}

    if backend == "flash_mvp":
        if not HAS_CUDA_EXT:
            return {"mode": "skip", "warmup": 0, "iters": 0, "reason": "ext_missing"}
        if head_dim != 64:
            return {"mode": "skip", "warmup": 0, "iters": 0, "reason": "mvp_d64_only"}
        if score <= 3e8:
            return {"mode": "timed", "warmup": 5, "iters": 20, "reason": "small"}
        elif score <= 2e9:
            return {"mode": "timed", "warmup": 3, "iters": 8, "reason": "medium"}
        else:
            return {"mode": "timed", "warmup": 2, "iters": 4, "reason": "large"}

    raise ValueError(f"Unknown backend: {backend}")


def should_compute_reference(batch, heads, seq_len):
    return estimate_ref_attention_mb(batch, heads, seq_len) <= MAX_REF_ATTENTION_MB


def benchmark_timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def measure_peak_memory(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(DEVICE)
    return out, peak


def safe_backend_run(name, fn, policy, ref, flops):
    row = {
        "backend": name,
        "status": "ok",
        "timing_mode": policy["mode"],
        "policy_reason": policy["reason"],
        "median_ms": None,
        "tflops": None,
        "peak_memory_mb": None,
        "max_abs_err": None,
        "correct": None,
    }

    if policy["mode"] == "skip":
        row["status"] = "skipped"
        return row

    try:
        if policy["mode"] == "timed":
            timing = benchmark_timed(fn, policy["warmup"], policy["iters"])
            out, peak = measure_peak_memory(fn)
            row["median_ms"] = float(timing["median_ms"])
            row["tflops"] = flops / (row["median_ms"] * 1e9)
        elif policy["mode"] == "mem_only":
            out, peak = measure_peak_memory(fn)
        else:
            raise ValueError(f"Unknown mode: {policy['mode']}")

        row["peak_memory_mb"] = peak / 1e6

        if ref is not None:
            corr = check_correctness(out, ref)
            row["max_abs_err"] = corr["max_abs_err"]
            row["correct"] = corr["allclose"]

        return row

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            torch.cuda.empty_cache()
            row["status"] = "oom"
            return row
        raise


def make_inputs(batch, heads, seq_len, head_dim):
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE)
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE)
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=DEVICE)
    return Q, K, V


def run_one_config(batch, heads, seq_len, head_dim, causal):
    Q, K, V = make_inputs(batch, heads, seq_len, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    flops = attention_flops(batch, heads, seq_len, head_dim, causal)

    ref = None
    ref_mode = "skipped_large"
    if should_compute_reference(batch, heads, seq_len):
        ref = attention_reference(Q, K, V, causal=causal, scale=scale)
        ref_mode = "full_reference"

    rows = []

    def fn_math():
        with sdpa_ctx("math"):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)

    def fn_eff():
        with sdpa_ctx("efficient"):
            return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)

    backends = [
        ("sdpa_math", fn_math),
        ("sdpa_efficient", fn_eff),
    ]

    if HAS_CUDA_EXT:
        def fn_naive():
            return _C.naive_attention_forward(Q, K, V, causal, scale)

        def fn_flash():
            return _C.flash_attention_forward(Q, K, V, causal, scale)

        backends.append(("naive_cuda", fn_naive))
        backends.append(("flash_mvp", fn_flash))

    for name, fn in backends:
        policy = benchmark_policy(name, batch, heads, seq_len, head_dim)
        row = safe_backend_run(name, fn, policy, ref, flops)
        row.update({
            "batch": batch,
            "heads": heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "causal": causal,
            "reference_mode": ref_mode,
        })
        rows.append(row)

    del Q, K, V, ref
    torch.cuda.empty_cache()

    return rows


def build_configs():
    configs = []
    for causal in CAUSALS:
        for head_dim in HEAD_DIMS:
            for heads in NUM_HEADS:
                for batch in BATCHES:
                    for seq_len in SEQ_LENS:
                        configs.append((batch, heads, seq_len, head_dim, causal))
    return configs


def print_row(row):
    status = row["status"]
    backend = row["backend"]

    if status == "skipped":
        print(f"  {backend:18s}  SKIP      reason={row['policy_reason']}")
        return

    if status == "oom":
        print(f"  {backend:18s}  OOM       mode={row['timing_mode']}")
        return

    ms = row["median_ms"]
    tf = row["tflops"]
    mb = row["peak_memory_mb"]
    err = row["max_abs_err"]
    corr = row["correct"]

    if row["timing_mode"] == "timed":
        print(
            f"  {backend:18s}  {ms:8.3f} ms  "
            f"{tf:7.3f} TFLOPS  "
            f"{mb:8.1f} MB  "
            f"err={err if err is not None else 'n/a'}  "
            f"{'PASS' if corr else ('N/A' if corr is None else 'FAIL')}"
        )
    else:
        print(
            f"  {backend:18s}  mem-only      "
            f"{mb:8.1f} MB  "
            f"err={err if err is not None else 'n/a'}  "
            f"{'PASS' if corr else ('N/A' if corr is None else 'FAIL')}"
        )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA extension available: {HAS_CUDA_EXT}")

    configs = build_configs()
    all_rows = []

    total = len(configs)
    for idx, (batch, heads, seq_len, head_dim, causal) in enumerate(configs, start=1):
        print(f"\n[{idx}/{total}] B={batch} H={heads} N={seq_len} d={head_dim} causal={causal}")
        try:
            rows = run_one_config(batch, heads, seq_len, head_dim, causal)
            for row in rows:
                print_row(row)
            all_rows.extend(rows)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg:
                print("  CONFIG OOM -> skipping config")
                torch.cuda.empty_cache()
                continue
            raise

    if all_rows:
        csv_path = RESULTS_DIR / "benchmark_phase2_mvp.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved: {csv_path}")

    print("\n=== Summary counts by backend/status ===")
    summary = {}
    for row in all_rows:
        key = (row["backend"], row["status"], row["timing_mode"])
        summary[key] = summary.get(key, 0) + 1

    for key, count in sorted(summary.items()):
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from reference.attention_ref import attention_reference
from reference.correctness import check_correctness
import flash_attn_sm75_cuda as _C


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    times = []

    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def sdpa_eff(Q, K, V, causal, scale):
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx = sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
    except Exception:
        ctx = torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=False,
            enable_mem_efficient=True
        )

    with ctx:
        return F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)


def run(batch, heads, seq_len, head_dim, causal):
    scale = 1.0 / math.sqrt(head_dim)
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    if causal:
        flops //= 2

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

    ref = attention_reference(Q, K, V, causal=causal, scale=scale)

    def measure(fn):
        ms = bench(fn)
        out = fn()
        err = check_correctness(out, ref)["max_abs_err"]
        return ms, flops / (ms * 1e9), err

    sdpa_ms, _, _     = measure(lambda: sdpa_eff(Q, K, V, causal, scale))
    flash_ms, tf, err = measure(lambda: _C.flash_attention_forward(Q, K, V, causal, scale))

    return {
        "batch": batch,
        "heads": heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "causal": causal,
        "sdpa_eff_ms": sdpa_ms,
        "flash_ms": flash_ms,
        "flash_tflops": tf,
        "flash_err": err,
        "slower_than_sdpa": flash_ms / sdpa_ms,
    }


configs = []
for causal in [False, True]:
    for hd in [64, 128]:
        for n in [128, 256, 512, 1024, 2048, 4096]:
            configs.append((1, 8, n, hd, causal))
        for n in [512, 1024]:
            configs.append((1, 32, n, hd, causal))
            configs.append((4,  8, n, hd, causal))


def main():
    print(f"GPU: {torch.cuda.get_device_name()}\n")

    header = (
        f"{'B':>2}{'H':>4}{'N':>6}{'d':>5}{'csl':>5}  "
        f"{'sdpa_eff ms':>12}{'flash ms':>10}{'TFLOPS':>10}"
        f"{'err':>12}{'vs_sdpa':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for cfg in configs:
        try:
            r = run(*cfg)
            print(
                f"{r['batch']:>2}{r['heads']:>4}{r['seq_len']:>6}{r['head_dim']:>5}{str(r['causal']):>5}  "
                f"{r['sdpa_eff_ms']:>12.3f}{r['flash_ms']:>10.3f}{r['flash_tflops']:>10.3f}"
                f"{r['flash_err']:>12.6f}{r['slower_than_sdpa']:>9.1f}x"
            )
            rows.append(r)
        except Exception as ex:
            print(f"SKIP {cfg}: {ex}")

    csv_path = RESULTS_DIR / "benchmark_d64_d128.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {csv_path}")

    print("\n=== Summary by head_dim (non-causal only) ===")
    for hd in [64, 128]:
        sub = [r for r in rows if r["head_dim"] == hd and not r["causal"]]
        if not sub:
            continue
        avg_tf = sum(r["flash_tflops"] for r in sub) / len(sub)
        avg_vs = sum(r["slower_than_sdpa"] for r in sub) / len(sub)
        print(f"d={hd}: avg {avg_tf:.3f} TFLOPS, avg {avg_vs:.1f}x slower than sdpa_eff")


if __name__ == "__main__":
    main()

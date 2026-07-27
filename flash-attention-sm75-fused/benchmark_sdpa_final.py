import os
import re
import subprocess
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda")
props = torch.cuda.get_device_properties(device)

print(f"Device : {props.name}")
print(f"SM     : {props.major}{props.minor}")
print(f"Memory : {props.total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Backend probing
# ─────────────────────────────────────────────────────────────────────────────
def backend_available(backend: SDPBackend, causal=False):
    q = torch.randn(1, 1, 16, 64, dtype=torch.float16, device=device)
    try:
        with sdpa_kernel(backends=[backend]):
            F.scaled_dot_product_attention(q, q, q, is_causal=causal)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False

flash_ok   = backend_available(SDPBackend.FLASH_ATTENTION, causal=False)
memeff_ok  = backend_available(SDPBackend.EFFICIENT_ATTENTION, causal=False)
math_ok    = backend_available(SDPBackend.MATH, causal=False)

print("SDPA backend probe:")
print(f"  Flash attention backend : {'YES' if flash_ok else 'NO (SM75 not supported)'}")
print(f"  Mem-efficient backend   : {'YES' if memeff_ok else 'NO'}")
print(f"  Math backend            : {'YES' if math_ok else 'NO'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Configs
# ─────────────────────────────────────────────────────────────────────────────
configs = [
    (4, 12, 1024, 64, "peak-full"),
    (4,  8, 2048, 64, "peak-large"),
    (8, 12,  512, 64, "peak-wide"),
    (1, 16, 2048, 64, "peak-heads"),
    (1,  8,  256, 64, "medium"),
    (1,  1,  512, 64, "single-head"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def median_ms(run_fn, warmup=20, reps=200):
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    vals = []
    for _ in range(reps):
        t0.record()
        run_fn()
        t1.record()
        torch.cuda.synchronize()
        vals.append(t0.elapsed_time(t1))

    vals.sort()
    return vals[len(vals) // 2]

def measure_sdpa(B, H, N, d, causal=False, backend="default", warmup=20, reps=200):
    Q = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
    K = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
    V = torch.randn(B, H, N, d, dtype=torch.float16, device=device)

    if backend == "default":
        def run():
            return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        return median_ms(run, warmup=warmup, reps=reps)

    elif backend == "mem_eff":
        if not memeff_ok:
            return None
        def run():
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        return median_ms(run, warmup=warmup, reps=reps)

    elif backend == "math":
        if not math_ok:
            return None
        def run():
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        return median_ms(run, warmup=warmup, reps=reps)

    elif backend == "flash":
        if not flash_ok:
            return None
        def run():
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        return median_ms(run, warmup=warmup, reps=reps)

    else:
        raise ValueError(f"Unknown backend: {backend}")

def parse_flatargs_output(text: str):
    # Expected block:
    # === Benchmark ...
    #   métrica     min        median     max
    #   ms          X          Y          Z
    m = re.search(r'\bms\b\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text)
    if not m:
        return None
    return float(m.group(2))

def parse_v2_output(text: str):
    # Expected block:
    # FULL       min  med  max ...
    # CAUSAL     min  med  max ...
    full = None
    causal = None

    m_full = re.search(r'^FULL\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text, re.MULTILINE)
    if m_full:
        full = float(m_full.group(2))

    m_causal = re.search(r'^CAUSAL\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text, re.MULTILINE)
    if m_causal:
        causal = float(m_causal.group(2))

    return full, causal

def run_binary(cmd):
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=180
        ).decode()
        return out
    except subprocess.CalledProcessError as e:
        return e.output.decode() if e.output else ""
    except Exception:
        return ""

def measure_flatargs(B, H, N, outer=7, inner=200):
    if not os.path.isfile("./flash_mha_flatargs"):
        return None
    out = run_binary(["./flash_mha_flatargs", str(B), str(H), str(N), str(outer), str(inner)])
    return parse_flatargs_output(out)

def measure_v2(B, H, N, outer=7, inner=200):
    if not os.path.isfile("./flash_mha_v2"):
        return None, None
    out = run_binary(["./flash_mha_v2", str(B), str(H), str(N), str(outer), str(inner)])
    return parse_v2_output(out)

def tflops(B, H, N, d, ms, causal=False):
    if ms is None or ms == 0:
        return None
    flops = B * H * 4.0 * N * N * d
    if causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12

def fmt_ms(x):
    return f"{x:.4f}" if x is not None else "N/A"

def fmt_tf(x):
    return f"{x:.3f}" if x is not None else "N/A"

def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b

def verdict_str(r):
    if r is None:
        return "N/A"
    if r < 1.0:
        return f"{r:.2f}x ← faster"
    if r < 2.0:
        return f"{r:.2f}x ← competitive"
    if r < 4.0:
        return f"{r:.2f}x ← significant gap"
    return f"{r:.2f}x ← research-grade"

# ─────────────────────────────────────────────────────────────────────────────
# Gather measurements
# ─────────────────────────────────────────────────────────────────────────────
results = {}

print("=" * 100)
print("MEASURING SDPA + CUSTOM KERNELS")
print("=" * 100)

for B, H, N, d, label in configs:
    print(f"Running: B={B} H={H} N={N} d={d} ({label})")

    sdpa_default_full   = measure_sdpa(B, H, N, d, causal=False, backend="default")
    sdpa_default_causal = measure_sdpa(B, H, N, d, causal=True,  backend="default")

    sdpa_memeff_full   = measure_sdpa(B, H, N, d, causal=False, backend="mem_eff") if memeff_ok else None
    sdpa_memeff_causal = measure_sdpa(B, H, N, d, causal=True,  backend="mem_eff") if memeff_ok else None

    flat_ms = measure_flatargs(B, H, N, outer=7, inner=200)
    v2_full_ms, v2_causal_ms = measure_v2(B, H, N, outer=7, inner=200)

    results[(B, H, N, d)] = {
        "label": label,
        "sdpa_default_full": sdpa_default_full,
        "sdpa_default_causal": sdpa_default_causal,
        "sdpa_memeff_full": sdpa_memeff_full,
        "sdpa_memeff_causal": sdpa_memeff_causal,
        "flat_ms": flat_ms,
        "v2_full_ms": v2_full_ms,
        "v2_causal_ms": v2_causal_ms,
    }

    print(f"  SDPA full default={fmt_ms(sdpa_default_full)} ms"
          + (f"  mem_eff={fmt_ms(sdpa_memeff_full)} ms" if sdpa_memeff_full is not None else ""))
    print(f"  SDPA causal default={fmt_ms(sdpa_default_causal)} ms"
          + (f"  mem_eff={fmt_ms(sdpa_memeff_causal)} ms" if sdpa_memeff_causal is not None else ""))
    print(f"  flatargs={fmt_ms(flat_ms)} ms  v2_full={fmt_ms(v2_full_ms)} ms  v2_causal={fmt_ms(v2_causal_ms)} ms")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────
sep = "=" * 120
print(sep)
print("TABLE 1 — FULL ATTENTION")
print(sep)
print(f"{'Config':<28} | {'flatargs':>10} | {'SDPA default':>12} | {'SDPA mem_eff':>12} | {'ratio vs best SDPA':>20}")
print("-" * 120)

full_ratios = []
for B, H, N, d, label in configs:
    r = results[(B, H, N, d)]
    best_sdpa = r["sdpa_default_full"]
    if r["sdpa_memeff_full"] is not None:
        best_sdpa = min(best_sdpa, r["sdpa_memeff_full"])
    rr = ratio(r["flat_ms"], best_sdpa)
    if rr is not None:
        full_ratios.append(rr)

    print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
          f"{fmt_ms(r['flat_ms']):>10} | "
          f"{fmt_ms(r['sdpa_default_full']):>12} | "
          f"{fmt_ms(r['sdpa_memeff_full']):>12} | "
          f"{verdict_str(rr):>20}")

print()
print(sep)
print("TABLE 2 — CAUSAL ATTENTION")
print(sep)
print(f"{'Config':<28} | {'v2-causal':>10} | {'SDPA default':>12} | {'SDPA mem_eff':>12} | {'ratio vs best SDPA':>20}")
print("-" * 120)

causal_ratios = []
for B, H, N, d, label in configs:
    r = results[(B, H, N, d)]
    best_sdpa = r["sdpa_default_causal"]
    if r["sdpa_memeff_causal"] is not None:
        best_sdpa = min(best_sdpa, r["sdpa_memeff_causal"])
    rr = ratio(r["v2_causal_ms"], best_sdpa)
    if rr is not None:
        causal_ratios.append(rr)

    print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
          f"{fmt_ms(r['v2_causal_ms']):>10} | "
          f"{fmt_ms(r['sdpa_default_causal']):>12} | "
          f"{fmt_ms(r['sdpa_memeff_causal']):>12} | "
          f"{verdict_str(rr):>20}")

print()
print(sep)
print("TABLE 3 — INTERNAL CAUSAL SPEEDUP")
print(sep)
print(f"{'Config':<28} | {'flatargs full':>12} | {'v2 causal':>10} | {'speedup':>10}")
print("-" * 120)

for B, H, N, d, label in configs:
    r = results[(B, H, N, d)]
    sp = ratio(r["flat_ms"], r["v2_causal_ms"])
    sp_str = f"{sp:.3f}x" if sp is not None else "N/A"
    print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
          f"{fmt_ms(r['flat_ms']):>12} | "
          f"{fmt_ms(r['v2_causal_ms']):>10} | "
          f"{sp_str:>10}")

print()
print(sep)
print("TABLE 4 — TFLOPS")
print(sep)
print(f"{'Config':<28} | {'flatargs TF':>11} | {'v2 causal TF':>12} | {'SDPA full TF':>12} | {'SDPA causal TF':>14}")
print("-" * 120)

for B, H, N, d, label in configs:
    r = results[(B, H, N, d)]

    best_sdpa_full = r["sdpa_default_full"]
    if r["sdpa_memeff_full"] is not None:
        best_sdpa_full = min(best_sdpa_full, r["sdpa_memeff_full"])

    best_sdpa_causal = r["sdpa_default_causal"]
    if r["sdpa_memeff_causal"] is not None:
        best_sdpa_causal = min(best_sdpa_causal, r["sdpa_memeff_causal"])

    print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
          f"{fmt_tf(tflops(B,H,N,d,r['flat_ms'], False)):>11} | "
          f"{fmt_tf(tflops(B,H,N,d,r['v2_causal_ms'], True)):>12} | "
          f"{fmt_tf(tflops(B,H,N,d,best_sdpa_full, False)):>12} | "
          f"{fmt_tf(tflops(B,H,N,d,best_sdpa_causal, True)):>14}")

# ─────────────────────────────────────────────────────────────────────────────
# Final verdict
# ─────────────────────────────────────────────────────────────────────────────
print()
print(sep)
print("VERDICT SUMMARY")
print(sep)

if full_ratios:
    full_ratios_sorted = sorted(full_ratios)
    med_full = full_ratios_sorted[len(full_ratios_sorted)//2]
    print(f"flatargs vs best SDPA-full   — median ratio: {med_full:.2f}x")
    if med_full < 2.0:
        print("  → Competitive with PyTorch SDPA. Publish V2 with strong highlight.")
    elif med_full < 4.0:
        print("  → Significant improvement over V1. Publish V2 and document gap honestly.")
    else:
        print("  → Research-grade MVP relative to SDPA. Publish with careful framing.")

if causal_ratios:
    causal_ratios_sorted = sorted(causal_ratios)
    med_causal = causal_ratios_sorted[len(causal_ratios_sorted)//2]
    print(f"v2-causal vs best SDPA-causal — median ratio: {med_causal:.2f}x")
    if med_causal < 2.0:
        print("  → Competitive causal kernel.")
    elif med_causal < 4.0:
        print("  → Significant causal implementation, but still behind SDPA.")
    else:
        print("  → Research-grade causal implementation; gap should be documented.")

print()
print("Context:")
print("  • PyTorch SDPA on SM75 does not use FlashAttention (FlashAttention requires SM80+).")
print("  • The strongest fair baseline on SM75 is 'default SDPA' and, when available,")
print("    the forced mem-efficient backend.")
print("  • Your contribution is still strong: register-fused WMMA softmax, persistent O_acc,")
print("    QK reuse, causal masking, and reproducible SM75 kernels from first principles.")
print(sep)

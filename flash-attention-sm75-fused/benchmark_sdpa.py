"""
benchmark_sdpa.py
Compares PyTorch scaled_dot_product_attention (SDPA) against
flash_mha_flatargs and flash_mha_v2 (causal) on RTX 2070 SM75.

Run:
    python benchmark_sdpa.py

Requirements:
    pip install torch  (>= 2.0 for SDPA)
    flash_mha_flatargs and flash_mha_v2 binaries must be in the same directory.
"""

import torch
import torch.nn.functional as F
import subprocess
import re
import time
import sys
import os

# ── Device info ───────────────────────────────────────────────────────────────
device = torch.device("cuda")
props  = torch.cuda.get_device_properties(device)
print(f"Device : {props.name}")
print(f"SM     : {props.major}{props.minor}")
print(f"Memory : {props.total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print()

# ── SDPA backend info ─────────────────────────────────────────────────────────
# PyTorch >= 2.0 exposes which flash-attention backend is selected.
# On SM75 it typically falls back to the "math" (unfused) backend because
# the official FlashAttention CUDA extension requires SM80+.
try:
    with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True):
        pass
    has_sdp_kernel_ctx = True
except Exception:
    has_sdp_kernel_ctx = False

print("SDPA backend probe:")
q_probe = torch.randn(1, 1, 16, 64, dtype=torch.float16, device=device)
try:
    with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                         enable_math=False,
                                         enable_mem_efficient=False):
        F.scaled_dot_product_attention(q_probe, q_probe, q_probe)
    flash_available = True
except RuntimeError:
    flash_available = False

try:
    with torch.backends.cuda.sdp_kernel(enable_flash=False,
                                         enable_math=False,
                                         enable_mem_efficient=True):
        F.scaled_dot_product_attention(q_probe, q_probe, q_probe)
    mem_eff_available = True
except RuntimeError:
    mem_eff_available = False

print(f"  Flash attention backend : {'YES' if flash_available else 'NO (SM75 not supported)'}")
print(f"  Mem-efficient backend   : {'YES' if mem_eff_available else 'NO'}")
print(f"  Math (unfused) backend  : YES (always available)")
print()

# ── Helper: measure SDPA ──────────────────────────────────────────────────────
def measure_sdpa(B, H, N, d, causal=False, backend="default",
                 warmup=20, reps=200):
    """
    Returns median latency in ms over `reps` iterations.
    """
    Q = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
    K = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
    V = torch.randn(B, H, N, d, dtype=torch.float16, device=device)

    def run():
        if backend == "flash" and flash_available:
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=True,
                                                  enable_math=False,
                                                  enable_mem_efficient=False)
        elif backend == "mem_eff" and mem_eff_available:
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False,
                                                  enable_math=False,
                                                  enable_mem_efficient=True)
        else:
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=flash_available,
                                                  enable_math=True,
                                                  enable_mem_efficient=mem_eff_available)
        with ctx:
            return F.scaled_dot_product_attention(Q, K, V,
                                                   is_causal=causal)

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Timed runs using CUDA events for accuracy
    samples = []
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        t0.record()
        run()
        t1.record()
        torch.cuda.synchronize()
        samples.append(t0.elapsed_time(t1))

    samples.sort()
    return {
        "min":    samples[0],
        "median": samples[len(samples) // 2],
        "max":    samples[-1],
    }

# ── Helper: read kernel median from binary ────────────────────────────────────
def measure_kernel(binary, B, H, N, outer=7, inner=200):
    """
    Calls the compiled CUDA binary and parses the median latency.
    Returns median ms or None on failure.
    """
    if not os.path.isfile(f"./{binary}"):
        return None
    cmd = [f"./{binary}", str(B), str(H), str(N), str(outer), str(inner)]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       timeout=120).decode()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return None

    # Parse "ms   X.XXXX  X.XXXX  X.XXXX"
    # The benchmark prints: min / median / max on the "ms" row
    match = re.search(
        r'ms\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        out)
    if match:
        return float(match.group(2))   # median
    return None

# ── Configs ───────────────────────────────────────────────────────────────────
# Format: (B, H, N, d, label)
configs = [
    (4, 12, 1024, 64, "peak-full"),
    (4,  8, 2048, 64, "peak-large"),
    (8, 12,  512, 64, "peak-wide"),
    (1, 16, 2048, 64, "peak-heads"),
    (1,  8,  256, 64, "medium"),
    (1,  1,  512, 64, "single-head"),
]

# ── SDPA backends to test ─────────────────────────────────────────────────────
backends = ["default"]
if flash_available:
    backends.append("flash")
if mem_eff_available:
    backends.append("mem_eff")

# ── Run SDPA measurements ─────────────────────────────────────────────────────
print("=" * 100)
print("SDPA FULL ATTENTION")
print("=" * 100)

sdpa_results = {}  # (B,H,N,d) -> {"default": ms, ...}

header = f"{'Config':<28} | "
for b in backends:
    header += f"{'SDPA-'+b+' med(ms)':<16} "
print(header)
print("-" * 100)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    sdpa_results[key] = {}
    row = f"B={B} H={H} N={N} d={d} ({label:<10}) | "
    for b in backends:
        r = measure_sdpa(B, H, N, d, causal=False, backend=b,
                          warmup=20, reps=200)
        sdpa_results[key][b] = r
        row += f"{r['median']:>10.4f} ms     "
    print(row)

# ── SDPA causal ───────────────────────────────────────────────────────────────
print()
print("=" * 100)
print("SDPA CAUSAL ATTENTION")
print("=" * 100)

sdpa_causal = {}

header = f"{'Config':<28} | "
for b in backends:
    header += f"{'SDPA-'+b+' med(ms)':<16} "
print(header)
print("-" * 100)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    sdpa_causal[key] = {}
    row = f"B={B} H={H} N={N} d={d} ({label:<10}) | "
    for b in backends:
        r = measure_sdpa(B, H, N, d, causal=True, backend=b,
                          warmup=20, reps=200)
        sdpa_causal[key][b] = r
        row += f"{r['median']:>10.4f} ms     "
    print(row)

# ── Kernel measurements ───────────────────────────────────────────────────────
print()
print("=" * 100)
print("CUSTOM KERNEL MEASUREMENTS")
print("=" * 100)

kernel_full   = {}  # (B,H,N,d) -> median ms
kernel_causal = {}

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    mf = measure_kernel("flash_mha_flatargs", B, H, N, outer=7, inner=200)
    mc = measure_kernel("flash_mha_v2",       B, H, N, outer=7, inner=200)
    kernel_full[key]   = mf
    kernel_causal[key] = mc
    print(f"B={B} H={H} N={N} d={d}: "
          f"flatargs={mf:.4f} ms  " if mf else
          f"B={B} H={H} N={N} d={d}: flatargs=N/A  ", end="")
    print(f"v2-causal={mc:.4f} ms" if mc else "v2-causal=N/A")

# ── Head-to-head comparison ───────────────────────────────────────────────────
SEP = "=" * 110

def ratio_str(kernel_ms, sdpa_ms):
    if kernel_ms is None or sdpa_ms is None or sdpa_ms == 0:
        return "  N/A  "
    r = kernel_ms / sdpa_ms
    flag = ""
    if r <= 1.0:
        flag = " ← FASTER"
    elif r <= 2.0:
        flag = " ← competitive"
    elif r <= 4.0:
        flag = " ← significant gap"
    else:
        flag = " ← research-grade"
    return f"{r:5.2f}x{flag}"

def tflops(B, H, N, d, ms, causal=False):
    if ms is None or ms == 0:
        return 0.0
    flops = B * H * 4.0 * N * N * d
    if causal:
        flops *= 0.5
    return flops / (ms * 1e-3) / 1e12

print()
print(SEP)
print("HEAD-TO-HEAD: flash_mha_flatargs vs SDPA (FULL ATTENTION)")
print(SEP)
print(f"{'Config':<28} | "
      f"{'flatargs med':<14} {'flatargs TF':<13} | "
      f"{'SDPA-default':<14} {'SDPA TF':<12} | "
      f"{'ratio (kernel/sdpa)':<28}")
print("-" * 110)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    km  = kernel_full.get(key)
    sm  = sdpa_results[key]["default"]["median"]
    ktf = tflops(B, H, N, d, km,  causal=False)
    stf = tflops(B, H, N, d, sm,  causal=False)
    km_str  = f"{km:.4f} ms" if km else "   N/A   "
    ktf_str = f"{ktf:.3f} TF" if km else "  N/A   "
    print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
          f"{km_str:<14} {ktf_str:<13} | "
          f"{sm:<10.4f} ms   {stf:<8.3f} TF  | "
          f"{ratio_str(km, sm)}")

print()
print(SEP)
print("HEAD-TO-HEAD: flash_mha_v2 (CAUSAL) vs SDPA (CAUSAL)")
print(SEP)
print(f"{'Config':<28} | "
      f"{'v2-causal med':<14} {'v2-causal TF':<13} | "
      f"{'SDPA-default':<14} {'SDPA TF':<12} | "
      f"{'ratio (kernel/sdpa)':<28}")
print("-" * 110)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    km  = kernel_causal.get(key)
    sm  = sdpa_causal[key]["default"]["median"]
    ktf = tflops(B, H, N, d, km, causal=True)
    stf = tflops(B, H, N, d, sm, causal=True)
    km_str  = f"{km:.4f} ms" if km else "   N/A   "
    ktf_str = f"{ktf:.3f} TF" if km else "  N/A   "
    print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
          f"{km_str:<14} {ktf_str:<13} | "
          f"{sm:<10.4f} ms   {stf:<8.3f} TF  | "
          f"{ratio_str(km, sm)}")

# ── Internal causal speedup ───────────────────────────────────────────────────
print()
print(SEP)
print("CAUSAL SPEEDUP: v2-causal vs flatargs-full")
print("(How much faster is our causal kernel vs our own full kernel)")
print(SEP)
print(f"{'Config':<28} | "
      f"{'flatargs-full':<14} {'v2-causal':<12} | "
      f"{'latency speedup':<18} {'TFLOPS causal'}")
print("-" * 90)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    kf = kernel_full.get(key)
    kc = kernel_causal.get(key)
    if kf and kc:
        spd = kf / kc
        ktf = tflops(B, H, N, d, kc, causal=True)
        print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
              f"{kf:.4f} ms       {kc:.4f} ms   | "
              f"{spd:6.3f}×              {ktf:.3f} TF")
    else:
        print(f"B={B} H={H} N={N} d={d} ({label:<10}) | N/A")

# ── SDPA causal vs full speedup ───────────────────────────────────────────────
print()
print(SEP)
print("SDPA CAUSAL SPEEDUP: sdpa-causal vs sdpa-full  (reference)")
print(SEP)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    sf  = sdpa_results[key]["default"]["median"]
    sc  = sdpa_causal[key]["default"]["median"]
    spd = sf / sc
    print(f"B={B} H={H} N={N} d={d} ({label:<10}) | "
          f"full={sf:.4f} ms  causal={sc:.4f} ms  speedup={spd:.3f}×")

# ── Summary verdict ───────────────────────────────────────────────────────────
print()
print(SEP)
print("VERDICT SUMMARY")
print(SEP)

ratios_full   = []
ratios_causal = []

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    kf  = kernel_full.get(key)
    kc  = kernel_causal.get(key)
    sf  = sdpa_results[key]["default"]["median"]
    sc  = sdpa_causal[key]["default"]["median"]
    if kf and sf:
        ratios_full.append(kf / sf)
    if kc and sc:
        ratios_causal.append(kc / sc)

if ratios_full:
    med_ratio_full = sorted(ratios_full)[len(ratios_full)//2]
    print(f"flatargs vs SDPA-full   — median ratio: {med_ratio_full:.2f}x")
    if med_ratio_full <= 1.0:
        verdict = "FASTER than PyTorch SDPA. Publish with strong claim."
    elif med_ratio_full <= 2.0:
        verdict = "Competitive with PyTorch SDPA. Publish V2 with strong highlight."
    elif med_ratio_full <= 4.0:
        verdict = "Significant improvement over V1. Publish V2, document gap honestly."
    else:
        verdict = "Research-grade MVP. Publish V2 with careful framing."
    print(f"  → {verdict}")

if ratios_causal:
    med_ratio_causal = sorted(ratios_causal)[len(ratios_causal)//2]
    print(f"v2-causal vs SDPA-causal — median ratio: {med_ratio_causal:.2f}x")
    if med_ratio_causal <= 1.0:
        verdict = "FASTER than PyTorch SDPA causal. Publish with strong claim."
    elif med_ratio_causal <= 2.0:
        verdict = "Competitive with PyTorch SDPA causal."
    elif med_ratio_causal <= 4.0:
        verdict = "Significant improvement over naive. Gap vs SDPA documented."
    else:
        verdict = "Research-grade. Pure WMMA + warp-softmax exploration."
    print(f"  → {verdict}")

print()
print("Context:")
print("  PyTorch SDPA on SM75 does NOT use FlashAttention CUDA kernels.")
print("  FlashAttention official extension requires SM80+ (Ampere).")
print("  SDPA on SM75 uses the memory-efficient or math (unfused) backend.")
print("  A fair comparison is: our kernel vs what PyTorch actually runs on SM75.")
print(SEP)


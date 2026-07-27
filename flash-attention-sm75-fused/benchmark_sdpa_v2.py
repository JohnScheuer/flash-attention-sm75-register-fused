"""
benchmark_sdpa_v2.py
Versão corrigida com parsing robusto para ambos os binários.
"""

import torch
import torch.nn.functional as F
import subprocess
import re
import os

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda")
props  = torch.cuda.get_device_properties(device)
print(f"Device : {props.name}  SM{props.major}{props.minor}")
print(f"PyTorch: {torch.__version__}")
print()

# ── SDPA backend probe ────────────────────────────────────────────────────────
def probe_backend(enable_flash, enable_math, enable_mem_eff):
    q = torch.randn(1, 1, 16, 64, dtype=torch.float16, device=device)
    try:
        with torch.backends.cuda.sdp_kernel(
                enable_flash=enable_flash,
                enable_math=enable_math,
                enable_mem_efficient=enable_mem_eff):
            F.scaled_dot_product_attention(q, q, q)
        return True
    except RuntimeError:
        return False

flash_ok   = probe_backend(True,  False, False)
memeff_ok  = probe_backend(False, False, True)
print(f"Flash backend    : {'YES' if flash_ok  else 'NO (SM75 — requires SM80+)'}")
print(f"Mem-eff backend  : {'YES' if memeff_ok else 'NO'}")
print(f"Math backend     : YES (always)")
print()

# ── SDPA measurement ──────────────────────────────────────────────────────────
def measure_sdpa(B, H, N, d, causal=False, warmup=20, reps=300):
    Q = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
    K = torch.randn(B, H, N, d, dtype=torch.float16, device=device)
    V = torch.randn(B, H, N, d, dtype=torch.float16, device=device)

    # Use best available backend
    ctx = torch.backends.cuda.sdp_kernel(
        enable_flash=flash_ok,
        enable_math=True,
        enable_mem_efficient=memeff_ok)

    def run():
        with ctx:
            return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        t0.record(); run(); t1.record()
        torch.cuda.synchronize()
        samples.append(t0.elapsed_time(t1))

    samples.sort()
    return samples[len(samples)//2]  # median ms

# ── Kernel measurement — parsing robusto ──────────────────────────────────────
def measure_kernel(binary, B, H, N, outer=7, inner=200):
    """
    Parseia a mediana em ms da saída de qualquer kernel do projeto.
    Aceita dois formatos:
      Formato A (flatargs): linha com "ms  X.XXXX  X.XXXX  X.XXXX"
      Formato B (v2):       linha com "ms  X.XXXX  X.XXXX  X.XXXX" (mesmo)
      Formato C (fallback): qualquer linha com "median" seguido de número
    """
    if not os.path.isfile(f"./{binary}"):
        print(f"  [warn] {binary} not found")
        return None

    cmd = [f"./{binary}", str(B), str(H), str(N), str(outer), str(inner)]
    try:
        raw = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, timeout=180).decode()
    except subprocess.TimeoutExpired:
        print(f"  [warn] {binary} timeout")
        return None
    except subprocess.CalledProcessError as e:
        raw = e.output.decode() if e.output else ""

    # Tentar formato principal: "  ms    X.XXXX    X.XXXX    X.XXXX"
    # O segundo número é a mediana.
    m = re.search(
        r'\bms\b\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        raw)
    if m:
        return float(m.group(2))

    # Fallback: buscar "median" + número na mesma linha
    m = re.search(r'median\s*[:\|]?\s*([\d.]+)', raw, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # Fallback 2: buscar padrão "X.XXXX ms" onde há 3 valores em sequência
    nums = re.findall(r'([\d]+\.[\d]{3,4})\s*ms', raw)
    if len(nums) >= 2:
        return float(nums[1])  # segundo = mediana

    print(f"  [warn] {binary}: could not parse output")
    print(f"  Raw (first 400 chars): {raw[:400]}")
    return None

# ── Configs ───────────────────────────────────────────────────────────────────
configs = [
    (4, 12, 1024, 64, "peak-full"),
    (4,  8, 2048, 64, "peak-large"),
    (8, 12,  512, 64, "peak-wide"),
    (1, 16, 2048, 64, "peak-heads"),
    (1,  8,  256, 64, "medium"),
    (1,  1,  512, 64, "single-head"),
]

def tflops(B, H, N, d, ms_val, causal=False):
    if ms_val is None or ms_val <= 0:
        return None
    flops = B * H * 4.0 * N * N * d * (0.5 if causal else 1.0)
    return flops / (ms_val * 1e-3) / 1e12

def fmt_ms(v):
    return f"{v:.4f}" if v else "  N/A  "

def fmt_tf(v):
    return f"{v:.3f}" if v else " N/A  "

def ratio(kernel_ms, sdpa_ms):
    if kernel_ms is None or sdpa_ms is None or sdpa_ms == 0:
        return None
    return kernel_ms / sdpa_ms

def verdict(r):
    if r is None:    return "N/A"
    if r <= 1.00:    return f"{r:.2f}x ← FASTER than SDPA"
    if r <= 2.00:    return f"{r:.2f}x ← competitive"
    if r <= 4.00:    return f"{r:.2f}x ← significant gap"
    return           f"{r:.2f}x ← research-grade"

# ── Collect all numbers ───────────────────────────────────────────────────────
print("Collecting measurements (this takes a few minutes)...\n")

results = {}
for B, H, N, d, label in configs:
    key = (B, H, N, d)
    print(f"  {label}: B={B} H={H} N={N} d={d}")

    sdpa_full   = measure_sdpa(B, H, N, d, causal=False)
    sdpa_causal = measure_sdpa(B, H, N, d, causal=True)
    flat_ms     = measure_kernel("flash_mha_flatargs", B, H, N)
    v2_full_ms  = measure_kernel("flash_mha_v2",       B, H, N)

    # v2_causal: passar flag causal ao binário não é diretamente suportado
    # O binário flash_mha_v2 benchmarka AMBOS e imprime duas linhas.
    # Vamos parsear o resultado causal separadamente.
    v2_causal_ms = None
    if os.path.isfile("./flash_mha_v2"):
        cmd = ["./flash_mha_v2", str(B), str(H), str(N), "7", "200"]
        try:
            raw = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, timeout=180).decode()
            # Linha "CAUSAL" aparece depois de "FULL"
            # Formato: "CAUSAL    X.XXXX   X.XXXX   X.XXXX ..."
            m = re.search(
                r'CAUSAL\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                raw)
            if m:
                v2_causal_ms = float(m.group(2))  # mediana
            else:
                # Tentar segunda ocorrência do padrão "ms X X X"
                all_ms = re.findall(
                    r'\bms\b\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', raw)
                if len(all_ms) >= 2:
                    v2_causal_ms = float(all_ms[1][1])  # segunda linha, mediana
        except Exception:
            pass

    results[key] = {
        "label":        label,
        "sdpa_full":    sdpa_full,
        "sdpa_causal":  sdpa_causal,
        "flat":         flat_ms,
        "v2_full":      v2_full_ms,
        "v2_causal":    v2_causal_ms,
    }
    print(f"    SDPA full={fmt_ms(sdpa_full)}  causal={fmt_ms(sdpa_causal)}")
    print(f"    flatargs={fmt_ms(flat_ms)}  v2-full={fmt_ms(v2_full_ms)}  v2-causal={fmt_ms(v2_causal_ms)}")

SEP  = "=" * 115
SEP2 = "-" * 115

# ── Table 1: FULL ATTENTION ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("TABLE 1 — FULL ATTENTION  (median latency, ms)")
print(f"{'':28} {'flatargs':>10} {'SDPA-best':>10} {'ratio':>8}  "
      f"{'flatargs TF':>12} {'SDPA TF':>10}")
print(SEP2)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    r   = results[key]
    rat = ratio(r["flat"], r["sdpa_full"])
    print(f"B={B} H={H} N={N} d={d} ({label:<10})  "
          f"{fmt_ms(r['flat']):>10}  "
          f"{fmt_ms(r['sdpa_full']):>10}  "
          f"{verdict(rat):<30}  "
          f"{fmt_tf(tflops(B,H,N,d,r['flat'])):>8} TF  "
          f"{fmt_tf(tflops(B,H,N,d,r['sdpa_full'])):>8} TF")

# ── Table 2: CAUSAL ATTENTION ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("TABLE 2 — CAUSAL ATTENTION  (median latency, ms)")
print(f"{'':28} {'v2-causal':>10} {'SDPA-causal':>11} {'ratio':>8}  "
      f"{'v2 TF(caus)':>12} {'SDPA TF':>10}")
print(SEP2)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    r   = results[key]
    rat = ratio(r["v2_causal"], r["sdpa_causal"])
    print(f"B={B} H={H} N={N} d={d} ({label:<10})  "
          f"{fmt_ms(r['v2_causal']):>10}  "
          f"{fmt_ms(r['sdpa_causal']):>11}  "
          f"{verdict(rat):<30}  "
          f"{fmt_tf(tflops(B,H,N,d,r['v2_causal'],causal=True)):>8} TF  "
          f"{fmt_tf(tflops(B,H,N,d,r['sdpa_causal'],causal=True)):>8} TF")

# ── Table 3: Causal speedup (our kernels) ─────────────────────────────────────
print(f"\n{SEP}")
print("TABLE 3 — CAUSAL SPEEDUP  (v2-causal vs flatargs-full)")
print(f"{'':28} {'flatargs':>10} {'v2-causal':>10} {'speedup':>10}  "
      f"{'v2 TF(caus)':>12}")
print(SEP2)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    r   = results[key]
    spd = None
    if r["flat"] and r["v2_causal"]:
        spd = r["flat"] / r["v2_causal"]
    spd_str = f"{spd:.3f}x" if spd else "N/A"
    print(f"B={B} H={H} N={N} d={d} ({label:<10})  "
          f"{fmt_ms(r['flat']):>10}  "
          f"{fmt_ms(r['v2_causal']):>10}  "
          f"{spd_str:>10}  "
          f"{fmt_tf(tflops(B,H,N,d,r['v2_causal'],causal=True)):>8} TF")

# ── Table 4: SDPA causal vs full (reference) ─────────────────────────────────
print(f"\n{SEP}")
print("TABLE 4 — SDPA INTERNAL CAUSAL SPEEDUP  (reference baseline)")
print(f"{'':28} {'SDPA-full':>10} {'SDPA-caus':>10} {'speedup':>10}")
print(SEP2)

for B, H, N, d, label in configs:
    key = (B, H, N, d)
    r   = results[key]
    spd = None
    if r["sdpa_full"] and r["sdpa_causal"]:
        spd = r["sdpa_full"] / r["sdpa_causal"]
    spd_str = f"{spd:.3f}x" if spd else "N/A"
    print(f"B={B} H={H} N={N} d={d} ({label:<10})  "
          f"{fmt_ms(r['sdpa_full']):>10}  "
          f"{fmt_ms(r['sdpa_causal']):>10}  "
          f"{spd_str:>10}")

# ── VERDICT ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("VERDICT")
print(SEP)

ratios_full   = [ratio(results[(B,H,N,d)]["flat"],     results[(B,H,N,d)]["sdpa_full"])
                 for B,H,N,d,_ in configs
                 if ratio(results[(B,H,N,d)]["flat"], results[(B,H,N,d)]["sdpa_full"])]
ratios_causal = [ratio(results[(B,H,N,d)]["v2_causal"], results[(B,H,N,d)]["sdpa_causal"])
                 for B,H,N,d,_ in configs
                 if ratio(results[(B,H,N,d)]["v2_causal"], results[(B,H,N,d)]["sdpa_causal"])]

if ratios_full:
    med_f = sorted(ratios_full)[len(ratios_full)//2]
    print(f"\nflatargs vs SDPA-full     median ratio: {med_f:.2f}x")
    if   med_f <= 1.0: v = "FASTER than PyTorch SDPA. Publish with strong claim."
    elif med_f <= 2.0: v = "Competitive. Publish V2 with strong highlight."
    elif med_f <= 4.0: v = "Significant improvement over V1. Document gap honestly."
    else:              v = "Research-grade MVP. Careful framing required."
    print(f"  → {v}")

if ratios_causal:
    med_c = sorted(ratios_causal)[len(ratios_causal)//2]
    print(f"\nv2-causal vs SDPA-causal  median ratio: {med_c:.2f}x")
    if   med_c <= 1.0: v = "FASTER than PyTorch SDPA causal."
    elif med_c <= 2.0: v = "Competitive with PyTorch SDPA causal."
    elif med_c <= 4.0: v = "Significant improvement over naive causal."
    else:              v = "Research-grade causal implementation."
    print(f"  → {v}")

print(f"""
Context:
  PyTorch SDPA on SM75 uses mem-efficient or math backend.
  FlashAttention official CUDA extension requires SM80+ (Ampere).
  The mem-efficient backend IS a fused kernel (xformers lineage).
  SDPA-mem_eff is a legitimate and strong baseline for SM75.

  Our kernels are pure WMMA + warp-shuffle — no external dependencies.
  Target: educational/research correctness + competitive SM75 performance.
{SEP}""")


"""
Monkey-patch para substituir a atenção do HuggingFace pelo
FlashAttention SM75 customizado.

Estratégia:
  - Prefill (Q.seq_len == K.seq_len, sem KV cache): usa flash kernel
  - Decode step (Q.seq_len == 1, KV cache ativo): passa para original sem modificação
  - Qualquer outro caso incompatível: passa para original sem modificação
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional

from flash_attention_sm75 import flash_attention_forward


_original_sdpa = F.scaled_dot_product_attention
_patch_active   = False

_stats = {
    "total":            0,
    "flash":            0,
    "fallback":         0,
    "fallback_reasons": {},
}


def _record_fallback(reason: str):
    _stats["fallback"] += 1
    _stats["fallback_reasons"][reason] = \
        _stats["fallback_reasons"].get(reason, 0) + 1


def _patched_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    **kwargs,
):
    _stats["total"] += 1

    # Determine eligibility for our flash kernel
    reason = None

    if not query.is_cuda:
        reason = "not_cuda"
    elif query.dtype != torch.float16:
        reason = "not_fp16"
    elif query.dim() != 4:
        reason = "not_4d"
    elif query.size(-1) not in (64, 128):
        reason = "head_dim_unsupported"
    elif dropout_p != 0.0:
        reason = "dropout"
    elif attn_mask is not None:
        reason = "has_attn_mask"
    elif query.size(2) != key.size(2):
        # KV cache decode: Q is [B, H, 1, D], K is [B, H_kv, N, D]
        reason = "kvcache_decode"

    if reason is not None:
        _record_fallback(reason)
        # Pass everything through unchanged — do NOT modify args
        return _original_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )

    # Our flash kernel handles GQA expansion internally via ops.py
    _stats["flash"] += 1
    return flash_attention_forward(
        query, key, value,
        causal=is_causal,
        scale=scale,
    )


def apply_patch():
    global _patch_active
    if _patch_active:
        return

    # Patch both the module reference and the global name
    import torch.nn.functional as _F
    _F.scaled_dot_product_attention = _patched_sdpa

    # Some versions of transformers import directly from torch.nn.functional
    # so we also patch via the torch namespace
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

    _patch_active = True
    print("[flash-attention-sm75] Patch applied.")


def remove_patch():
    global _patch_active
    import torch.nn.functional as _F
    _F.scaled_dot_product_attention = _original_sdpa
    torch.nn.functional.scaled_dot_product_attention = _original_sdpa
    _patch_active = False
    print("[flash-attention-sm75] Patch removed.")


def patch_stats():
    return dict(_stats, active=_patch_active)


# ── End-to-end test ──────────────────────────────────────────────────────────

def run_generation_test(model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    device = "cuda"
    dtype  = torch.float16

    print(f"\n{'='*60}")
    print(f"Model : {model_name}")
    print(f"{'='*60}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    prompt = "The key insight of FlashAttention is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_new  = 64

    # ── Without patch ─────────────────────────────────────────────────────
    print("\n--- Generation WITHOUT flash-attention-sm75 patch ---")
    with torch.no_grad():
        t0      = time.perf_counter()
        out_ref = model.generate(**inputs, max_new_tokens=n_new, do_sample=False)
        t_ref   = time.perf_counter() - t0

    text_ref   = tokenizer.decode(out_ref[0], skip_special_tokens=True)
    tokens_ref = out_ref.shape[-1] - inputs["input_ids"].shape[-1]
    print(f"  {text_ref}")
    print(f"  Tokens : {tokens_ref}  |  "
          f"Time : {t_ref*1000:.0f} ms  |  "
          f"Throughput : {tokens_ref/t_ref:.1f} tok/s")

    # ── Apply patch ───────────────────────────────────────────────────────
    apply_patch()
    # Reset stats for clean measurement
    _stats["total"] = 0
    _stats["flash"] = 0
    _stats["fallback"] = 0
    _stats["fallback_reasons"].clear()

    print("\n--- Generation WITH flash-attention-sm75 patch ---")
    with torch.no_grad():
        t0      = time.perf_counter()
        out_pat = model.generate(**inputs, max_new_tokens=n_new, do_sample=False)
        t_pat   = time.perf_counter() - t0

    text_pat   = tokenizer.decode(out_pat[0], skip_special_tokens=True)
    tokens_pat = out_pat.shape[-1] - inputs["input_ids"].shape[-1]
    print(f"  {text_pat}")
    print(f"  Tokens : {tokens_pat}  |  "
          f"Time : {t_pat*1000:.0f} ms  |  "
          f"Throughput : {tokens_pat/t_pat:.1f} tok/s")

    # ── Stats ──────────────────────────────────────────────────────────────
    s = patch_stats()
    print(f"\n--- Patch stats ---")
    print(f"  Total SDPA calls   : {s['total']}")
    print(f"  → flash kernel     : {s['flash']}")
    print(f"  → fallback         : {s['fallback']}")
    if s["fallback_reasons"]:
        for reason, count in sorted(
            s["fallback_reasons"].items(), key=lambda x: -x[1]
        ):
            print(f"      {reason:<40} : {count}")

    # ── Correctness ────────────────────────────────────────────────────────
    print("\n--- Correctness ---")
    if text_ref == text_pat:
        print("  [PASS] Output text identical.")
    else:
        print("  [WARN] Output text differs:")
        print(f"    ref     : {text_ref}")
        print(f"    patched : {text_pat}")

    remove_patch()
    print(f"\n{'='*60}\nDone.\n{'='*60}\n")


if __name__ == "__main__":
    run_generation_test("Qwen/Qwen2-0.5B-Instruct")

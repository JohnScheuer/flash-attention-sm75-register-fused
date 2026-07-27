"""
Testa se o prefill com flash kernel produz logits idênticos ao baseline.
Sem geração, sem sampling — só um forward pass e comparação de logits.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from flash_attention_sm75 import flash_attention_forward
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
from llama_attention_replacement import apply_patch, remove_patch, _stats

def main():
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    device     = "cuda"
    dtype      = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True
    )
    model.eval()

    prompts = [
        "The key insight of FlashAttention is",
        "In the field of natural language processing,",
        "The attention mechanism was introduced",
    ]

    print(f"{'='*60}")
    print("Prefill logit comparison: baseline vs flash kernel")
    print(f"{'='*60}\n")

    all_pass = True

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Baseline forward (sem patch)
        with torch.no_grad():
            out_ref = model(**inputs)
        logits_ref = out_ref.logits  # [B, N, vocab]

        # Flash forward (com patch)
        apply_patch()
        _stats["total"] = 0; _stats["flash"] = 0; _stats["fallback"] = 0
        with torch.no_grad():
            out_flash = model(**inputs)
        logits_flash = out_flash.logits
        remove_patch()

        # Compara logits do último token (o que decide a próxima geração)
        last_ref   = logits_ref[0, -1, :].float()
        last_flash = logits_flash[0, -1, :].float()

        max_diff   = (last_ref - last_flash).abs().max().item()
        top1_ref   = last_ref.argmax().item()
        top1_flash = last_flash.argmax().item()
        tok_match  = (top1_ref == top1_flash)

        print(f"Prompt : \"{prompt[:50]}\"")
        print(f"  Flash calls        : {_stats['flash']}")
        print(f"  max_abs_err logits : {max_diff:.6f}")
        print(f"  top-1 token match  : {'[PASS]' if tok_match else '[FAIL]'}")
        print(f"  top-1 ref          : {tokenizer.decode([top1_ref])!r}")
        print(f"  top-1 flash        : {tokenizer.decode([top1_flash])!r}")

        if not tok_match:
            all_pass = False

        print()

    print("=" * 60)
    print("ALL PASSED" if all_pass else "SOME FAILED")
    print("=" * 60)

if __name__ == "__main__":
    main()

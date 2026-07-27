import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import torch
import flash_attn_sm75_cuda as _C

from reference.attention_ref import attention_reference
from reference.correctness import check_correctness, print_correctness


def run_case(batch, heads, seq_len, causal):
    print(f"\n--- FlashAttention MVP: B={batch} H={heads} N={seq_len} d=64 causal={causal} ---")

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, 64, dtype=torch.float16, device='cuda')
    K = torch.randn(batch, heads, seq_len, 64, dtype=torch.float16, device='cuda')
    V = torch.randn(batch, heads, seq_len, 64, dtype=torch.float16, device='cuda')

    scale = 1.0 / math.sqrt(64)

    ref = attention_reference(Q, K, V, causal=causal, scale=scale)
    out = _C.flash_attention_forward(Q, K, V, causal, scale)

    result = check_correctness(
        out, ref,
        name="flash_mvp vs reference",
        atol=1e-2, rtol=1e-2,
    )
    print_correctness(result)
    return result["allclose"]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    all_pass = True

    # Start small, then scale
    cases = [
        (1, 8, 128, False),
        (1, 8, 256, False),
        (1, 8, 512, False),
        (1, 16, 128, False),
        (2, 8, 128, False),

        # causal smoke
        (1, 8, 128, True),
        (1, 8, 256, True),
    ]

    for case in cases:
        ok = run_case(*case)
        all_pass = all_pass and ok

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()

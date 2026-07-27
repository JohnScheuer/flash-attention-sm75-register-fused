import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import torch
import flash_attn_sm75_cuda as _C

from reference.attention_ref import attention_reference
from reference.correctness import check_correctness, print_correctness


def run_case(batch, heads, seq_len, head_dim, causal):
    label = f"B={batch} H={heads} N={seq_len} d={head_dim} causal={causal}"
    print(f"\n--- {label} ---")

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
    scale = 1.0 / math.sqrt(head_dim)

    ref = attention_reference(Q, K, V, causal=causal, scale=scale)
    out = _C.flash_attention_forward(Q, K, V, causal, scale)

    result = check_correctness(
        out, ref,
        name="flash vs reference",
        atol=1e-2, rtol=1e-2,
    )
    print_correctness(result)
    return result["allclose"]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    all_pass = True

    print("=== d=64 regression ===")
    cases_64 = [
        (1, 8,  128, 64,  False),
        (1, 8,  512, 64,  False),
        (1, 8, 1024, 64,  True),
    ]
    for case in cases_64:
        ok = run_case(*case)
        all_pass = all_pass and ok

    print("\n=== d=128 new ===")
    cases_128 = [
        (1,  8,  128, 128, False),
        (1,  8,  256, 128, False),
        (1,  8,  512, 128, False),
        (1,  8, 1024, 128, False),
        (2,  8,  256, 128, False),
        (1, 16,  512, 128, False),
        (1,  8,  128, 128, True),
        (1,  8,  256, 128, True),
        (1,  8,  512, 128, True),
        (1,  8, 1024, 128, True),
    ]
    for case in cases_128:
        ok = run_case(*case)
        all_pass = all_pass and ok

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()

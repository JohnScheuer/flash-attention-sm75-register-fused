import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import torch
import flash_attn_sm75_cuda as _C

from reference.attention_ref import attention_reference
from reference.correctness import check_correctness, print_correctness


def run_case(batch, heads, seq_len, causal):
    label = f"B={batch} H={heads} N={seq_len} d=64 causal={causal}"
    print(f"\n--- WMMA QK: {label} ---")

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, 64, dtype=torch.float16, device='cuda')
    K = torch.randn(batch, heads, seq_len, 64, dtype=torch.float16, device='cuda')
    V = torch.randn(batch, heads, seq_len, 64, dtype=torch.float16, device='cuda')
    scale = 1.0 / math.sqrt(64)

    ref    = attention_reference(Q, K, V, causal=causal, scale=scale)
    scalar = _C.flash_attention_forward_scalar(Q, K, V, causal, scale)
    wmma   = _C.flash_attention_forward(Q, K, V, causal, scale)

    r1 = check_correctness(wmma, ref,    name="wmma_qk vs reference", atol=1e-2, rtol=1e-2)
    r2 = check_correctness(wmma, scalar, name="wmma_qk vs scalar",    atol=1e-2, rtol=1e-2)

    print_correctness(r1)
    print_correctness(r2)

    return r1["allclose"] and r2["allclose"]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    cases = [
        (1,  8, 128,  False),
        (1,  8, 256,  False),
        (1,  8, 512,  False),
        (1,  8, 1024, False),
        (1, 16, 128,  False),
        (2,  8, 256,  False),
        (1,  8, 128,  True),
        (1,  8, 256,  True),
        (1,  8, 512,  True),
        (1,  8, 1024, True),
    ]

    all_pass = True
    for case in cases:
        ok = run_case(*case)
        all_pass = all_pass and ok

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")


if __name__ == "__main__":
    main()

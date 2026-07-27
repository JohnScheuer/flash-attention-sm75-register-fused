"""
Test correctness of naive attention against PyTorch reference.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import math
import flash_attn_sm75_cuda as _C
from reference.attention_ref import attention_reference, attention_pytorch_sdpa
from reference.correctness import check_correctness, print_correctness


def run_test(batch, heads, seq_len, head_dim, causal, device='cuda'):
    label = f"B={batch} H={heads} N={seq_len} d={head_dim} causal={causal}"
    print(f"\n--- {label} ---")

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    K = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)
    V = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device=device)

    scale = 1.0 / math.sqrt(head_dim)

    # Reference
    ref = attention_reference(Q, K, V, causal=causal, scale=scale)

    # PyTorch SDPA
    sdpa = attention_pytorch_sdpa(Q, K, V, causal=causal, scale=scale)

    # Our naive CUDA
    naive = _C.naive_attention_forward(Q, K, V, causal, scale)

    # Check SDPA vs reference
    r1 = check_correctness(sdpa, ref, name="SDPA vs reference")
    print_correctness(r1)

    # Check naive vs reference
    r2 = check_correctness(naive, ref, name="naive_cuda vs reference")
    print_correctness(r2)

    # Check naive vs SDPA
    r3 = check_correctness(naive, sdpa, name="naive_cuda vs SDPA")
    print_correctness(r3)

    return r2["allclose"]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = 'cuda'
    all_pass = True

    # Standard shapes
    for causal in [False, True]:
        for head_dim in [64, 128]:
            for seq_len in [128, 256, 512, 1024]:
                for batch in [1, 2]:
                    for heads in [8, 16]:
                        ok = run_test(batch, heads, seq_len, head_dim, causal, device)
                        all_pass = all_pass and ok

    # Edge cases
    ok = run_test(1, 1, 1, 64, False, device)
    all_pass = all_pass and ok

    ok = run_test(1, 1, 16, 64, True, device)
    all_pass = all_pass and ok

    ok = run_test(1, 32, 128, 64, True, device)
    all_pass = all_pass and ok

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")


if __name__ == "__main__":
    main()

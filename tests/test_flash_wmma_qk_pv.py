import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import math, torch
import flash_attn_sm75_cuda as _C
from reference.attention_ref import attention_reference
from reference.correctness import check_correctness, print_correctness


def run_case(batch, heads, seq_len, causal):
    label = f"B={batch} H={heads} N={seq_len} d=64 causal={causal}"
    print(f"\n--- WMMA QK+PV: {label} ---")
    torch.manual_seed(42)
    Q = torch.randn(batch,heads,seq_len,64,dtype=torch.float16,device='cuda')
    K = torch.randn(batch,heads,seq_len,64,dtype=torch.float16,device='cuda')
    V = torch.randn(batch,heads,seq_len,64,dtype=torch.float16,device='cuda')
    scale = 1.0/math.sqrt(64)

    ref    = attention_reference(Q,K,V,causal=causal,scale=scale)
    wmma_qk = _C.flash_attention_forward_wmma_qk(Q,K,V,causal,scale)
    wmma_pv = _C.flash_attention_forward(Q,K,V,causal,scale)

    r1 = check_correctness(wmma_pv, ref,     name="wmma_qk_pv vs reference", atol=1e-2,rtol=1e-2)
    r2 = check_correctness(wmma_pv, wmma_qk, name="wmma_qk_pv vs wmma_qk",  atol=1e-2,rtol=1e-2)
    print_correctness(r1)
    print_correctness(r2)
    return r1["allclose"] and r2["allclose"]


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    cases = [
        (1, 8, 128,  False),
        (1, 8, 256,  False),
        (1, 8, 512,  False),
        (1, 8, 1024, False),
        (2, 8, 256,  False),
        (1,16, 512,  False),
        (1, 8, 128,  True),
        (1, 8, 256,  True),
        (1, 8, 512,  True),
        (1, 8, 1024, True),
    ]
    all_pass = all(run_case(*c) for c in cases)
    print("\n" + "="*60)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")

if __name__ == "__main__":
    main()

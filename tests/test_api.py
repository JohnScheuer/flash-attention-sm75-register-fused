import torch
import traceback
from flash_attention_sm75 import flash_attention_forward

def main():
    print("Testing Python API for flash_attention_sm75...\n")
    
    Q = torch.randn(1, 8, 128, 64, dtype=torch.float16, device='cuda')
    K = torch.randn(1, 8, 128, 64, dtype=torch.float16, device='cuda')
    V = torch.randn(1, 8, 128, 64, dtype=torch.float16, device='cuda')

    # Test 1: Forward normal
    try:
        out = flash_attention_forward(Q, K, V, causal=True)
        print("[PASS] Forward pass runs successfully.")
        assert out.shape == Q.shape
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")

    # Test 2: Unsupported head dim
    Q_bad = torch.randn(1, 8, 128, 96, dtype=torch.float16, device='cuda')
    try:
        flash_attention_forward(Q_bad, Q_bad, Q_bad)
        print("[FAIL] Did not catch bad head_dim.")
    except RuntimeError as e:
        if "supports head_dim=64 or 128" in str(e):
            print("[PASS] Successfully caught unsupported head_dim.")
        else:
            print(f"[FAIL] Wrong error message for head_dim: {e}")

    # Test 3: Backward failure
    try:
        Q.requires_grad = True
        out = flash_attention_forward(Q, K, V)
        out.sum().backward()
        print("[FAIL] Did not catch backward pass.")
    except NotImplementedError as e:
        if "Backward pass is not implemented" in str(e):
            print("[PASS] Successfully blocked backward pass gracefully.")
        else:
            print(f"[FAIL] Wrong error message for backward: {e}")

if __name__ == "__main__":
    main()

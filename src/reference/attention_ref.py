"""
Reference attention implementation in PyTorch for correctness validation.
"""

import torch
import torch.nn.functional as F
import math


def attention_reference(Q, K, V, causal=False, scale=None):
    """
    Standard attention: O = softmax(Q @ K^T / sqrt(d)) @ V
    
    Args:
        Q: [batch, heads, seq_len, head_dim] float16
        K: [batch, heads, seq_len, head_dim] float16
        V: [batch, heads, seq_len, head_dim] float16
        causal: bool
        scale: float (default: 1/sqrt(head_dim))
    
    Returns:
        O: [batch, heads, seq_len, head_dim] float16
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.size(-1))

    # Compute in FP32 for reference accuracy
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()

    S = torch.matmul(Q_f, K_f.transpose(-2, -1)) * scale

    if causal:
        seq_len = Q.size(-2)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        S.masked_fill_(mask, float('-inf'))

    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V_f)

    return O.half()


def attention_pytorch_sdpa(Q, K, V, causal=False, scale=None):
    """
    PyTorch SDPA for comparison.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.size(-1))

    return F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=causal,
        scale=scale,
    )

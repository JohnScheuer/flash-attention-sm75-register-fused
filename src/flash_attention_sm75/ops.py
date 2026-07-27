import math
import torch
import flash_attn_sm75_cuda as _C


def _expand_kv_for_gqa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Expande K e V para casar com o número de heads de Q (GQA/MQA).
    Q:  [B, H_q,  N, D]
    K:  [B, H_kv, N, D]   H_kv divides H_q
    V:  [B, H_kv, N, D]
    """
    h_q  = Q.size(1)
    h_kv = K.size(1)

    if h_q == h_kv:
        return K, V

    if h_q % h_kv != 0:
        raise RuntimeError(
            f"GQA expansion: Q heads ({h_q}) not divisible by K/V heads ({h_kv})"
        )

    factor = h_q // h_kv
    K = K.repeat_interleave(factor, dim=1).contiguous()
    V = V.repeat_interleave(factor, dim=1).contiguous()
    return K, V


class FlashAttentionSM75Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, scale):
        head_dim = Q.size(-1)

        # GQA/MQA expansion FIRST, before any contiguous call
        K, V = _expand_kv_for_gqa(Q, K, V)

        # Now make contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        out = _C.flash_attention_forward(Q, K, V, causal, float(scale))
        ctx.save_for_backward(Q, K, V, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "Backward pass not implemented. "
            "flash-attention-sm75 is for inference only."
        )


def flash_attention_forward(Q, K, V, causal=False, scale=None):
    """
    Drop-in for F.scaled_dot_product_attention on NVIDIA Turing (SM75).
    Supports MHA and GQA/MQA.

    Args:
        Q: [B, H_q,  N, D]  float16  cuda
        K: [B, H_kv, N, D]  float16  cuda
        V: [B, H_kv, N, D]  float16  cuda
        causal: bool
        scale:  float (default 1/sqrt(D))

    Returns:
        O: [B, H_q, N, D]  float16  cuda
    """
    if Q.dim() != 4:
        raise ValueError("Q must be 4D [B, H, N, D]")

    head_dim = Q.size(-1)
    if head_dim not in (64, 128):
        raise RuntimeError(
            f"flash_attention_sm75 supports head_dim=64 or 128, got {head_dim}"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    return FlashAttentionSM75Function.apply(Q, K, V, causal, scale)

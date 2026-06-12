# LoRA (Low-Rank Adaptation) module for VGGT Aggregator blocks.
#
# Provides LoRALinear (wraps a frozen nn.Linear with trainable low-rank adapters)
# and LoRABlock (wraps a frozen Attention Block to produce LoRA-adapted outputs).
#
# Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.

import math
import torch
import torch.nn as nn
from typing import Optional


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around a frozen nn.Linear.

    Given a frozen linear layer W ∈ R^{d_out × d_in}, LoRA adds a trainable
    bypass: y = W·x + (α/r) · B·A·x, where A ∈ R^{r × d_in} and B ∈ R^{d_out × r}.

    The original linear's parameters are NOT copied — we keep a reference to it.
    Only A and B are trainable.

    Args:
        original_linear: The frozen nn.Linear to wrap.
        rank: Rank of the low-rank decomposition.
        alpha: Scaling factor (the LoRA "alpha").
        dropout: Dropout probability applied before the low-rank path.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear  # frozen, not copied
        d_in = original_linear.in_features
        d_out = original_linear.out_features

        self.rank = rank
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)

        # Dropout (optional)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path (no grad through original_linear)
        base_out = self.original_linear(x)
        # LoRA path (trainable)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base_out + lora_out * self.scaling


class LoRAAttention(nn.Module):
    """Wraps an existing Attention module with LoRA adapters on qkv and proj.

    The original Attention's parameters stay frozen. LoRA adapters are added
    on top of `attn.qkv` and `attn.proj`.

    Args:
        original_attn: The frozen Attention module to wrap.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_attn = original_attn  # frozen reference

        # Create LoRA wrappers for qkv and proj
        self.lora_qkv = LoRALinear(original_attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
        self.lora_proj = LoRALinear(original_attn.proj, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        """Forward with LoRA-adapted qkv and proj, using FlashAttention."""
        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend

        attn = self.original_attn
        B, N, C = x.shape

        # Use LoRA-adapted qkv instead of original
        qkv = self.lora_qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)

        # Apply RoPE if available
        if attn.rope is not None and pos is not None:
            q = attn.rope(q, pos)
            k = attn.rope(k, pos)

        # FlashAttention backend dispatch (matching Attention class)
        dropout_p = attn.attn_drop.p if self.training else 0.0
        if q.dtype == torch.bfloat16:
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            with torch.nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        x = x.transpose(1, 2).reshape(B, N, C)

        # Use LoRA-adapted proj instead of original
        x = self.lora_proj(x)
        x = attn.proj_drop(x)
        return x


class LoRABlock(nn.Module):
    """Wraps a frozen Block with LoRA-adapted attention.

    This module produces LoRA-adapted output by:
    1. Using LoRAAttention (LoRA on qkv + proj) for the attention path
    2. Reusing the original frozen Block's LayerNorm, MLP, LayerScale, etc.

    The original Block is NOT modified. LoRABlock references its submodules.

    Args:
        original_block: The frozen Block to adapt.
        rank: LoRA rank.
        alpha: LoRA alpha.
        dropout: LoRA dropout.
    """

    def __init__(
        self,
        original_block: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_block = original_block  # frozen reference

        # Create LoRA-adapted attention
        self.lora_attn = LoRAAttention(
            original_block.attn, rank=rank, alpha=alpha, dropout=dropout
        )

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        """Forward: frozen norm/mlp/layerscale + LoRA-adapted attention."""
        blk = self.original_block

        # Attention path with LoRA
        attn_out = self.lora_attn(blk.norm1(x), pos=pos)
        attn_out = blk.ls1(attn_out)
        x = x + attn_out

        # MLP path (frozen, but still computed — needed for correct output)
        mlp_out = blk.ls2(blk.mlp(blk.norm2(x)))
        x = x + mlp_out

        return x

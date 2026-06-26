# d4rt-style independent-query cross-attention decoder (adapted for inverse rendering).
#
# Borrowed paradigm from Open-d4rt (src/model/{query_embedding,decoder}.py):
#   query = Fourier(u,v) [+ Fourier(direction)] [+ local RGB patch]
#         -> cross-attend to backbone memory tokens (NO query self-attention)
#         -> per-query latent -> task head.
#
# Used here to predict, per pixel: materials (mlp tail) and per-pixel HDR env
# (direction written INTO the query). See d4rt_inverse_rendering_design.md.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FourierFeatures(nn.Module):
    """sin/cos positional encoding for a low-dim coordinate (uv: 2, direction: 3)."""

    def __init__(self, input_dim: int, num_bands: int = 8, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_bands).float() * torch.pi  # [num_bands]
        self.register_buffer("freqs", freqs, persistent=False)
        self.output_dim = input_dim * (2 * num_bands + (1 if include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., input_dim]
        xb = x.unsqueeze(-1) * self.freqs  # [..., input_dim, num_bands]
        feats = [torch.sin(xb), torch.cos(xb)]
        out = torch.cat([t.flatten(-2) for t in feats], dim=-1)  # [..., input_dim*2*num_bands]
        if self.include_input:
            out = torch.cat([x, out], dim=-1)
        return out


class QueryEmbedder(nn.Module):
    """Build per-query tokens from (u,v) [+ direction] [+ local RGB patch].

    Args:
        hidden_dim: decoder dim D.
        use_direction: add a Fourier(unit-vector direction) term (lighting branch).
        use_patch: add a local KxK RGB patch term (material branch high-freq detail).
        patch_size: K for the local RGB patch.
    """

    def __init__(self, hidden_dim: int, use_direction: bool = False,
                 use_patch: bool = False, patch_size: int = 9, num_bands: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_direction = use_direction
        self.use_patch = use_patch
        self.patch_size = patch_size

        self.uv_enc = FourierFeatures(2, num_bands=num_bands)
        self.uv_proj = nn.Linear(self.uv_enc.output_dim, hidden_dim)

        if use_direction:
            self.dir_enc = FourierFeatures(3, num_bands=num_bands)
            self.dir_proj = nn.Linear(self.dir_enc.output_dim, hidden_dim)

        if use_patch:
            self.patch_proj = nn.Sequential(
                nn.Linear(3 * patch_size * patch_size, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def _local_patches(self, image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sample a KxK RGB patch around each (u,v). image:[B,3,H,W], u/v:[B,M] in [0,1]."""
        B, M = u.shape
        K = self.patch_size
        # build KxK sampling grid offsets in normalized coords
        dev = image.device
        H, W = image.shape[-2:]
        lin = torch.linspace(-(K // 2), K // 2, K, device=dev)
        oy, ox = torch.meshgrid(lin, lin, indexing="ij")          # [K,K]
        ox = (ox / max(W - 1, 1)) * 2.0                            # step in grid_sample coords (~2/W per px)
        oy = (oy / max(H - 1, 1)) * 2.0
        # base grid in [-1,1]
        gx = (u * 2 - 1).view(B, M, 1, 1) + ox.view(1, 1, K, K)
        gy = (v * 2 - 1).view(B, M, 1, 1) + oy.view(1, 1, K, K)
        grid = torch.stack([gx, gy], dim=-1).reshape(B, M, K * K, 2)  # [B, M, K*K, 2]
        sampled = F.grid_sample(image, grid, mode="bilinear", align_corners=True,
                                padding_mode="border")            # [B, 3, M, K*K]
        return sampled.permute(0, 2, 1, 3).reshape(B, M, 3 * K * K)

    def forward(self, u: torch.Tensor, v: torch.Tensor,
                direction: Optional[torch.Tensor] = None,
                image: Optional[torch.Tensor] = None) -> torch.Tensor:
        # u,v: [B,M]; direction: [B,M,3]; image: [B,3,H,W]
        uv = torch.stack([u, v], dim=-1)                          # [B,M,2]
        token = self.uv_proj(self.uv_enc(uv))
        if self.use_direction:
            assert direction is not None, "direction query enabled but none provided"
            token = token + self.dir_proj(self.dir_enc(direction))
        if self.use_patch and image is not None:
            token = token + self.patch_proj(self._local_patches(image, u, v))
        return self.out_norm(token)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN. Queries attend to memory; no query self-attn."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 3.5, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.norm_q(q), self.norm_kv(memory), self.norm_kv(memory), need_weights=False)
        q = q + a
        q = q + self.ff(self.norm_ff(q))
        return q


class IndependentQueryDecoder(nn.Module):
    """Stack of cross-attention blocks (queries independent across the set)."""

    def __init__(self, hidden_dim: int, num_layers: int = 8, num_heads: int = 16, mlp_ratio: float = 3.5):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = query
        for blk in self.blocks:
            x = blk(x, memory)
        return self.out_norm(x)

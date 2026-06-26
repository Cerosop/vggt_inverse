# d4rt-style inverse rendering heads (NEW path, switchable with the old InverseHeads).
#
# Two branches, both driven by the query cross-attention decoder over VGGT tokens:
#   - Material  : per-pixel queries (UV + local RGB patch) -> MLP -> albedo/metallic/
#                 roughness/normal/shading at the patch grid -> bilinear upsample.
#   - Lighting  : per-(pixel, direction) queries (direction WRITTEN INTO the query)
#                 -> MLP -> RGB radiance. Training samples random (pixel,dir) pairs;
#                 inference queries the full (Hs x Ws x env_h x env_w) grid -> per-pixel env.
#
# See d4rt_inverse_rendering_design.md.  Default-off; old SG/DPT path is untouched.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

from vggt.heads.query_decoder import QueryEmbedder, IndependentQueryDecoder


def _hemisphere_dirs(env_h: int, env_w: int) -> torch.Tensor:
    """Unit vectors for an (env_h x env_w) upper-hemisphere grid, pole = +y (local normal).

    elevation theta in [0, pi/2] over rows; azimuth phi in [0, 2pi) over cols.
    Returns [env_h*env_w, 3].
    """
    theta = (torch.arange(env_h).float() + 0.5) / env_h * (math.pi / 2)   # [env_h]
    phi = (torch.arange(env_w).float() + 0.5) / env_w * (2 * math.pi)     # [env_w]
    th, ph = torch.meshgrid(theta, phi, indexing="ij")                    # [env_h, env_w]
    x = torch.sin(th) * torch.cos(ph)
    y = torch.cos(th)                 # pole (+y) = normal
    z = torch.sin(th) * torch.sin(ph)
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)                  # [env_h*env_w, 3]


class D4RTInverseHeads(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        decoder_dim: int = 1024,
        num_layers: int = 8,
        num_heads: int = 16,
        patch_size: int = 14,
        enable_material: bool = True,
        enable_lighting: bool = True,
        env_h: int = 8,
        env_w: int = 16,
        light_spatial_h: int = 60,
        light_spatial_w: int = 80,
        num_light_samples: int = 2048,
        material_patch_size: int = 9,
        frames_chunk_size: int = 8,   # accepted for API parity (unused here)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.enable_material = enable_material
        self.enable_lighting = enable_lighting
        self.env_h, self.env_w = env_h, env_w
        self.light_spatial_h, self.light_spatial_w = light_spatial_h, light_spatial_w
        self.num_light_samples = num_light_samples
        self.head_names = ["albedo", "metallic", "roughness", "normal", "shading"]

        self.mem_proj = nn.Linear(dim_in, decoder_dim)

        if enable_material:
            self.mat_query = QueryEmbedder(decoder_dim, use_direction=False,
                                           use_patch=True, patch_size=material_patch_size)
            self.mat_decoder = IndependentQueryDecoder(decoder_dim, num_layers, num_heads)
            self.mat_trunk = nn.Sequential(nn.Linear(decoder_dim, decoder_dim), nn.GELU())
            self.mat_heads = nn.ModuleDict({
                "albedo": nn.Linear(decoder_dim, 3),
                "metallic": nn.Linear(decoder_dim, 1),
                "roughness": nn.Linear(decoder_dim, 1),
                "normal": nn.Linear(decoder_dim, 3),
                "shading": nn.Linear(decoder_dim, 3),
            })

        if enable_lighting:
            self.light_query = QueryEmbedder(decoder_dim, use_direction=True, use_patch=False)
            self.light_decoder = IndependentQueryDecoder(decoder_dim, num_layers, num_heads)
            self.light_head = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, 3))
            self.register_buffer("dir_grid", _hemisphere_dirs(env_h, env_w), persistent=False)  # [D_env,3]

    # ---- memory from VGGT tokens (use last intermediate's patch tokens) ----
    def _build_memory(self, tokens_list: List[torch.Tensor], patch_start_idx: int):
        x = tokens_list[-1]                       # [B,S,P,2048]
        B, S, P, C = x.shape
        patch = x[:, :, patch_start_idx:, :]      # [B,S,Npatch,2048]
        Npatch = patch.shape[2]
        mem = self.mem_proj(patch).reshape(B * S, Npatch, -1)   # [B*S, Npatch, D]
        return mem, B, S, Npatch

    @staticmethod
    def _apply_material_activation(name, x):
        if name == "normal":
            return F.normalize(torch.tanh(x), p=2, dim=-1, eps=1e-8)
        return torch.sigmoid(x)

    def _material_branch(self, mem, images, B, S):
        # patch grid resolution
        _, _, H, W = images.shape  # images here are [B*S,3,H,W]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        dev = mem.device
        # grid uv centers in [0,1]
        us = (torch.arange(Wp, device=dev).float() + 0.5) / Wp
        vs = (torch.arange(Hp, device=dev).float() + 0.5) / Hp
        vv, uu = torch.meshgrid(vs, us, indexing="ij")            # [Hp,Wp]
        u = uu.reshape(1, -1).expand(B * S, -1)                   # [B*S, Hp*Wp]
        v = vv.reshape(1, -1).expand(B * S, -1)
        q = self.mat_query(u, v, image=images)                   # [B*S, M, D]
        z = self.mat_decoder(q, mem)                             # [B*S, M, D]
        z = self.mat_trunk(z)
        out = {}
        for name in self.head_names:
            raw = self.mat_heads[name](z)                        # [B*S, M, c]
            c = raw.shape[-1]
            grid = raw.reshape(B * S, Hp, Wp, c).permute(0, 3, 1, 2)  # [B*S,c,Hp,Wp]
            up = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
            up = up.permute(0, 2, 3, 1)                            # [B*S,H,W,c]
            up = self._apply_material_activation(name, up)
            out[name] = up.reshape(B, S, H, W, c)
        return out

    def _light_radiance(self, mem, u, v, direction):
        """mem:[B*S,N,D]; u,v:[B*S,M]; direction:[B*S,M,3] -> radiance [B*S,M,3] (softplus)."""
        q = self.light_query(u, v, direction=direction)
        z = self.light_decoder(q, mem)
        return F.softplus(self.light_head(z))

    def _light_branch_sampled(self, mem, B, S):
        """Training: sample random (spatial pixel, direction) pairs; predict radiance."""
        BS = B * S
        M = self.num_light_samples
        Hs, Ws = self.light_spatial_h, self.light_spatial_w
        Dn = self.env_h * self.env_w
        dev = mem.device
        spatial_idx = torch.randint(0, Hs * Ws, (BS, M), device=dev)   # [BS,M]
        dir_idx = torch.randint(0, Dn, (BS, M), device=dev)            # [BS,M]
        sh = (spatial_idx // Ws).float(); sw = (spatial_idx % Ws).float()
        u = (sw + 0.5) / Ws; v = (sh + 0.5) / Hs                       # [BS,M]
        direction = self.dir_grid[dir_idx]                             # [BS,M,3]
        rad = self._light_radiance(mem, u, v, direction)              # [BS,M,3]
        return (rad.reshape(B, S, M, 3),
                spatial_idx.reshape(B, S, M),
                dir_idx.reshape(B, S, M))

    @torch.no_grad()
    def predict_env_dense(self, tokens_list, patch_start_idx, spatial_h=None, spatial_w=None):
        """Inference: full per-pixel env [B,S,Hs,Ws,env_h,env_w,3] at a spatial grid."""
        mem, B, S, _ = self._build_memory(tokens_list, patch_start_idx)
        Hs = spatial_h or self.light_spatial_h
        Ws = spatial_w or self.light_spatial_w
        Dn = self.env_h * self.env_w
        dev = mem.device
        sh = (torch.arange(Hs, device=dev).float() + 0.5) / Hs
        sw = (torch.arange(Ws, device=dev).float() + 0.5) / Ws
        vv, uu = torch.meshgrid(sh, sw, indexing="ij")                # [Hs,Ws]
        # one query per (spatial, direction)
        u = uu.reshape(-1)[:, None].expand(Hs * Ws, Dn).reshape(-1)   # [Hs*Ws*Dn]
        v = vv.reshape(-1)[:, None].expand(Hs * Ws, Dn).reshape(-1)
        d = self.dir_grid[None].expand(Hs * Ws, Dn, 3).reshape(-1, 3)
        u = u[None].expand(B * S, -1); v = v[None].expand(B * S, -1)
        d = d[None].expand(B * S, -1, -1)
        rad = self._light_radiance(mem, u, v, d)                      # [BS, Hs*Ws*Dn, 3]
        return rad.reshape(B, S, Hs, Ws, self.env_h, self.env_w, 3)

    def forward(self, tokens_list: List[torch.Tensor], images: torch.Tensor,
                patch_start_idx: int) -> Dict[str, torch.Tensor]:
        B, S, _, H, W = images.shape
        mem, _, _, _ = self._build_memory(tokens_list, patch_start_idx)
        images_bs = images.reshape(B * S, 3, H, W)
        preds: Dict[str, torch.Tensor] = {}

        if self.enable_material:
            preds.update(self._material_branch(mem, images_bs, B, S))

        if self.enable_lighting:
            rad, spix, didx = self._light_branch_sampled(mem, B, S)
            preds["light_pred"] = rad           # [B,S,M,3]
            preds["light_spatial_idx"] = spix   # [B,S,M]
            preds["light_dir_idx"] = didx       # [B,S,M]
            # Eval: also emit a coarse dense env for visualization (cheap grid).
            if not self.training:
                gh = min(self.light_spatial_h, 16)
                gw = min(self.light_spatial_w, 16)
                preds["pred_env_pixel"] = self.predict_env_dense(
                    tokens_list, patch_start_idx, gh, gw)   # [B,S,gh,gw,env_h,env_w,3]
        return preds

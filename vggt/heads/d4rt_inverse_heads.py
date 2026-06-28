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
from torch.utils.checkpoint import checkpoint
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
        num_material_samples: int = 0,   # 0 = dense grid+upsample (old); >0 = per-pixel sampled
        enable_render: bool = False,
        num_render_pixels: int = 64,
        render_demo_upsample: int = 4,    # demo render: 1 lighting sample per f×f block, upsample f
        enable_dynamic_weighting: bool = False,
        cross_frame: bool = False,        # True = each query attends to ALL frames (d4rt-style)
        max_frames: int = 32,
        frames_chunk_size: int = 8,   # accepted for API parity (unused here)
    ):
        super().__init__()
        self.patch_size = patch_size
        self.enable_material = enable_material
        self.enable_lighting = enable_lighting
        self.env_h, self.env_w = env_h, env_w
        self.light_spatial_h, self.light_spatial_w = light_spatial_h, light_spatial_w
        self.num_light_samples = num_light_samples
        self.num_material_samples = num_material_samples
        self.enable_render = enable_render
        self.num_render_pixels = num_render_pixels
        self.render_demo_upsample = render_demo_upsample
        self.cross_frame = cross_frame
        self._npatch = None               # set in _build_memory (for frame-id recovery)
        self.head_names = ["albedo", "metallic", "roughness", "normal", "shading"]

        # Memory projection. Match d4rt: Identity when no dim change (no resolution loss),
        # a Linear only if decoder_dim != backbone dim. Set decoder_dim == dim_in (2048)
        # to keep the full backbone features (no reduction).
        self.mem_proj = nn.Identity() if decoder_dim == dim_in else nn.Linear(dim_in, decoder_dim)
        # Frame embedding for cross-frame attention (added to memory tokens by their frame
        # id AND to each query by its target frame, so a query can tell its own frame from
        # the others while still attending across all frames — like d4rt's t_src/t_tgt).
        self.frame_embed = nn.Embedding(max_frames, decoder_dim) if cross_frame else None

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
            # Per-cell solid angle for the render integral (matches dir_grid ordering).
            _th = (torch.arange(env_h).float() + 0.5) / env_h * (math.pi / 2)
            _sa = (torch.sin(_th) * (math.pi / 2 / env_h) * (2 * math.pi / env_w))
            _sa = _sa[:, None].expand(env_h, env_w).reshape(-1)
            self.register_buffer("solid_angle", _sa, persistent=False)  # [D_env]

        # Dynamic loss weighting (Kendall): one learnable log-variance per task,
        # covering the material heads AND the per-pixel light losses. The loss module
        # reads task_log_var / task_log_var_names from predictions and applies
        # 0.5*exp(-s)*L + 0.5*s per task. Tasks not listed fall back to static weights.
        self.enable_dynamic_weighting = enable_dynamic_weighting
        names = []
        if enable_material:
            names += list(self.head_names)               # albedo/metallic/roughness/normal/shading
        if enable_lighting:
            names += ["per_pixel_env"]
        if enable_render:
            names += ["per_pixel_render"]
        self.dyn_task_names = names
        if enable_dynamic_weighting and len(names) > 0:
            self.task_log_var = nn.Parameter(torch.zeros(len(names)))
        else:
            self.register_parameter("task_log_var", None)

    # ---- memory from VGGT tokens (use last intermediate's patch tokens) ----
    def _build_memory(self, tokens_list: List[torch.Tensor], patch_start_idx: int):
        x = tokens_list[-1]                       # [B,S,P,2048]
        B, S, P, C = x.shape
        patch = x[:, :, patch_start_idx:, :]      # [B,S,Npatch,2048]
        Npatch = patch.shape[2]
        self._npatch = Npatch                     # remember for query frame-id recovery
        mem = self.mem_proj(patch)                # [B,S,Npatch,D]
        D = mem.shape[-1]
        if self.cross_frame:
            # tag each token with its frame id, then make EVERY query (of a scene) attend
            # to ALL frames of that scene: mem -> [B*S, S*Npatch, D] (replicated per frame).
            fe = self.frame_embed(torch.arange(S, device=mem.device))   # [S, D]
            mem = mem + fe[None, :, None, :]
            mem = mem.reshape(B, S * Npatch, D)
            mem = mem[:, None].expand(B, S, S * Npatch, D).reshape(B * S, S * Npatch, D)
        else:
            mem = mem.reshape(B * S, Npatch, D)   # [B*S, Npatch, D]
        return mem, B, S, Npatch

    def _add_query_frame_embed(self, q, mem):
        """Add each query's own-frame embedding (cross-frame mode). q:[B*S,M,D]."""
        if not self.cross_frame or self.frame_embed is None or self._npatch is None:
            return q
        BS = q.shape[0]
        S = max(1, mem.shape[1] // self._npatch)
        fid = torch.arange(BS, device=q.device) % S    # frame id of each batch element
        return q + self.frame_embed(fid)[:, None, :]

    @staticmethod
    def _apply_material_activation(name, x):
        if name == "normal":
            return F.normalize(torch.tanh(x), p=2, dim=-1, eps=1e-8)
        return torch.sigmoid(x)

    def _material_features(self, mem, images_bs, u, v):
        """Per-query RAW (pre-activation) material head outputs.

        u,v:[BS,M] in [0,1] -> dict name -> [BS,M,c]. d4rt-style: the query is an
        arbitrary (u,v) coordinate (NOT a fixed token grid), so this supports both a
        dense grid and random per-pixel sampling.
        """
        q = self.mat_query(u, v, image=images_bs)                # [BS,M,D]
        q = self._add_query_frame_embed(q, mem)
        z = self.mat_trunk(self.mat_decoder(q, mem))             # [BS,M,D]
        return {name: self.mat_heads[name](z) for name in self.head_names}

    def _material_dense(self, mem, images_bs, B, S):
        """Eval/old path: query the patch grid, bilinear upsample to full res."""
        _, _, H, W = images_bs.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        dev = mem.device
        us = (torch.arange(Wp, device=dev).float() + 0.5) / Wp
        vs = (torch.arange(Hp, device=dev).float() + 0.5) / Hp
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        u = uu.reshape(1, -1).expand(B * S, -1)
        v = vv.reshape(1, -1).expand(B * S, -1)
        raw = self._material_features(mem, images_bs, u, v)
        out = {}
        for name, r in raw.items():
            c = r.shape[-1]
            grid = r.reshape(B * S, Hp, Wp, c).permute(0, 3, 1, 2)
            up = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
            up = self._apply_material_activation(name, up.permute(0, 2, 3, 1))
            out[name] = up.reshape(B, S, H, W, c)
        return out

    def _material_sampled(self, mem, images_bs, B, S):
        """Train path: query num_material_samples RANDOM per-pixel coords (d4rt-style)."""
        BS = B * S
        N = self.num_material_samples
        dev = mem.device
        u = torch.rand(BS, N, device=dev)                        # [BS,N] in [0,1)
        v = torch.rand(BS, N, device=dev)
        raw = self._material_features(mem, images_bs, u, v)
        out = {name: self._apply_material_activation(name, r).reshape(B, S, N, -1)
               for name, r in raw.items()}
        uv = torch.stack([u, v], dim=-1).reshape(B, S, N, 2)     # [B,S,N,2]
        return out, uv

    def _material_at(self, mem, images_bs, u, v, B, S):
        """Activated per-pixel material at given uv [BS,M] -> dict name->[B,S,M,c]."""
        raw = self._material_features(mem, images_bs, u, v)
        return {name: self._apply_material_activation(name, r).reshape(B, S, u.shape[1], -1)
                for name, r in raw.items()}

    def _light_radiance(self, mem, u, v, direction):
        """mem:[B*S,N,D]; u,v:[B*S,M]; direction:[B*S,M,3] -> radiance [B*S,M,3] (softplus)."""
        q = self.light_query(u, v, direction=direction)
        q = self._add_query_frame_embed(q, mem)
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

    def _light_env_tiles(self, mem, u, v, chunk=2048):
        """Full env (all Dn dirs) at P pixels. u,v:[BS,P] -> [BS,P,Dn,3].

        The render loss needs every pixel's full Dn-direction env, i.e. P*Dn light
        queries WITH gradients — far more than the (pixel,dir) env loss. To keep this
        affordable we (a) chunk the queries, and (b) gradient-checkpoint each chunk so
        backward recomputes its activations instead of storing them. Peak memory is
        then bounded by one chunk regardless of num_render_pixels.
        """
        BS, P = u.shape
        Dn = self.dir_grid.shape[0]
        u_e = u[:, :, None].expand(BS, P, Dn).reshape(BS, P * Dn)
        v_e = v[:, :, None].expand(BS, P, Dn).reshape(BS, P * Dn)
        d_e = self.dir_grid[None, None].expand(BS, P, Dn, 3).reshape(BS, P * Dn, 3)
        use_ckpt = self.training and torch.is_grad_enabled()
        outs = []
        for s in range(0, P * Dn, chunk):
            e = min(s + chunk, P * Dn)
            if use_ckpt:
                rad = checkpoint(self._light_radiance, mem, u_e[:, s:e], v_e[:, s:e],
                                 d_e[:, s:e], use_reentrant=False)
            else:
                rad = self._light_radiance(mem, u_e[:, s:e], v_e[:, s:e], d_e[:, s:e])
            outs.append(rad)
        return torch.cat(outs, dim=1).reshape(BS, P, Dn, 3)

    def _render_inputs(self, mem, images_bs, B, S):
        """Sample P pixels; return their full env tiles + per-pixel material + uv.

        Returns (env [B,S,P,Dn,3], uv [B,S,P,2] in [0,1], material dict name->[B,S,P,c]).
        """
        BS = B * S
        P = self.num_render_pixels
        Hs, Ws = self.light_spatial_h, self.light_spatial_w
        dev = mem.device
        spatial_idx = torch.randint(0, Hs * Ws, (BS, P), device=dev)
        sh = (spatial_idx // Ws).float(); sw = (spatial_idx % Ws).float()
        u = (sw + 0.5) / Ws; v = (sh + 0.5) / Hs                      # [BS,P]
        env = self._light_env_tiles(mem, u, v)                       # [BS,P,Dn,3]
        mat = self._material_at(mem, images_bs, u, v, B, S)          # name->[B,S,P,c]
        uv = torch.stack([u, v], dim=-1)                             # [BS,P,2]
        return env.reshape(B, S, P, -1, 3), uv.reshape(B, S, P, 2), mat

    @torch.no_grad()
    def predict_material_dense(self, tokens_list, images, patch_start_idx, chunk=16384):
        """Full per-pixel material maps for demo / inference -> dict name->[B,S,H,W,c].

        Queries EVERY pixel (chunked) so the maps show the true per-pixel detail the
        model was trained for (vs the patch-grid+bilinear `_material_dense`).
        """
        B, S, _, H, W = images.shape
        mem, _, _, _ = self._build_memory(tokens_list, patch_start_idx)
        images_bs = images.reshape(B * S, 3, H, W)
        dev = mem.device
        us = (torch.arange(W, device=dev).float() + 0.5) / W
        vs = (torch.arange(H, device=dev).float() + 0.5) / H
        vv, uu = torch.meshgrid(vs, us, indexing="ij")
        u_all = uu.reshape(1, -1).expand(B * S, -1)      # [BS, H*W]
        v_all = vv.reshape(1, -1).expand(B * S, -1)
        HW = H * W
        acc = {name: [] for name in self.head_names}
        for s in range(0, HW, chunk):
            e = min(s + chunk, HW)
            raw = self._material_features(mem, images_bs, u_all[:, s:e], v_all[:, s:e])
            for name, r in raw.items():
                acc[name].append(self._apply_material_activation(name, r))
        out = {}
        for name in self.head_names:
            cat = torch.cat(acc[name], dim=1)            # [BS, HW, c]
            out[name] = cat.reshape(B, S, H, W, cat.shape[-1])
        return out

    @torch.no_grad()
    def render_demo(self, tokens_list, images, patch_start_idx, material=None,
                    point_map=None, camera_pos=None):
        """Smart full-res render: full-res material × lighting computed on a coarse
        (H/f, W/f) grid then upsampled by f = self.render_demo_upsample.

        Diffuse = full-res albedo·(1-m)/π × upsample(coarse irradiance)  → material-sharp.
        Specular (if point_map/camera_pos given) is evaluated on the coarse grid and
        upsampled (lighting is low-freq, so this is cheap and visually fine).
        `material`: optional precomputed full-res maps {name:[B,S,H,W,c]} (else queried).
        Returns sRGB [B,S,H,W,3].
        """
        from vggt.heads.per_pixel_renderer import render_pixels
        B, S, _, H, W = images.shape
        mem, _, _, _ = self._build_memory(tokens_list, patch_start_idx)
        BS = B * S
        f = max(1, int(self.render_demo_upsample))
        Hl, Wl = max(1, H // f), max(1, W // f)        # coarse lighting grid
        dev = mem.device

        # full-res material (reuse precomputed maps if given)
        if material is not None:
            mat = {n: material[n].reshape(BS, H, W, -1) for n in self.head_names if n in material}
        else:
            md = self.predict_material_dense(tokens_list, images, patch_start_idx)
            mat = {n: md[n].reshape(BS, H, W, -1) for n in self.head_names}
        alb, nrm, rgh, met = mat["albedo"], mat["normal"], mat["roughness"], mat["metallic"]  # [BS,H,W,c]

        # coarse lighting grid uv
        sh = (torch.arange(Hl, device=dev).float() + 0.5) / Hl
        sw = (torch.arange(Wl, device=dev).float() + 0.5) / Wl
        vv, uu = torch.meshgrid(sh, sw, indexing="ij")
        u = uu.reshape(1, -1).expand(BS, -1)           # [BS, Hl*Wl]
        v = vv.reshape(1, -1).expand(BS, -1)
        env = self._light_env_tiles(mem, u, v, chunk=16384)   # [BS, Hl*Wl, Dn, 3]
        Dn = env.shape[2]

        # diffuse irradiance (material-free) -> upsample to full res
        wgt = (self.dir_grid[:, 1].clamp(min=0.0) * self.solid_angle)[None, None, :, None]  # [1,1,Dn,1]
        E = (env * wgt).sum(dim=2).reshape(BS, Hl, Wl, 3).permute(0, 3, 1, 2)               # [BS,3,Hl,Wl]
        E = F.interpolate(E, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        rendered = alb * (1.0 - met) / math.pi * E     # [BS,H,W,3] full-res material × upsampled light

        # specular: evaluate coarse (coarse material + view), upsample, add
        if point_map is not None and camera_pos is not None:
            def _coarse(x):
                xc = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2).float(), (Hl, Wl))
                return xc.permute(0, 2, 3, 1).reshape(BS, Hl * Wl, -1)
            alb_c = _coarse(alb); nrm_c = F.normalize(_coarse(nrm), dim=-1, eps=1e-8)
            rgh_c = _coarse(rgh); met_c = _coarse(met)
            pm = point_map.reshape(BS, point_map.shape[2], point_map.shape[3], 3).permute(0, 3, 1, 2).float()
            grid = (torch.stack([u, v], -1).reshape(BS, Hl * Wl, 1, 2) * 2 - 1).float()
            pos = F.grid_sample(pm, grid, mode="bilinear", align_corners=True,
                                padding_mode="border").squeeze(-1).permute(0, 2, 1)
            cam = camera_pos.reshape(BS, 3)[:, None, :]
            view = F.normalize(cam - pos, dim=-1, eps=1e-8)
            Nc = BS * Hl * Wl
            _, spec = render_pixels(
                alb_c.reshape(Nc, 3), nrm_c.reshape(Nc, 3), rgh_c.reshape(Nc, 1), met_c.reshape(Nc, 1),
                env.reshape(Nc, Dn, 3), self.dir_grid, self.solid_angle,
                view_dir=view.reshape(Nc, 3), return_parts=True)
            spec = spec.reshape(BS, Hl, Wl, 3).permute(0, 3, 1, 2)
            spec = F.interpolate(spec, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            rendered = rendered + spec

        rendered = rendered / (rendered + 1.0)
        rendered = rendered.clamp(min=1e-5).pow(1.0 / 2.2)
        return rendered.reshape(B, S, H, W, 3)

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
            if self.num_material_samples > 0:
                # d4rt-style per-pixel: random pixel coords at BOTH train and eval, so
                # the val loss stays per-pixel + cheap (no dense 37x37 grid). Full dense
                # maps for the demo / inference come from predict_material_dense().
                mat, muv = self._material_sampled(mem, images_bs, B, S)
                preds.update(mat)                  # name -> [B,S,N,c]
                preds["material_uv"] = muv         # [B,S,N,2]
            else:
                # num_material_samples=0: dense patch-grid + bilinear upsample (old path).
                preds.update(self._material_dense(mem, images_bs, B, S))

        if self.enable_lighting:
            rad, spix, didx = self._light_branch_sampled(mem, B, S)
            preds["light_pred"] = rad           # [B,S,M,3]
            preds["light_spatial_idx"] = spix   # [B,S,M]
            preds["light_dir_idx"] = didx       # [B,S,M]
            # Training: per-pixel env tiles + material at the same pixels for the
            # BRDF render loss (needs material; full Dn dirs/pixel).
            if self.enable_render and self.enable_material and self.training:
                renv, ruv, rmat = self._render_inputs(mem, images_bs, B, S)
                preds["render_env"] = renv                      # [B,S,P,Dn,3]
                preds["render_uv"] = ruv                        # [B,S,P,2]
                preds["render_dir_grid"] = self.dir_grid        # [Dn,3]
                preds["render_solid_angle"] = self.solid_angle  # [Dn]
                for _k, _v in rmat.items():
                    preds[f"render_{_k}"] = _v                  # render_albedo/normal/...
            # Eval: also emit a coarse dense env for visualization (cheap grid).
            if not self.training:
                gh = min(self.light_spatial_h, 16)
                gw = min(self.light_spatial_w, 16)
                preds["pred_env_pixel"] = self.predict_env_dense(
                    tokens_list, patch_start_idx, gh, gw)   # [B,S,gh,gw,env_h,env_w,3]

        # Expose the dynamic-weighting log-variance. Add a zero "ghost" onto one live
        # prediction so every task_log_var entry has a (zero) gradient — keeps DDP from
        # flagging it unused and avoids "marked ready twice" with the loss-side grad.
        if self.enable_dynamic_weighting and self.task_log_var is not None:
            ghost = self.task_log_var.sum() * 0.0
            for _cand in ("albedo", "light_pred", "normal", "render_env"):
                if _cand in preds:
                    preds[_cand] = preds[_cand] + ghost
                    break
            preds["task_log_var"] = self.task_log_var
            preds["task_log_var_names"] = self.dyn_task_names
        return preds

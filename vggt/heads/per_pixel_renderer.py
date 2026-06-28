# Per-pixel Cook-Torrance renderer for the d4rt per-pixel HDR env.
#
# Unlike brdf_renderer.py (which integrates a global SG over the sphere), this
# renders ONE pixel at a time by summing Cook-Torrance over that pixel's own
# incident environment (the 8x16 imenvlow tile), which is expressed in the
# pixel's NORMAL-LOCAL hemisphere (pole = +y = surface normal).
#
#     L_out(p) = sum_{omega_i}  f_BRDF(omega_i, omega_o) * L_env(p, omega_i)
#                               * (n . omega_i) * dOmega_i
#
# Diffuse term is frame-free (the env is already normal-local, so n.omega_i is
# just the elevation cosine). Specular needs the view direction omega_o, which
# is transformed into the same normal-local frame via a deterministic tangent
# basis (azimuth convention is assumed to match imenvlow's; diffuse is azimuth-
# invariant so it is unaffected). Assumes the predicted normal and the
# geometry-derived view dir live in the same (world) frame, matching
# brdf_renderer.py.

import math
import torch
import torch.nn.functional as F

from vggt.heads.brdf_renderer import ggx_ndf, smith_geometry, fresnel_schlick


def hemisphere_dirs_and_solid_angle(env_h: int, env_w: int, device=None):
    """Local-hemisphere unit dirs (pole=+y) + per-cell solid angle.

    MUST match D4RTInverseHeads._hemisphere_dirs ordering: theta over rows
    (elevation from +y pole), phi over cols, flattened row-major.
    Returns (dirs [env_h*env_w, 3], solid_angle [env_h*env_w]).
    """
    theta = (torch.arange(env_h, device=device).float() + 0.5) / env_h * (math.pi / 2)  # [env_h]
    phi = (torch.arange(env_w, device=device).float() + 0.5) / env_w * (2 * math.pi)     # [env_w]
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    x = torch.sin(th) * torch.cos(ph)
    y = torch.cos(th)
    z = torch.sin(th) * torch.sin(ph)
    dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)            # [env_h*env_w, 3]
    dtheta = (math.pi / 2) / env_h
    dphi = (2 * math.pi) / env_w
    sa = (torch.sin(th) * dtheta * dphi).reshape(-1)               # [env_h*env_w]
    return dirs, sa


def _build_tangent_frame(n: torch.Tensor):
    """Orthonormal (t, b) tangents for world normals n [N,3]. Returns t,b [N,3]."""
    ref = torch.zeros_like(n)
    ref[..., 2] = 1.0                                              # prefer +z
    # avoid degeneracy when n ~ +/-z: fall back to +x
    deg = (n[..., 2].abs() > 0.99).unsqueeze(-1)
    ref_alt = torch.zeros_like(n); ref_alt[..., 0] = 1.0
    ref = torch.where(deg, ref_alt, ref)
    t = ref - (ref * n).sum(-1, keepdim=True) * n
    t = F.normalize(t, dim=-1, eps=1e-8)
    b = torch.cross(n, t, dim=-1)
    return t, b


def render_pixels(
    albedo: torch.Tensor,      # [N, 3]   in [0,1]
    normal: torch.Tensor,      # [N, 3]   world, unit
    roughness: torch.Tensor,   # [N, 1]
    metallic: torch.Tensor,    # [N, 1]
    env: torch.Tensor,         # [N, D, 3] incident HDR radiance per local dir
    dir_grid: torch.Tensor,    # [D, 3]   local hemisphere dirs (pole=+y)
    solid_angle: torch.Tensor, # [D]
    view_dir: torch.Tensor = None,  # [N, 3] world view dir (cam - pos), unit; None -> diffuse only
    tonemap_gamma: bool = True,
    return_parts: bool = False,     # if True, return (diffuse, specular) LINEAR (pre-tonemap)
) -> torch.Tensor:
    """Cook-Torrance render of N pixels from their per-pixel env. Returns [N, 3]
    (or (diffuse[N,3], specular[N,3]) linear if return_parts)."""
    N = albedo.shape[0]
    D = dir_grid.shape[0]
    dg = dir_grid.to(albedo.dtype)
    sa = solid_angle.to(albedo.dtype)
    nl = dg[:, 1].clamp(min=0.0)                       # n.omega_i in local frame = elevation cos [D]
    w = (nl * sa)[None, :, None]                       # [1, D, 1] cos * dOmega

    # --- Diffuse (frame-free) ---
    diffuse_albedo = albedo * (1.0 - metallic)         # [N,3]
    irradiance = (env * w).sum(dim=1)                  # [N,3]  sum_i L * cos * dOmega
    diffuse = diffuse_albedo / math.pi * irradiance    # [N,3]

    specular = torch.zeros_like(diffuse)

    # --- Specular (needs view dir in the local frame) ---
    if view_dir is not None:
        n = F.normalize(normal, dim=-1, eps=1e-8)
        t, b = _build_tangent_frame(n)
        # view in local axes (x=t, y=n, z=b)
        vx = (view_dir * t).sum(-1, keepdim=True)
        vy = (view_dir * n).sum(-1, keepdim=True)
        vz = (view_dir * b).sum(-1, keepdim=True)
        v_local = torch.cat([vx, vy, vz], dim=-1)      # [N,3]

        rough = roughness.clamp(0.04, 1.0)
        alpha = (rough * rough)                        # [N,1]
        f0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)  # [N,3]

        wi = dg[None, :, :]                            # [1,D,3]
        vloc = v_local[:, None, :]                     # [N,1,3]
        h = F.normalize(wi + vloc, dim=-1, eps=1e-8)   # [N,D,3]
        ndoth = h[..., 1:2].clamp(min=0.0)             # [N,D,1] (n=+y local)
        ndotv = vy[:, None, :].clamp(min=1e-4)         # [N,1,1]
        ndotl = nl[None, :, None]                      # [1,D,1]
        ldoth = (wi * h).sum(-1, keepdim=True).clamp(min=0.0)  # [N,D,1]

        a = alpha[:, None, :]                          # [N,1,1]
        Dggx = ggx_ndf(ndoth, a)                       # [N,D,1]
        G = smith_geometry(ndotl.expand(N, D, 1).squeeze(-1),
                           ndotv.expand(N, D, 1).squeeze(-1),
                           a.expand(N, D, 1).squeeze(-1)).unsqueeze(-1)  # [N,D,1]
        Fr = fresnel_schlick(ldoth, f0[:, None, :])    # [N,D,3]
        denom = 4.0 * ndotl * ndotv + 1e-8             # [N,D,1]
        spec_brdf = Dggx * G * Fr / denom              # [N,D,3]
        specular = (spec_brdf * env * w).sum(dim=1)    # [N,3]

    if return_parts:
        return diffuse, specular                       # linear, pre-tonemap

    rendered = diffuse + specular
    if tonemap_gamma:
        rendered = rendered / (rendered + 1.0)                 # Reinhard -> [0,1)
        rendered = rendered.clamp(min=1e-5).pow(1.0 / 2.2)     # sRGB gamma (match brdf_renderer)
    return rendered

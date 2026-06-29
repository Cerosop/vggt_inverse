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
# just the elevation cosine). Specular reconstructs each env cell's direction in
# the CAMERA frame using OpenRooms' own per-pixel tangent convention
# (InverseRenderingOfIndoorScene renderingLayer):
#     up   = (0,1,0)  (camera up)
#     camy = normalize(up - (up.n) n)        # bitangent, local y (Az=90)
#     camx = -normalize(cross(camy, n))      # tangent,  local x (Az=0)
#     omega = lx*camx + ly*camy + lz*n,  lx=sinEl cosAz, ly=sinEl sinAz, lz=cosEl
# so the predicted env (cell-matched to the imenvlow GT) is read with the SAME
# azimuth zero the GT was authored in. REQUIRES: `normal` and `view_dir` are both
# in OpenRooms' y-UP camera frame (x-right, y-up, z-back). The predicted normal is
# already y-up (trained on OpenRooms normal.png); callers rotate the world view into
# this frame via R_or = C @ R_vggt, C=diag(1,-1,-1) the OpenCV(y-down)->y-up flip
# (see sg_loss / trainer). RESIDUAL(unverified): the z-back handedness — validate
# specular against a GT (env,image) sample on a glossy surface before trusting it.

import math
import torch
import torch.nn.functional as F

from vggt.heads.brdf_renderer import ggx_ndf, smith_geometry, fresnel_schlick


def hemisphere_dirs_and_solid_angle(env_h: int, env_w: int, device=None):
    """Local-hemisphere unit dirs (pole=+y) + per-cell solid angle, OpenRooms azimuth.

    MUST match D4RTInverseHeads._hemisphere_dirs: El over rows (elevation from the
    +y pole), Az over cols with OpenRooms' convention Az=((j+0.5)/env_w-0.5)*2pi
    in [-pi,pi], flattened row-major.
    Returns (dirs [env_h*env_w, 3], solid_angle [env_h*env_w]).
    """
    theta = (torch.arange(env_h, device=device).float() + 0.5) / env_h * (math.pi / 2)  # El [env_h]
    phi = (((torch.arange(env_w, device=device).float() + 0.5) / env_w) - 0.5) * (2 * math.pi)  # Az [-pi,pi]
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    x = torch.sin(th) * torch.cos(ph)                              # along camx (Az=0)
    y = torch.cos(th)                                              # pole = normal
    z = torch.sin(th) * torch.sin(ph)                              # along camy (Az=90)
    dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)            # [env_h*env_w, 3]
    dtheta = (math.pi / 2) / env_h
    dphi = (2 * math.pi) / env_w
    sa = (torch.sin(th) * dtheta * dphi).reshape(-1)               # [env_h*env_w]
    return dirs, sa


def _openrooms_tangent(n: torch.Tensor):
    """OpenRooms per-pixel env tangent frame (camera frame). n:[N,3] camera-frame normal.

    Matches InverseRenderingOfIndoorScene renderingLayer:
        up   = (0,1,0)                    # camera up
        camy = normalize(up - (up.n) n)   # bitangent (local y, Az=90)
        camx = -normalize(cross(camy, n)) # tangent   (local x, Az=0)
    Returns (camx, camy). Guards the n||up singularity (OpenRooms does not) by
    switching the reference to (0,0,1) there, so those pixels don't NaN (their
    azimuth is arbitrary at the pole anyway).
    """
    up = torch.zeros_like(n); up[..., 1] = 1.0
    degen = (n[..., 1].abs() > 0.99).unsqueeze(-1)
    up_alt = torch.zeros_like(n); up_alt[..., 2] = 1.0
    up = torch.where(degen, up_alt, up)
    camy = up - (up * n).sum(-1, keepdim=True) * n
    camy = F.normalize(camy, dim=-1, eps=1e-8)
    camx = -F.normalize(torch.cross(camy, n, dim=-1), dim=-1, eps=1e-8)
    return camx, camy


def render_pixels(
    albedo: torch.Tensor,      # [N, 3]   in [0,1]
    normal: torch.Tensor,      # [N, 3]   CAMERA-frame, unit
    roughness: torch.Tensor,   # [N, 1]
    metallic: torch.Tensor,    # [N, 1]
    env: torch.Tensor,         # [N, D, 3] incident HDR radiance per local dir
    dir_grid: torch.Tensor,    # [D, 3]   local hemisphere dirs (pole=+y, OpenRooms Az)
    solid_angle: torch.Tensor, # [D]
    view_dir: torch.Tensor = None,  # [N, 3] CAMERA-frame view dir (unit); None -> diffuse only
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

    # --- Specular: reconstruct env-cell directions in the CAMERA frame (OpenRooms) ---
    if view_dir is not None:
        n = F.normalize(normal, dim=-1, eps=1e-8)          # [N,3] camera-frame normal
        v = F.normalize(view_dir, dim=-1, eps=1e-8)        # [N,3] camera-frame view (surface->cam)
        camx, camy = _openrooms_tangent(n)                 # [N,3] each

        # cell direction omega = dg.x*camx + dg.y*n + dg.z*camy   (dg = sinEl cosAz, cosEl, sinEl sinAz)
        # -> matches OpenRooms l = lx*camx + ly*camy + lz*n with the GT's azimuth zero.
        omega = (dg[None, :, 0:1] * camx[:, None, :]
                 + dg[None, :, 1:2] * n[:, None, :]
                 + dg[None, :, 2:3] * camy[:, None, :])     # [N,D,3] camera frame

        rough = roughness.clamp(0.04, 1.0)
        alpha = (rough * rough)                            # [N,1]
        f0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, metallic)  # [N,3]

        ndotv = (v * n).sum(-1, keepdim=True).clamp(min=1e-4)  # [N,1] = N·V

        # Energy conservation: the fraction reflected specularly (Fresnel at N·V) is
        # removed from the diffuse budget. kd = (1 - F(N·V)); combined with the
        # (1-metallic) already in diffuse_albedo, diffuse weight = (1-metallic)(1-F).
        # (IBL split-sum style: a single Fresnel at the view angle, not per incident dir.)
        F_ndotv = fresnel_schlick(ndotv.clamp(0.0, 1.0), f0)  # [N,3]
        diffuse = diffuse * (1.0 - F_ndotv)

        h = F.normalize(omega + v[:, None, :], dim=-1, eps=1e-8)        # [N,D,3]
        ndoth = (h * n[:, None, :]).sum(-1, keepdim=True).clamp(min=0.0)  # [N,D,1]
        ndotv_d = ndotv[:, None, :]                        # [N,1,1]
        ndotl = nl[None, :, None]                          # [1,D,1] (= cosEl = n·omega)
        ldoth = (omega * h).sum(-1, keepdim=True).clamp(min=0.0)  # [N,D,1]

        a = alpha[:, None, :]                              # [N,1,1]
        Dggx = ggx_ndf(ndoth, a)                           # [N,D,1]
        G = smith_geometry(ndotl.expand(N, D, 1).squeeze(-1),
                           ndotv_d.expand(N, D, 1).squeeze(-1),
                           a.expand(N, D, 1).squeeze(-1)).unsqueeze(-1)  # [N,D,1]
        Fr = fresnel_schlick(ldoth, f0[:, None, :])        # [N,D,3]
        denom = 4.0 * ndotl * ndotv_d + 1e-8               # [N,D,1]
        spec_brdf = Dggx * G * Fr / denom                  # [N,D,3]
        specular = (spec_brdf * env * w).sum(dim=1)        # [N,3]

    if return_parts:
        return diffuse, specular                       # linear, pre-tonemap

    rendered = diffuse + specular
    if tonemap_gamma:
        rendered = rendered / (rendered + 1.0)                 # Reinhard -> [0,1)
        rendered = rendered.clamp(min=1e-5).pow(1.0 / 2.2)     # sRGB gamma (match brdf_renderer)
    return rendered

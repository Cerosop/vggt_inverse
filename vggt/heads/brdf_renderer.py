# Cook-Torrance GGX Microfacet BRDF Renderer with SG Lighting.
#
# Differentiable renderer that combines:
# - Spherical Gaussian (SG) environment lighting
# - GGX NDF (Normal Distribution Function)
# - Smith geometry term
# - Fresnel-Schlick approximation
#
# Reference:
#   Walter et al., "Microfacet Models for Refraction", EGSR 2007
#   Wang et al., "All-Frequency Rendering of Dynamic, Spatially-Varying BRDFs", SIGGRAPH 2009

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BRDFRenderer(nn.Module):
    """Cook-Torrance GGX Microfacet BRDF renderer with SG lighting.

    Renders images given material properties, geometry, and SG lighting parameters.
    All computations are differentiable.

    The renderer evaluates the rendering equation per-pixel:
        L_o = ∫ f(ω_i, ω_o) * L_i(ω_i) * (n · ω_i) dω_i

    where f is the Cook-Torrance BRDF and L_i is represented as a sum of SG lobes.

    Args:
        render_downsample: Downsample factor for rendering (for VRAM efficiency).
            E.g., 2 means render at half resolution then upsample.
    """

    def __init__(self, render_downsample: int = 1):
        super().__init__()
        self.render_downsample = render_downsample

    def forward(
        self,
        albedo: torch.Tensor,
        normal: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        sg_params: torch.Tensor,
        point_map: torch.Tensor,
        camera_pos: torch.Tensor,
        shading: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render images using Cook-Torrance GGX BRDF with SG lighting.

        Args:
            albedo:    [B, S, H, W, 3]  base color in [0, 1]
            normal:    [B, S, H, W, 3]  unit surface normals
            roughness: [B, S, H, W, 1]  roughness in [0, 1]
            metallic:  [B, S, H, W, 1]  metallic in [0, 1]
            sg_params: [B, S, num_lobes, 7] SG parameters per-frame (dir3, sharp1, amp3)
            point_map: [B, S, H, W, 3]  3D world positions
            camera_pos:[B, S, 3]         camera positions in world space
            shading:   [B, S, H, W, 3]  optional shadow/AO modulation

        Returns:
            rendered: [B, S, H, W, 3] rendered image in [0, ∞) (HDR, needs tone-mapping for display)
        """
        B, S, H, W, _ = albedo.shape

        # Optionally downsample for VRAM efficiency
        ds = self.render_downsample
        if ds > 1:
            albedo_r = self._downsample(albedo, ds)
            normal_r = self._downsample(normal, ds)
            roughness_r = self._downsample(roughness, ds)
            metallic_r = self._downsample(metallic, ds)
            point_map_r = self._downsample(point_map, ds)
            if shading is not None:
                shading_r = self._downsample(shading, ds)
            else:
                shading_r = None
        else:
            albedo_r = albedo
            normal_r = normal
            roughness_r = roughness
            metallic_r = metallic
            point_map_r = point_map
            shading_r = shading

        _, _, Hr, Wr, _ = albedo_r.shape

        # Compute view direction: normalize(camera_pos - point_map)
        cam_pos_expanded = camera_pos.unsqueeze(2).unsqueeze(3)  # [B, S, 1, 1, 3]
        view_dir = cam_pos_expanded - point_map_r  # [B, S, Hr, Wr, 3]
        view_dir = F.normalize(view_dir, p=2, dim=-1, eps=1e-8)

        # Ensure normal is unit-length
        normal_r = F.normalize(normal_r, p=2, dim=-1, eps=1e-8)

        # Clamp roughness to avoid division issues
        roughness_clamped = torch.clamp(roughness_r, min=0.04, max=1.0)
        alpha = roughness_clamped ** 2  # alpha = roughness^2 for GGX

        # Fresnel at normal incidence: F0
        # For dielectrics: F0 ≈ 0.04, for metals: F0 = albedo
        f0_dielectric = torch.full_like(albedo_r, 0.04)
        f0 = torch.lerp(f0_dielectric, albedo_r, metallic_r)  # [B, S, Hr, Wr, 3]

        # Diffuse albedo (metals have no diffuse)
        diffuse_albedo = albedo_r * (1.0 - metallic_r)  # [B, S, Hr, Wr, 3]

        # Split SG parameters
        sg_dir = sg_params[:, :, :, :3]        # [B, S, num_lobes, 3]
        sg_sharp = sg_params[:, :, :, 3:4]     # [B, S, num_lobes, 1]
        sg_amp = sg_params[:, :, :, 4:7]       # [B, S, num_lobes, 3]

        num_lobes = sg_params.shape[2]

        # Accumulate lighting over all SG lobes
        diffuse_accum = torch.zeros_like(albedo_r)   # [B, S, Hr, Wr, 3]
        specular_accum = torch.zeros_like(albedo_r)  # [B, S, Hr, Wr, 3]

        for lobe_idx in range(num_lobes):
            lobe_dir = sg_dir[:, :, lobe_idx, :]      # [B, S, 3]
            lobe_sharp = sg_sharp[:, :, lobe_idx, :]   # [B, S, 1]
            lobe_amp = sg_amp[:, :, lobe_idx, :]       # [B, S, 3]

            # Expand lobe params to spatial dims [B, S, 1, 1, 3/1]
            l_dir = lobe_dir[:, :, None, None, :]       # [B, S, 1, 1, 3]
            l_sharp = lobe_sharp[:, :, None, None, :]   # [B, S, 1, 1, 1]
            l_amp = lobe_amp[:, :, None, None, :]       # [B, S, 1, 1, 3]

            # light_dir is the direction TO the light
            light_dir = l_dir.expand(-1, -1, Hr, Wr, -1)  # [B, S, Hr, Wr, 3]

            # n · l
            ndotl = (normal_r * light_dir).sum(dim=-1, keepdim=True)  # [B, S, Hr, Wr, 1]
            ndotl = torch.clamp(ndotl, min=0.0)

            # SG value along the surface/lobe alignment (used for the specular term).
            sg_eval = l_amp * torch.exp(l_sharp * (ndotl - 1.0))  # [B, S, Hr, Wr, 3]

            # Diffuse: treat each SG lobe as a directional light whose intensity is the
            # lobe's TOTAL flux  ∫ G dω = amp * 2π * (1 - e^{-2λ}) / λ , then Lambertian
            # shade it (the 1/π is applied once below). The previous code used the SG
            # *value* (no solid-angle integral), which under-estimated irradiance for
            # broad lobes and made renders ~10x too dark.
            lobe_flux = l_amp * (2.0 * math.pi * (1.0 - torch.exp(-2.0 * l_sharp)) / l_sharp)  # [B,S,1,1,3]
            diffuse_contrib = lobe_flux * ndotl  # [B, S, Hr, Wr, 3]

            # --- Specular: Simplified SG × GGX ---
            # Half vector
            h = F.normalize(light_dir + view_dir, p=2, dim=-1, eps=1e-8)

            ndoth = (normal_r * h).sum(dim=-1, keepdim=True).clamp(min=0.0)
            ndotv = (normal_r * view_dir).sum(dim=-1, keepdim=True).clamp(min=1e-4)
            ldoth = (light_dir * h).sum(dim=-1, keepdim=True).clamp(min=0.0)

            # GGX NDF
            D = ggx_ndf(ndoth, alpha)  # [B, S, Hr, Wr, 1]

            # Smith Geometry
            G = smith_geometry(ndotl.squeeze(-1), ndotv.squeeze(-1), alpha.squeeze(-1))  # [B, S, Hr, Wr]
            G = G.unsqueeze(-1)  # [B, S, Hr, Wr, 1]

            # Fresnel-Schlick
            F_term = fresnel_schlick(ldoth, f0)  # [B, S, Hr, Wr, 3]

            # Cook-Torrance specular = D * G * F / (4 * ndotl * ndotv)
            denom = 4.0 * ndotl * ndotv + 1e-8
            specular_brdf = D * G * F_term / denom  # [B, S, Hr, Wr, 3]

            specular_contrib = specular_brdf * sg_eval * ndotl

            diffuse_accum = diffuse_accum + diffuse_contrib
            specular_accum = specular_accum + specular_contrib

        # Final rendering equation
        diffuse_term = diffuse_albedo / math.pi * diffuse_accum
        specular_term = specular_accum

        rendered = diffuse_term + specular_term  # [B, S, Hr, Wr, 3]

        # Apply optional shading (shadow/AO modulation)
        if shading_r is not None:
            rendered = rendered * shading_r

        # Simple Reinhard tone mapping of the (linear) radiance to [0, 1].
        rendered = rendered / (rendered + 1.0)

        # sRGB gamma encode. The GT images are sRGB-encoded but the render is linear
        # radiance; comparing/displaying linear vs sRGB makes the render ~2x too dark
        # (verified: linear mean 0.22 vs input 0.43; gamma-encoded 0.47 ≈ input).
        # clamp(min) bounds the gamma gradient on near-black pixels.
        rendered = rendered.clamp(min=1e-5).pow(1.0 / 2.2)

        # Upsample back to original resolution if downsampled
        if ds > 1:
            rendered = self._upsample(rendered, H, W)

        return rendered

    @staticmethod
    def _downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
        """Downsample [B, S, H, W, C] spatial dims by factor."""
        B, S, H, W, C = x.shape
        x_flat = x.reshape(B * S, H, W, C).permute(0, 3, 1, 2)  # [BS, C, H, W]
        x_down = F.interpolate(
            x_flat,
            size=(H // factor, W // factor),
            mode="bilinear",
            align_corners=False,
        )
        _, _, Hd, Wd = x_down.shape
        return x_down.permute(0, 2, 3, 1).reshape(B, S, Hd, Wd, C)

    @staticmethod
    def _upsample(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Upsample [B, S, h, w, C] back to [B, S, H, W, C]."""
        B, S, h, w, C = x.shape
        x_flat = x.reshape(B * S, h, w, C).permute(0, 3, 1, 2)
        x_up = F.interpolate(x_flat, size=(H, W), mode="bilinear", align_corners=False)
        return x_up.permute(0, 2, 3, 1).reshape(B, S, H, W, C)


# ============================================================================
# BRDF Helper Functions
# ============================================================================


def ggx_ndf(ndoth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """GGX/Trowbridge-Reitz Normal Distribution Function.

    D(h) = α² / (π * ((n·h)² * (α² - 1) + 1)²)

    Args:
        ndoth: [B, S, H, W, 1] dot(normal, half_vector), clamped to [0, 1]
        alpha: [B, S, H, W, 1] roughness² parameter

    Returns:
        D: [B, S, H, W, 1]
    """
    a2 = alpha * alpha
    ndoth2 = ndoth * ndoth
    denom = ndoth2 * (a2 - 1.0) + 1.0
    denom = denom * denom * math.pi
    return a2 / (denom + 1e-8)


def smith_geometry(ndotl: torch.Tensor, ndotv: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Smith's Schlick-GGX Geometry Function (combined G1 for light and view).

    G(n,v,l) = G1(n,v) * G1(n,l)
    G1(n,x) = n·x / (n·x * (1 - k) + k)

    `alpha` here is α = roughness² (the GGX width). We use the IBL / environment-
    lighting remap k = α / 2 = roughness²/2 (Karis 2013), since our lighting is the
    per-pixel environment (NOT an analytic punctual light, whose remap would be
    k = (roughness+1)²/8). Previously this computed k = α²/2 = roughness⁴/2, which
    over-squared α and gave too-weak geometric shadowing (specular too bright).

    Args:
        ndotl: [B, S, H, W] clamped to [0, 1]
        ndotv: [B, S, H, W] clamped to [1e-4, 1]
        alpha: [B, S, H, W] roughness²

    Returns:
        G: [B, S, H, W]
    """
    k = alpha / 2.0                                    # IBL: k = α/2 = roughness²/2
    g1_v = ndotv / (ndotv * (1.0 - k) + k + 1e-8)
    g1_l = ndotl / (ndotl * (1.0 - k) + k + 1e-8)
    return g1_v * g1_l


def fresnel_schlick(
    ldoth: torch.Tensor,
    f0: torch.Tensor,
) -> torch.Tensor:
    """Fresnel-Schlick approximation.

    F(θ) = F0 + (1 - F0) * (1 - cos θ)^5

    Args:
        ldoth: [B, S, H, W, 1] dot(light_dir, half_vector), clamped [0, 1]
        f0:    [B, S, H, W, 3] reflectance at normal incidence

    Returns:
        F: [B, S, H, W, 3]
    """
    return f0 + (1.0 - f0) * (1.0 - ldoth).clamp(min=0.0).pow(5)


def compute_camera_positions(pose_enc: torch.Tensor) -> torch.Tensor:
    """Extract camera positions from VGGT pose encoding.

    VGGT pose_enc format (absT_quaR_FoV): first 3 dims are absolute translation.

    Args:
        pose_enc: [B, S, 9] — [tx, ty, tz, qw, qx, qy, qz, fov_x, fov_y]

    Returns:
        camera_pos: [B, S, 3] camera position in world space
    """
    return pose_enc[..., :3]

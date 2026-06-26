# SG (Spherical Gaussian) Loss Functions for VGGT.
#
# Includes:
# - Hungarian Matching Loss: optimal assignment between predicted and GT SG lobes
# - Repulsion Loss: prevents lobes from collapsing to the same direction
# - BRDF Render Loss: MSE between rendered and input images

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger(__name__)


class SGLoss(nn.Module):
    """Loss functions for Spherical Gaussian lighting estimation.

    Args:
        weight_brdf_render: Weight for BRDF render loss.
        weight_repulsion: Weight for lobe repulsion loss.
        repulsion_threshold: Cosine similarity threshold for repulsion.
        weight_hungarian: Weight for Hungarian matching loss (when GT SG available).
        enable_hungarian: Whether to enable Hungarian matching loss.
        render_downsample: Downsample factor for BRDF rendering (VRAM saving).
    """

    def __init__(
        self,
        weight_brdf_render: float = 1.0,
        weight_repulsion: float = 0.01,
        repulsion_threshold: float = 0.9,
        weight_hungarian: float = 1.0,
        enable_hungarian: bool = False,
        render_downsample: int = 2,
        sg_phase1_end: float = 0.2,
        sg_phase2_end: float = 0.4,
        weight_sg_l2: float = 1.0,
        enable_sg_l2: bool = True,
        enable_env_map_log_l2: bool = True,
        weight_env_map_log_l2: float = 1.0,
        enable_diffuse_constraint: bool = False,
        weight_diffuse_constraint: float = 0.5,
        geometry_source: str = "pred",
    ):
        super().__init__()
        # BRDF render geometry source: "pred" (model camera/point head predictions)
        # or "gt" (dataset-provided camera_pos / gt_point_map).
        self.geometry_source = geometry_source
        self.weight_brdf_render = weight_brdf_render
        self.weight_repulsion = weight_repulsion
        self.repulsion_threshold = repulsion_threshold
        # Hungarian is subsumed by L2 phase GT loss logic
        self.enable_hungarian = enable_hungarian
        self.weight_hungarian = weight_hungarian
        self.render_downsample = render_downsample
        self.sg_phase1_end = sg_phase1_end
        self.sg_phase2_end = sg_phase2_end
        self.enable_sg_l2 = enable_sg_l2
        self.weight_sg_l2 = weight_sg_l2
        self.enable_env_map_log_l2 = enable_env_map_log_l2
        self.weight_env_map_log_l2 = weight_env_map_log_l2
        self.enable_diffuse_constraint = enable_diffuse_constraint
        self.weight_diffuse_constraint = weight_diffuse_constraint

        # Lazy import to avoid circular dependency
        self._brdf_renderer = None

    @property
    def brdf_renderer(self):
        if self._brdf_renderer is None:
            from vggt.heads.brdf_renderer import BRDFRenderer
            self._brdf_renderer = BRDFRenderer(render_downsample=self.render_downsample)
        return self._brdf_renderer

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute SG-related losses.

        Args:
            predictions: Dict with keys like "sg_params", "albedo", "normal", etc.
            batch: Dict with "images", optional "gt_sg", etc.

        Returns:
            Dict of loss tensors.
        """
        # --- Extract Phase and Dataset ---
        phase_ratio = batch.get("phase_ratio", 0.0)
        if isinstance(phase_ratio, torch.Tensor):
            phase_ratio = phase_ratio.item()

        dataset_name = batch.get("dataset_name", "unknown")
        if isinstance(dataset_name, (list, tuple)):
            dataset_name = dataset_name[0]

        loss_dict = {}
        total_sg_loss = torch.tensor(0.0, device=_get_device(predictions))

        sg_params = predictions.get("sg_params")
        if sg_params is None:
            return loss_dict

        # --- Repulsion Loss (Always active) ---
        repulsion_val = repulsion_loss(sg_params, threshold=self.repulsion_threshold)
        loss_dict["loss_sg_repulsion"] = repulsion_val
        total_sg_loss = total_sg_loss + repulsion_val * self.weight_repulsion

        # --- Phase Routing & BRDF Render Loss ---
        compute_brdf = True
        if dataset_name not in ["openroomsff", "hypersim"]:
            if phase_ratio <= self.sg_phase2_end:
                # Under phase 2, BRDF render is only for openroomsff and hypersim
                compute_brdf = False

        if compute_brdf:
            brdf_render_val = self._compute_brdf_render_loss(predictions, batch)
            if brdf_render_val is not None:
                loss_dict["loss_brdf_render"] = brdf_render_val
                total_sg_loss = total_sg_loss + brdf_render_val * self.weight_brdf_render

        # --- SG L2 GT Loss (Phase 1, openroomsff only) ---
        compute_l2 = False
        if self.enable_sg_l2 and phase_ratio <= self.sg_phase1_end and dataset_name == "openroomsff":
            compute_l2 = True
            
        if (compute_l2 or self.enable_hungarian) and "gt_sg" in batch and batch["gt_sg"] is not None:
            # We use the Hungarian matching loss to solve the permutation ambiguity of L2 loss
            weight = self.weight_sg_l2 if compute_l2 else self.weight_hungarian
            hungarian_val = hungarian_matching_loss(sg_params, batch["gt_sg"])
            loss_dict["loss_sg_l2"] = hungarian_val
            total_sg_loss = total_sg_loss + hungarian_val * weight

        # --- Env Map Log-L2 Loss (Phase 1, openroomsff only) ---
        if self.enable_env_map_log_l2 and phase_ratio <= self.sg_phase1_end and dataset_name == "openroomsff":
            if "gt_env_map" in batch and batch["gt_env_map"] is not None:
                env_map_val = self._compute_env_map_log_l2_loss(sg_params, batch["gt_env_map"])
                if env_map_val is not None:
                    loss_dict["loss_sg_env_map_log_l2"] = env_map_val
                    total_sg_loss = total_sg_loss + env_map_val * self.weight_env_map_log_l2

        # --- Diffuse Irradiance Constraint ---
        # The per-frame diffuse illumination GT is the shading image (shading.png ->
        # gt_shading); `gt_diffuse_illumination` is kept as an optional override if a
        # dataset ever provides a dedicated HDR diffuse map. Fires on any dataset that
        # provides a shading GT (e.g. openroomsff, hypersim).
        # NOTE: the SG-derived diffuse irradiance is HDR ([0, inf), softplus amplitudes)
        # whereas the shading GT is LDR [0, 1], so it is Reinhard tone-mapped before the
        # comparison (see diffuse_irradiance_constraint, tonemap=True).
        if self.enable_diffuse_constraint and predictions.get("normal") is not None:
            gt_diffuse = batch.get("gt_diffuse_illumination")
            if gt_diffuse is None:
                gt_diffuse = batch.get("gt_shading")
            if gt_diffuse is not None:
                diffuse_val = diffuse_irradiance_constraint(
                    sg_params,
                    predictions["normal"],
                    gt_diffuse,
                    batch.get("gt_mask"),
                    tonemap=True,
                )
                loss_dict["loss_sg_diffuse_constraint"] = diffuse_val
                total_sg_loss = total_sg_loss + diffuse_val * self.weight_diffuse_constraint

        loss_dict["loss_sg_total"] = total_sg_loss
        return loss_dict

    def _compute_brdf_render_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Compute BRDF render loss: MSE(rendered, input_image).

        Requires: albedo, normal, roughness, metallic, sg_params,
                  world_points, pose_enc in predictions.
                  images in batch.
        """
        required_pred_keys = ["albedo", "normal", "roughness", "metallic", "sg_params"]
        for key in required_pred_keys:
            if key not in predictions or predictions[key] is None:
                return None

        if "images" not in batch:
            return None

        # Geometry source: "pred" uses the model's camera/point head predictions,
        # "gt" uses dataset-provided geometry (camera_pos / gt_point_map). Defaults
        # to "pred" (the dataset does not yet ship geometry GT).
        camera_pos = None
        point_map = None
        if self.geometry_source == "gt":
            if "camera_pos" in batch:
                camera_pos = batch["camera_pos"]
            point_map = batch.get("gt_point_map")
        else:  # "pred"
            if predictions.get("pose_enc") is not None:
                from vggt.heads.brdf_renderer import compute_camera_positions
                camera_pos = compute_camera_positions(predictions["pose_enc"])
            point_map = predictions.get("world_points")

        if point_map is None or camera_pos is None:
            return None

        images = batch["images"]  # [B, S, 3, H, W]
        images_hwc = images.permute(0, 1, 3, 4, 2).contiguous()  # [B, S, H, W, 3]

        # Render using BRDF.
        # NOTE: do NOT pass `shading` here. `shading` is the predicted full lighting
        # term; the BRDF render already computes lighting from the SG, so multiplying
        # by shading double-counts illumination and darkens the render ~1.4x.
        rendered = self.brdf_renderer(
            albedo=predictions["albedo"],
            normal=predictions["normal"],
            roughness=predictions["roughness"],
            metallic=predictions["metallic"],
            sg_params=predictions["sg_params"],
            point_map=point_map,
            camera_pos=camera_pos,
        )

        # Resize rendered to match image resolution if needed
        if rendered.shape[2:4] != images_hwc.shape[2:4]:
            B, S, H_r, W_r, C = rendered.shape
            H_i, W_i = images_hwc.shape[2], images_hwc.shape[3]
            rendered_flat = rendered.reshape(B * S, H_r, W_r, C).permute(0, 3, 1, 2)
            rendered_flat = F.interpolate(
                rendered_flat, size=(H_i, W_i), mode="bilinear", align_corners=False
            )
            rendered = rendered_flat.permute(0, 2, 3, 1).reshape(B, S, H_i, W_i, C)

        # Apply mask if available. An all-zero gt_mask is a placeholder for datasets
        # that ship no mask.png (e.g. openroomsff) — treat it as "no mask" and
        # supervise the full image rather than returning 0 (which silently disabled
        # the render loss on those datasets).
        mask = batch.get("gt_mask")
        if mask is not None:
            if mask.ndim == 5 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            mask = mask > 0.5
            mask_expanded = mask.unsqueeze(-1).expand_as(rendered)
            if mask_expanded.sum() >= 1:
                diff_sq = (rendered - images_hwc) ** 2
                return diff_sq[mask_expanded].mean()
            # else: empty/placeholder mask -> fall through to full-image MSE

        return F.mse_loss(rendered, images_hwc)

    def _compute_env_map_log_l2_loss(self, sg_params, gt_env_map, resolution=(64, 128)):
        """Compute Log-L2 loss between reconstructed and GT environment maps."""
        B, S, K, _ = sg_params.shape

        # Dataloader may return a python list for optional env-map fields.
        if isinstance(gt_env_map, (list, tuple)):
            if len(gt_env_map) == 0:
                return None
            if not all(isinstance(x, torch.Tensor) for x in gt_env_map):
                return None
            try:
                gt_env_map = torch.stack(gt_env_map, dim=0)
            except Exception:
                return None

        if not isinstance(gt_env_map, torch.Tensor):
            return None

        gt_env_map = gt_env_map.to(device=sg_params.device, dtype=sg_params.dtype)

        # Accept [B, S, 3, H, W] or [B, S, H, W, 3].
        if gt_env_map.ndim != 5:
            return None

        if gt_env_map.shape[0] != B or gt_env_map.shape[1] != S:
            return None

        if gt_env_map.shape[2] == 3:
            gt_env_map_cf = gt_env_map
        elif gt_env_map.shape[-1] == 3:
            gt_env_map_cf = gt_env_map.permute(0, 1, 4, 2, 3).contiguous()
        else:
            return None

        # Some OpenRooms HDR env maps contain a few inf/nan pixels (corrupt HDR capture
        # / SG-fit overflow). Treat those as bad and EXCLUDE them from the loss (do not
        # supervise the prediction toward a fake value). Build a validity mask from the
        # original GT, then sanitize so the arithmetic stays finite.
        valid_cf = torch.isfinite(gt_env_map_cf).all(dim=2, keepdim=True).to(gt_env_map_cf.dtype)  # [B,S,1,H,W]
        gt_env_map_cf = torch.nan_to_num(gt_env_map_cf, nan=0.0, posinf=0.0, neginf=0.0)

        H_gt, W_gt = gt_env_map_cf.shape[-2:]

        # Reconstruct env map from SG (rendered at a lower resolution for efficiency)
        pred_env_map = render_env_map_from_sg(sg_params, height=resolution[0], width=resolution[1])  # [B, S, h, w, 3]
        pred_env_map = pred_env_map.permute(0, 1, 4, 2, 3)  # [B, S, 3, h, w]

        # Downsample GT and the validity mask to the pred resolution.
        gt_env_map_small = F.interpolate(
            gt_env_map_cf.view(B * S, 3, H_gt, W_gt),
            size=resolution, mode="bilinear", align_corners=False,
        ).view(B, S, 3, resolution[0], resolution[1])
        valid_small = F.interpolate(
            valid_cf.view(B * S, 1, H_gt, W_gt),
            size=resolution, mode="bilinear", align_corners=False,
        ).view(B, S, 1, resolution[0], resolution[1])
        # Keep only fully-valid downsampled pixels (no corrupt source pixel mixed in).
        valid_small = (valid_small > 0.999).to(pred_env_map.dtype)

        # Masked Log-L2: MSE(log(p+1), log(g+1)) over valid GT pixels only.
        per_elem = (torch.log1p(pred_env_map) - torch.log1p(gt_env_map_small)) ** 2  # [B,S,3,h,w]
        denom = valid_small.sum() * per_elem.shape[2]  # * channels
        if float(denom) < 1.0:
            return None  # no valid GT pixels in this batch
        return (per_elem * valid_small).sum() / denom


# ============================================================================
# Loss Functions
# ============================================================================


def repulsion_loss(
    sg_params: torch.Tensor,
    threshold: float = 0.9,
) -> torch.Tensor:
    """Repulsion loss to prevent SG lobes from collapsing to similar directions.

    Penalizes pairs of lobes whose direction cosine similarity exceeds the threshold.
    Loss = Σ_{i<j} max(0, cos_sim(μ_i, μ_j) - threshold)²

    Args:
        sg_params: [B, S, num_lobes, 7] — only directions [:, :, :, :3] are used.
        threshold: Cosine similarity threshold above which penalty is applied.

    Returns:
        Scalar loss tensor.
    """
    if sg_params.ndim == 4:
        B, S, L, _ = sg_params.shape
        sg_params = sg_params.view(B * S, L, 7)
        
    directions = sg_params[:, :, :3]  # [B*S, num_lobes, 3]
    directions = F.normalize(directions, p=2, dim=-1, eps=1e-8)

    B_S, L, _ = directions.shape

    # Compute pairwise cosine similarities: [B, L, L]
    cos_sim = torch.bmm(directions, directions.transpose(1, 2))  # [B, L, L]

    # Create upper-triangular mask (exclude diagonal and lower triangle).
    # Expand over flattened batch-view dimension [B*S] to match cos_sim.
    mask = torch.triu(torch.ones(L, L, device=cos_sim.device, dtype=torch.bool), diagonal=1)
    mask = mask.unsqueeze(0).expand(B_S, -1, -1)  # [B*S, L, L]

    # Extract upper-triangular values
    cos_sim_pairs = cos_sim[mask]  # [B * L*(L-1)/2]

    # Penalize pairs exceeding threshold
    violation = torch.clamp(cos_sim_pairs - threshold, min=0.0)
    loss = (violation ** 2).mean()

    return loss


def hungarian_matching_loss(
    pred_sg: torch.Tensor,
    gt_sg: torch.Tensor,
) -> torch.Tensor:
    """Hungarian matching loss for SG lobes.

    Uses the Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
    the optimal assignment between predicted and GT lobes, then computes the
    L2 loss on matched pairs.

    Args:
        pred_sg: [B, S, num_pred_lobes, 7]
        gt_sg:   [B, S, num_gt_lobes, 7]

    Returns:
        Scalar loss tensor.
    """
    if pred_sg.ndim == 4:
        pred_sg = pred_sg.view(-1, pred_sg.shape[2], 7)
    if gt_sg.ndim == 4:
        gt_sg = gt_sg.view(-1, gt_sg.shape[2], 7)

    B_S = pred_sg.shape[0]
    total_loss = torch.tensor(0.0, device=pred_sg.device)

    for b in range(B_S):
        pred = pred_sg[b]  # [num_pred, 7]
        gt = gt_sg[b]      # [num_gt, 7]

        # SG GT is fitted from HDR env maps and can contain inf/nan lobes in a few
        # corrupt scenes. Drop those lobes so they are neither matched nor supervised
        # (treat the bad GT value as absent rather than as a real zero lobe).
        finite_gt = torch.isfinite(gt).all(dim=-1)
        if not bool(finite_gt.any()):
            continue
        gt = gt[finite_gt]

        # Decompose into comparable scales before matching/scoring. Sharpness spans
        # [1, 1000] and amplitude is HDR ([0, inf)); comparing them with raw L2 makes
        # the loss explode (a sharpness diff of a few hundred squares to tens of
        # thousands) and makes the assignment ignore direction. Use cosine distance
        # for direction, log-space for sharpness, and log1p for amplitude so every
        # term is O(1).
        p_dir = F.normalize(pred[:, :3], dim=-1)           # [P, 3]
        g_dir = F.normalize(gt[:, :3], dim=-1)             # [G, 3]
        p_logsh = torch.log(pred[:, 3:4].clamp(min=1e-3))  # [P, 1]
        g_logsh = torch.log(gt[:, 3:4].clamp(min=1e-3))    # [G, 1]
        p_logamp = torch.log1p(pred[:, 4:7].clamp(min=0.0))  # [P, 3]
        g_logamp = torch.log1p(gt[:, 4:7].clamp(min=0.0))    # [G, 3]

        # Cost matrix [P, G]: direction cosine distance + log-sharpness + log-amplitude.
        dir_cost = 1.0 - p_dir @ g_dir.t()                 # [P, G]
        sh_cost = (p_logsh - g_logsh.t()) ** 2             # [P, G]
        amp_cost = torch.cdist(p_logamp, g_logamp) ** 2    # [P, G]
        cost = dir_cost + sh_cost + amp_cost

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        # Loss on matched pairs (same balanced terms).
        dir_loss = (1.0 - (p_dir[row_ind] * g_dir[col_ind]).sum(dim=-1)).mean()
        sh_loss = F.mse_loss(p_logsh[row_ind], g_logsh[col_ind])
        amp_loss = F.mse_loss(p_logamp[row_ind], g_logamp[col_ind])

        total_loss = total_loss + dir_loss + sh_loss + amp_loss

    return total_loss / max(B_S, 1)


def diffuse_irradiance_constraint(
    sg_params: torch.Tensor,
    normal: torch.Tensor,
    gt_diffuse: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tonemap: bool = True,
) -> torch.Tensor:
    """Constraints SG to match GT diffuse illumination when integrated over normal.

    Args:
        sg_params: [B, S, L, 7] (mu: 3, lambda: 1, color: 3)
        normal: [B, S, H, W, 3] (L2 normalized)
        gt_diffuse: [B, S, H, W, 3] — typically the shading GT (LDR, [0, 1]).
        mask: optional boolean mask
        tonemap: if True, Reinhard tone-map the (HDR) SG diffuse irradiance to [0, 1)
            before comparison, so it matches the LDR shading GT scale.
    """
    B, S, H, W, _ = normal.shape
    L = sg_params.shape[2]

    normals_flat = normal.view(B*S, H*W, 3) # [B*S, N, 3]
    
    if sg_params.ndim == 4:
        sg_params = sg_params.view(B*S, L, 7)
        
    mu  = F.normalize(sg_params[:, :, :3], dim=-1) # [B*S, L, 3]
    lam = sg_params[:, :, 3:4]                     # [B*S, L, 1]
    col = sg_params[:, :, 4:7]                     # [B*S, L, 3]

    pred_diffuse_flat = torch.zeros((B*S, H*W, 3), device=normal.device, dtype=normal.dtype)

    for bs in range(B*S):
        dot = torch.mm(normals_flat[bs], mu[bs].transpose(0, 1)) # [N, L]
        w = torch.exp(lam[bs].transpose(0, 1) * (dot - 1.0))    # [N, L]
        pred_diffuse_flat[bs] = (w.unsqueeze(-1) * col[bs].unsqueeze(0)).sum(dim=1) # [N, 3]

    pred_diffuse = pred_diffuse_flat.view(B, S, H, W, 3)

    # SG diffuse irradiance is HDR ([0, inf)); the shading GT is LDR [0, 1].
    # Reinhard tone-map brings the prediction into [0, 1) for a comparable scale.
    if tonemap:
        pred_diffuse = pred_diffuse / (pred_diffuse + 1.0)

    # Valid pixels = finite GT, intersected with a real spatial mask if one is given.
    # Non-finite GT pixels are excluded (treated as bad, not supervised). An all-zero
    # gt_mask is a placeholder for datasets without a mask.png and is ignored.
    valid = torch.isfinite(gt_diffuse).all(dim=-1)  # [B, S, H, W]
    if mask is not None:
        if mask.ndim == 5 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask > 0.5
        if mask.sum() >= 1:  # ignore all-zero placeholder masks
            valid = valid & mask

    gt_diffuse = torch.nan_to_num(gt_diffuse, nan=0.0, posinf=0.0, neginf=0.0)
    valid_expanded = valid.unsqueeze(-1).expand_as(pred_diffuse)
    if valid_expanded.sum() < 1:
        return (pred_diffuse * 0.0).sum()
    return ((pred_diffuse - gt_diffuse)[valid_expanded] ** 2).mean()


def render_env_map_from_sg(sg_params, height=64, width=128):
    """Reconstruct an equirectangular environment map from SG parameters.

    Args:
        sg_params: [B, S, K, 7] (mu: 3, lambda: 1, color: 3)
        height: Target resolution height
        width: Target resolution width

    Returns:
        [B, S, height, width, 3] environment map
    """
    B, S, K, _ = sg_params.shape
    device = sg_params.device
    
    # Create equirectangular direction grid
    theta = torch.linspace(0, torch.pi, height, device=device)
    phi = torch.linspace(0, 2 * torch.pi, width, device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.cos(theta)
    z = torch.sin(theta) * torch.sin(phi)
    dirs = torch.stack([x, y, z], dim=-1).view(1, 1, height * width, 3) # [1, 1, N, 3]
    
    # SG parameters
    mu  = F.normalize(sg_params[:, :, :, :3], dim=-1) # [B, S, K, 3]
    lam = sg_params[:, :, :, 3:4]                     # [B, S, K, 1]
    col = sg_params[:, :, :, 4:7]                     # [B, S, K, 3]
    
    # Flatten B, S for computation
    mu = mu.view(B*S, K, 3)
    lam = lam.view(B*S, K, 1)
    col = col.view(B*S, K, 3)
    dirs = dirs.view(1, height * width, 3)
    
    # Evaluate SG: sum_i color_i * exp(lambda_i * (dot(dir, mu_i) - 1))
    # Using batches for memory efficiency if needed, but B*S is usually small.
    # dot: [B*S, K, N]. Use matmul (not bmm) so the shared [1, N, 3] direction grid
    # broadcasts across the B*S batch instead of raising a batch-size mismatch.
    dot = torch.matmul(mu, dirs.transpose(1, 2)) # [B*S, K, 3] @ [1, 3, N] -> [B*S, K, N]
    
    # w: [B*S, K, N]
    w = torch.exp(lam * (dot - 1.0))
    
    # radiance: [B*S, 3, N]
    radiance = torch.bmm(col.transpose(1, 2), w) 
    
    # Reshape back to [B, S, height, width, 3]
    radiance = radiance.view(B, S, 3, height, width).permute(0, 1, 3, 4, 2)
    return radiance


def per_pixel_env_loss(
    light_pred: torch.Tensor,
    spatial_idx: torch.Tensor,
    dir_idx: torch.Tensor,
    gt_env_pixel: torch.Tensor,
    log_space: bool = True,
) -> torch.Tensor:
    """Masked log-L1 between the d4rt per-pixel radiance and the imenvlow GT.

    The d4rt lighting branch predicts radiance for randomly sampled (pixel, direction)
    pairs. This gathers the matching GT samples from the dense per-pixel env tile and
    compares them in sign-preserving log space (HDR radiance spans many orders).

    Args:
        light_pred:   [B, S, M, 3] predicted radiance (>= 0, softplus).
        spatial_idx:  [B, S, M] flattened spatial index into Hs*Ws.
        dir_idx:      [B, S, M] flattened direction index into env_h*env_w.
        gt_env_pixel: [B, S, Hs, Ws, env_h, env_w, 3] HDR per-pixel env GT. Frames
                      without GT are all-zero placeholders and are excluded per (b, s).
        log_space:    compare log1p(radiance) instead of raw radiance.

    Returns:
        Scalar loss (0 when no valid samples are present).
    """
    B, S, M = spatial_idx.shape
    Hs, Ws, eh, ew = gt_env_pixel.shape[2:6]
    SP, DN = Hs * Ws, eh * ew

    # Gather the GT radiance for each sampled (spatial, direction) pair.
    gt_flat = gt_env_pixel.reshape(B, S, SP * DN, 3).float()          # [B,S,SP*DN,3]
    comb_idx = (spatial_idx * DN + dir_idx).clamp(0, SP * DN - 1)     # [B,S,M]
    idx = comb_idx.unsqueeze(-1).expand(B, S, M, 3)
    gt_g = torch.gather(gt_flat, 2, idx)                             # [B,S,M,3]

    # Valid = frame actually has env GT (placeholders are all-zero) AND sample finite.
    has_env = gt_env_pixel.reshape(B, S, -1).abs().sum(-1) > 0        # [B,S]
    finite = torch.isfinite(gt_g).all(dim=-1)                        # [B,S,M]
    valid = has_env.unsqueeze(-1) & finite                          # [B,S,M]

    pred = light_pred.float().clamp(min=0.0)
    gt_g = torch.nan_to_num(gt_g, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    if log_space:
        pred = torch.log1p(pred)
        gt_g = torch.log1p(gt_g)

    ve = valid.unsqueeze(-1).expand_as(pred)
    if ve.sum() < 1:
        return (pred * 0.0).sum()
    return (pred - gt_g)[ve].abs().mean()


def _get_device(d: dict) -> torch.device:
    """Get device from the first tensor in a dict."""
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")

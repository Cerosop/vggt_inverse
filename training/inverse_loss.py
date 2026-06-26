# Inverse Rendering Loss module for VGGT.
#
# Loss functions:
# - Albedo:    MSE + Multi-Scale Gradient (MSG) + Scale-Invariant MSE (SI-MSE)
# - Metallic:  MSE + MSG
# - Roughness: MSE + MSG
# - Shading:   MSE + MSG
# - Normal:    Cosine Similarity Loss
#
# Additional losses (NEW):
# - Render Loss:    MSE(albedo * shading, input_image)
# - SSIM Loss:      1 - SSIM(pred, gt) per head (individually controllable)
# - Frequency Loss: L1 of FFT magnitude with high-freq weighting (individually controllable)
#
# Supports missing GTs: if a GT is not present in the batch, the
# corresponding head's loss is skipped.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


@dataclass(eq=False)
class InverseRenderingLoss(nn.Module):
    """Loss module for inverse rendering heads.

    Handles per-head loss computation with support for missing GTs.
    Each head can be independently enabled/disabled via its weight.

    Args:
        weight_albedo: Loss weight for albedo head.
        weight_metallic: Loss weight for metallic head.
        weight_roughness: Loss weight for roughness head.
        weight_normal: Loss weight for normal head.
        weight_shading: Loss weight for shading head.
        msg_scales: Number of scales for multi-scale gradient loss.
        enable_render_loss: Enable render loss (albedo * shading vs image).
        weight_render: Weight for render loss.
        ssim: Dict controlling per-head SSIM loss enables and weight.
        freq: Dict controlling per-head frequency loss enables and weight.
    """

    weight_albedo: float = 1.0
    weight_metallic: float = 1.0
    weight_roughness: float = 1.0
    weight_normal: float = 1.0
    weight_shading: float = 1.0
    msg_scales: int = 4

    # Render loss
    enable_render_loss: bool = False
    weight_render: float = 1.0

    # SG / BRDF render loss
    enable_brdf_render_loss: bool = False
    weight_brdf_render: float = 1.0
    weight_sg_repulsion: float = 0.01
    sg_repulsion_threshold: float = 0.9
    enable_sg_hungarian: bool = False
    weight_sg_hungarian: float = 1.0
    brdf_render_downsample: int = 2
    sg_phase1_end: float = 0.2
    sg_phase2_end: float = 0.4
    weight_sg_l2: float = 1.0
    enable_sg_l2: bool = True
    enable_env_map_log_l2: bool = True
    weight_env_map_log_l2: float = 1.0
    enable_diffuse_constraint: bool = False
    weight_diffuse_constraint: float = 0.5
    # BRDF render geometry source: "pred" (model camera/point heads) or "gt" (dataset).
    brdf_geometry_source: str = "pred"

    # d4rt per-pixel env loss (lighting_mode="per_pixel_env"). Masked log-L1 between
    # the sampled per-pixel radiance and the imenvlow GT (batch["gt_env_pixel"]).
    enable_per_pixel_env_loss: bool = False
    weight_per_pixel_env: float = 1.0

    # SSIM loss (per-head control via dict)
    ssim: Optional[Dict] = None

    # Frequency loss (per-head control via dict)
    freq: Optional[Dict] = None

    # Dispersion penalty — anti-collapse regularizer for metallic / roughness.
    # Penalizes predictions whose masked std drops below `dispersion_target_std`,
    # forcing the model away from "all-constant" degenerate solutions.
    enable_dispersion: bool = False
    weight_dispersion: float = 0.5
    dispersion_target_std: float = 0.08

    # Huber loss instead of L2 for the regression part of selected heads.
    # Huber is quadratic for small errors (|d|<=delta) and linear above, so it
    # is robust against per-pixel outliers from noisy or partially-corrupt GT
    # without sacrificing precision on well-behaved pixels.
    #
    # `use_huber` is per-head: pass either
    #   - a bool — applies to ALL regression heads (albedo/metallic/roughness/
    #     shading); kept for backward compatibility, or
    #   - a dict of per-head flags, e.g.
    #         {enable_albedo: False, enable_metallic: True, enable_roughness: True,
    #          enable_shading: False}
    # Heads whose flag is False keep plain L2/MSE. `normal` uses a cosine loss
    # and is never affected. `huber_delta` is shared by all Huber heads.
    use_huber: Union[bool, Dict] = False
    huber_delta: float = 0.1

    # Dynamic Loss Weighting (Kendall et al. 2018 uncertainty weighting).
    # When enabled the model exposes one learnable log-variance per head
    # (`task_log_var`); the loss applies
    #     L = sum_i [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
    # so heads with consistently large loss automatically downweight themselves
    # — they stop pulling gradient away from the other heads.
    enable_dynamic_weighting: bool = False

    def __post_init__(self):
        super().__init__()
        self.weights = {
            "albedo": self.weight_albedo,
            "metallic": self.weight_metallic,
            "roughness": self.weight_roughness,
            "normal": self.weight_normal,
            "shading": self.weight_shading,
        }

        # Parse SSIM config
        self.ssim_config = self.ssim if self.ssim is not None else {}
        self.ssim_weight = float(self.ssim_config.get("weight", 0.5))
        self.ssim_enables = {
            "albedo": bool(self.ssim_config.get("enable_albedo", False)),
            "metallic": bool(self.ssim_config.get("enable_metallic", False)),
            "roughness": bool(self.ssim_config.get("enable_roughness", False)),
            "normal": bool(self.ssim_config.get("enable_normal", False)),
            "shading": bool(self.ssim_config.get("enable_shading", False)),
        }

        # Parse frequency loss config
        self.freq_config = self.freq if self.freq is not None else {}
        self.freq_weight = float(self.freq_config.get("weight", 0.1))
        self.freq_downsample_size = int(self.freq_config.get("downsample_size", 256))
        self.freq_high_freq_weight = float(self.freq_config.get("high_freq_weight", 2.0))
        self.freq_enables = {
            "albedo": bool(self.freq_config.get("enable_albedo", False)),
            "metallic": bool(self.freq_config.get("enable_metallic", False)),
            "roughness": bool(self.freq_config.get("enable_roughness", False)),
            "normal": bool(self.freq_config.get("enable_normal", False)),
            "shading": bool(self.freq_config.get("enable_shading", False)),
        }

        # Parse per-head Huber config. `use_huber` is either a bool (applies to
        # every regression head) or a dict of per-head `enable_<head>` flags.
        # `normal` uses a cosine loss, so Huber never applies to it.
        _reg_heads = ("albedo", "metallic", "roughness", "shading")
        uh = self.use_huber
        if isinstance(uh, bool):
            self.huber_enables = {h: uh for h in _reg_heads}
        elif uh is None:
            self.huber_enables = {h: False for h in _reg_heads}
        else:
            # dict-like (plain dict or OmegaConf DictConfig)
            self.huber_enables = {
                h: bool(uh.get(f"enable_{h}", False)) for h in _reg_heads
            }

        # SG loss module (lazy-initialized)
        self._sg_loss = None
        if self.enable_brdf_render_loss:
            from sg_loss import SGLoss
            self._sg_loss = SGLoss(
                weight_brdf_render=self.weight_brdf_render,
                weight_repulsion=self.weight_sg_repulsion,
                repulsion_threshold=self.sg_repulsion_threshold,
                weight_hungarian=self.weight_sg_hungarian,
                enable_hungarian=self.enable_sg_hungarian,
                render_downsample=self.brdf_render_downsample,
                sg_phase1_end=self.sg_phase1_end,
                sg_phase2_end=self.sg_phase2_end,
                weight_sg_l2=self.weight_sg_l2,
                enable_sg_l2=self.enable_sg_l2,
                enable_env_map_log_l2=self.enable_env_map_log_l2,
                weight_env_map_log_l2=self.weight_env_map_log_l2,
                enable_diffuse_constraint=self.enable_diffuse_constraint,
                weight_diffuse_constraint=self.weight_diffuse_constraint,
                geometry_source=self.brdf_geometry_source,
            )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute inverse rendering losses.

        Args:
            predictions: Model outputs. Keys: "albedo", "metallic", etc.
                Each tensor has shape [B, S, H, W, C].
            batch: Ground truth dict. Keys: "gt_albedo", "gt_metallic", etc.
                Missing keys are skipped. Each tensor: [B, S, H, W, C].
                Optional masks: "mask_albedo", etc. [B, S, H, W] boolean.
                "images": [B, S, 3, H, W] input images (for render loss).

        Returns:
            Dict of losses including per-head losses and "loss_inv_total".
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=_get_device(predictions))

        # Collect per-head total losses so we can optionally apply Kendall
        # uncertainty weighting once all heads have been computed.
        per_head_loss: Dict[str, torch.Tensor] = {}

        # Pick regression / SI variants per head based on the per-head Huber flag.
        def _pick_reg(head_name):
            if self.huber_enables.get(head_name, False):
                reg = lambda p, g, m: masked_huber_loss(p, g, m, delta=self.huber_delta)
                si  = lambda p, g, m: scale_invariant_huber_loss(p, g, m, delta=self.huber_delta)
                return reg, si, "huber"
            return masked_mse_loss, scale_invariant_mse_loss, "mse"

        for head_name, weight in self.weights.items():
            gt_key = f"gt_{head_name}"

            # Skip if GT not available
            if gt_key not in batch or batch[gt_key] is None:
                continue

            # Skip if prediction not available
            if head_name not in predictions:
                continue

            pred = predictions[head_name]   # [B, S, H, W, C]
            gt = batch[gt_key]              # [B, S, H, W, C]
            mask = batch.get(f"mask_{head_name}", None)  # [B, S, H, W] optional

            if mask is not None:
                if mask.ndim == 5 and mask.shape[-1] == 1:
                    mask = mask.squeeze(-1) # [B, S, H, W]
                mask = mask > 0.5
            # fallback to global mask.png if specific mask is not present
            elif "gt_mask" in batch:
                mask = batch["gt_mask"]
                if mask.ndim == 5 and mask.shape[-1] == 1:
                    mask = mask.squeeze(-1) # [B, S, H, W]
                # Binarize mask just in case
                mask = mask > 0.5

            if head_name == "normal":
                # Build a valid-normal mask: GT pixels stored as RGB≈127 in [0,1]
                # decode to XYZ≈0 in [-1,1] — these are invalid and must be excluded.
                # gt is already in [-1, 1] space after _load_gt normalisation.
                gt_norm_sq = (gt ** 2).sum(dim=-1)          # [B, S, H, W]
                valid_gt_mask = gt_norm_sq > 1e-3            # exclude near-zero normals
                if mask is not None:
                    combined_mask = mask & valid_gt_mask
                else:
                    combined_mask = valid_gt_mask
                loss = cosine_similarity_loss(pred, gt, combined_mask)
                loss_dict[f"loss_inv_{head_name}"] = loss
            else:
                reg_fn, si_fn, reg_key_suffix = _pick_reg(head_name)
                loss_reg = reg_fn(pred, gt, mask)
                loss_msg = multi_scale_gradient_loss(pred, gt, mask, scales=self.msg_scales)
                loss = loss_reg + loss_msg
                loss_dict[f"loss_inv_{head_name}_{reg_key_suffix}"] = loss_reg
                loss_dict[f"loss_inv_{head_name}_msg"] = loss_msg

                # Scale-Invariant variant only for albedo
                if head_name == "albedo":
                    loss_si = si_fn(pred, gt, mask)
                    loss += loss_si
                    loss_dict[f"loss_inv_{head_name}_si"] = loss_si

                loss_dict[f"loss_inv_{head_name}"] = loss

                # --- Per-dataset breakdown (logging only) for metallic / roughness ---
                # A single batch usually mixes samples from several datasets
                # (inside_random sampling), and each dataset has a very different
                # metallic / roughness GT distribution. The aggregate above averages
                # over whatever datasets happened to be drawn this step, which makes
                # the per-step curve extremely noisy. Emit one extra scalar per
                # dataset present in the batch so each dataset's loss can be tracked
                # on its own clean curve. These are detached (logging only) and are
                # NOT added to total_loss — the aggregate already carries the
                # gradient for every sample. Components mirror the aggregate's
                # reg + MSG part (SSIM / freq / dispersion, if enabled, are excluded).
                if head_name in ("metallic", "roughness"):
                    per_ds = _per_dataset_reg_msg(
                        pred, gt, mask,
                        batch.get("dataset_name", None),
                        reg_fn, self.msg_scales,
                        single_value=loss,
                    )
                    for ds_name, ds_loss in per_ds.items():
                        loss_dict[f"loss_inv_{head_name}/{ds_name}"] = ds_loss

            # --- SSIM Loss (per-head) ---
            if self.ssim_enables.get(head_name, False):
                ssim_val = ssim_loss(pred, gt, mask)
                loss_dict[f"loss_inv_{head_name}_ssim"] = ssim_val
                loss = loss + ssim_val * self.ssim_weight

            # --- Frequency Loss (per-head) ---
            if self.freq_enables.get(head_name, False):
                freq_val = frequency_loss(
                    pred, gt, mask,
                    downsample_size=self.freq_downsample_size,
                    high_freq_weight=self.freq_high_freq_weight,
                )
                loss_dict[f"loss_inv_{head_name}_freq"] = freq_val
                loss = loss + freq_val * self.freq_weight

            # --- Dispersion penalty (anti-collapse) ---
            # Applied only to metallic / roughness, the two heads we observed
            # collapsing to near-constant predictions on interiorverse-like
            # scenes whose GT distribution is heavily skewed toward zero.
            if self.enable_dispersion and head_name in ("metallic", "roughness"):
                disp_val = dispersion_penalty(
                    pred, mask, target_std=self.dispersion_target_std,
                )
                loss_dict[f"loss_inv_{head_name}_disp"] = disp_val
                loss = loss + disp_val * self.weight_dispersion

            # Save per-head loss for later weighting
            per_head_loss[head_name] = (loss, weight)

        # --- Combine per-head losses ---
        log_var = predictions.get("task_log_var", None)
        task_names = predictions.get("task_log_var_names", None)
        if (self.enable_dynamic_weighting and log_var is not None
                and task_names is not None):
            # Kendall uncertainty weighting:
            #   L = sum_i [ 0.5 * exp(-s_i) * (static_weight_i * L_i) + 0.5 * s_i ]
            # Static `weight_{name}` is multiplied into L_i so user tuning still
            # influences scale (the learned s_i then adjusts on top).
            for head_name, (loss, static_w) in per_head_loss.items():
                if head_name not in task_names:
                    total_loss = total_loss + loss * static_w
                    continue
                idx = task_names.index(head_name)
                s = log_var[idx]
                weighted = 0.5 * torch.exp(-s) * (loss * static_w) + 0.5 * s
                total_loss = total_loss + weighted
                loss_dict[f"loss_inv_{head_name}_logvar"] = s.detach()
        else:
            for head_name, (loss, static_w) in per_head_loss.items():
                total_loss = total_loss + loss * static_w

        # --- Render Loss ---
        if self.enable_render_loss:
            render_val = render_loss(predictions, batch)
            if render_val is not None:
                loss_dict["loss_inv_render"] = render_val
                total_loss = total_loss + render_val * self.weight_render

        # --- SG / BRDF Render Loss ---
        if self._sg_loss is not None and "sg_params" in predictions:
            sg_loss_dict = self._sg_loss(predictions, batch)
            loss_dict.update(sg_loss_dict)
            sg_total = sg_loss_dict.get("loss_sg_total", 0)
            if isinstance(sg_total, torch.Tensor):
                total_loss = total_loss + sg_total

        # --- d4rt per-pixel env loss ---
        if (self.enable_per_pixel_env_loss
                and "light_pred" in predictions
                and "gt_env_pixel" in batch):
            from sg_loss import per_pixel_env_loss
            env_val = per_pixel_env_loss(
                predictions["light_pred"],
                predictions["light_spatial_idx"],
                predictions["light_dir_idx"],
                batch["gt_env_pixel"],
            )
            loss_dict["loss_per_pixel_env"] = env_val
            total_loss = total_loss + env_val * self.weight_per_pixel_env

        loss_dict["loss_inv_total"] = total_loss
        return loss_dict


# ============================================================================
# Loss functions
# ============================================================================


def masked_mse_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE loss with optional mask.

    Args:
        pred: [B, S, H, W, C]
        gt:   [B, S, H, W, C]
        mask: [B, S, H, W] boolean, optional.
    """
    diff_sq = (pred - gt) ** 2  # [B, S, H, W, C]

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand_as(diff_sq)  # [B, S, H, W, C]
        if mask_expanded.sum() < 1:
            return (pred * 0.0).sum()
        return diff_sq[mask_expanded].mean()
    else:
        return diff_sq.mean()


def masked_huber_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    delta: float = 0.1,
) -> torch.Tensor:
    """Per-pixel Huber loss with optional mask.

    Huber(diff) = { 0.5 * diff^2                   if |diff| <= delta
                  { delta * (|diff| - 0.5 * delta) otherwise
    Quadratic near 0 (smooth gradient), linear in the tail (outlier-robust).

    Args:
        pred: [B, S, H, W, C]
        gt:   [B, S, H, W, C]
        mask: [B, S, H, W] boolean, optional.
        delta: transition point between quadratic and linear regimes
               (in the same units as pred/gt; defaults to 0.1 for [0,1] images).
    """
    diff = pred - gt
    abs_diff = diff.abs()
    quad = 0.5 * diff * diff
    lin = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(abs_diff <= delta, quad, lin)

    if mask is not None:
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        if mask_expanded.sum() < 1:
            return (pred * 0.0).sum()
        return loss[mask_expanded].mean()
    else:
        return loss.mean()


def scale_invariant_huber_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    delta: float = 0.1,
) -> torch.Tensor:
    """Scale-Invariant variant of Huber: subtract per-sample median shift
    between pred and gt before computing Huber. Same idea as
    `scale_invariant_mse_loss` but with Huber's outlier robustness.
    """
    B = pred.shape[0]
    total_loss = torch.tensor(0.0, device=pred.device)
    valid = 0

    for b in range(B):
        p, g = pred[b], gt[b]
        if mask is not None:
            m = mask[b].unsqueeze(-1).expand_as(p)
            if m.sum() < 1:
                continue
            diff = p[m] - g[m]
        else:
            diff = (p - g).reshape(-1)
            m = None

        shift = diff.median()

        if mask is not None:
            si_diff = (p[m] - shift - g[m])
        else:
            si_diff = (p - shift - g).reshape(-1)

        abs_si = si_diff.abs()
        quad = 0.5 * si_diff * si_diff
        lin = delta * (abs_si - 0.5 * delta)
        per_pix = torch.where(abs_si <= delta, quad, lin)
        total_loss = total_loss + per_pix.mean()
        valid += 1

    if valid == 0:
        return (pred * 0.0).sum()
    return total_loss / valid


def dispersion_penalty(
    pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target_std: float = 0.08,
    min_valid_pixels: int = 100,
) -> torch.Tensor:
    """Anti-collapse regularizer: penalize predictions with masked std below
    `target_std`.

    Mechanism:
        loss = max(0, target_std - std(pred_masked))

    Properties:
        - Returns 0 when prediction has enough spatial variation (std >= target).
        - Becomes the dominant gradient signal when prediction degenerates
          toward a near-constant output (the failure mode we observed for
          interiorverse roughness / metallic).
        - Never punishes "too much variation" — only "too little".

    Args:
        pred: [B, S, H, W, C] in [0, 1] (after sigmoid).
        mask: [B, S, H, W] boolean, optional. Pixels with mask=False are
              ignored (e.g. scenes without this head's GT contribute nothing).
        target_std: minimum acceptable std below which the penalty kicks in.
        min_valid_pixels: bail out (return zero loss) if fewer valid pixels.

    Returns:
        Scalar loss in [0, target_std].
    """
    if mask is not None:
        # collapse boolean
        mask_bool = mask if mask.dtype == torch.bool else (mask > 0.5)
        if mask_bool.sum().item() < min_valid_pixels:
            return (pred * 0.0).sum()
        mask_expanded = mask_bool.unsqueeze(-1).expand_as(pred)
        vals = pred[mask_expanded]
    else:
        vals = pred.flatten()

    if vals.numel() < min_valid_pixels:
        return (pred * 0.0).sum()

    current_std = vals.std()
    # ReLU-like one-sided hinge: only penalize being too flat.
    return torch.clamp(target_std - current_std, min=0.0)


def _per_dataset_reg_msg(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor],
    dataset_names,
    reg_fn,
    msg_scales: int,
    single_value: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Split the reg + MSG loss by source dataset for per-dataset logging.

    Args:
        pred, gt: [B, S, H, W, C] prediction / ground truth.
        mask:     [B, S, H, W] boolean mask, or None.
        dataset_names: per-sample dataset identifier from ``batch["dataset_name"]``.
            Expected to be a list/sequence of length B (one string per batch
            element). A bare string (single-dataset batch) is also accepted.
        reg_fn: the regression loss callable used for this head (MSE or Huber).
        msg_scales: number of scales for the multi-scale gradient term.
        single_value: the already-computed aggregate head loss; reused as-is when
            the whole batch comes from one dataset (avoids recomputation).

    Returns:
        {dataset_name: detached scalar tensor}. Empty if dataset names are
        unavailable or malformed. Values are detached — logging only.
    """
    out: Dict[str, torch.Tensor] = {}
    if dataset_names is None:
        return out

    B = pred.shape[0]
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names] * B
    try:
        names = [str(n) for n in dataset_names]
    except TypeError:
        return out
    if len(names) != B or B == 0:
        return out

    groups: Dict[str, list] = {}
    for b, name in enumerate(names):
        groups.setdefault(name, []).append(b)

    # Whole batch is one dataset → per-dataset value == aggregate; reuse it.
    if len(groups) == 1:
        name = next(iter(groups))
        if single_value is not None and torch.is_tensor(single_value):
            out[name] = single_value.detach()
        else:
            l = reg_fn(pred, gt, mask) + multi_scale_gradient_loss(
                pred, gt, mask, scales=msg_scales
            )
            out[name] = l.detach()
        return out

    for name, idxs in groups.items():
        idx = torch.tensor(idxs, device=pred.device, dtype=torch.long)
        p = pred.index_select(0, idx)
        g = gt.index_select(0, idx)
        m = mask.index_select(0, idx) if mask is not None else None
        l = reg_fn(p, g, m) + multi_scale_gradient_loss(p, g, m, scales=msg_scales)
        out[name] = l.detach()
    return out


def scale_invariant_mse_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scale-Invariant MSE: remove per-sample median shift before computing MSE.

    For each sample in the batch, computes the median difference between
    pred and gt, subtracts it, then computes MSE.

    Args:
        pred: [B, S, H, W, C]
        gt:   [B, S, H, W, C]
        mask: [B, S, H, W] optional
    """
    B = pred.shape[0]
    total_loss = torch.tensor(0.0, device=pred.device)
    valid_count = 0

    for b in range(B):
        p = pred[b]   # [S, H, W, C]
        g = gt[b]     # [S, H, W, C]

        if mask is not None:
            m = mask[b].unsqueeze(-1).expand_as(p)  # [S, H, W, C]
            if m.sum() < 1:
                continue
            diff = p[m] - g[m]  # flat
        else:
            diff = (p - g).reshape(-1)

        median_shift = diff.median()

        if mask is not None:
            si_diff = (p[m] - median_shift - g[m])
        else:
            si_diff = (p - median_shift - g).reshape(-1)

        total_loss = total_loss + (si_diff ** 2).mean()
        valid_count += 1

    if valid_count == 0:
        return (pred * 0.0).sum()

    return total_loss / valid_count


def multi_scale_gradient_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scales: int = 4,
) -> torch.Tensor:
    """Multi-Scale Gradient (MSG) loss.

    Computes gradient differences at multiple scales by subsampling.

    Args:
        pred: [B, S, H, W, C]
        gt:   [B, S, H, W, C]
        mask: [B, S, H, W] optional
        scales: Number of subsampling scales.
    """
    # Reshape to [B*S, H, W, C] for gradient computation
    BS = pred.shape[0] * pred.shape[1]
    pred_flat = pred.reshape(BS, *pred.shape[2:])
    gt_flat = gt.reshape(BS, *gt.shape[2:])
    mask_flat = mask.reshape(BS, *mask.shape[2:]) if mask is not None else None

    total = torch.tensor(0.0, device=pred.device)
    for scale in range(scales):
        step = 2 ** scale
        p = pred_flat[:, ::step, ::step]
        g = gt_flat[:, ::step, ::step]
        m = mask_flat[:, ::step, ::step] if mask_flat is not None else None

        total = total + _gradient_loss(p, g, m)

    return total / scales


def _gradient_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gradient loss: L1 difference of spatial gradients.

    Args:
        pred: [N, H, W, C]
        gt:   [N, H, W, C]
        mask: [N, H, W] optional
    """
    diff = pred - gt  # [N, H, W, C]

    if mask is not None:
        mask_c = mask.unsqueeze(-1).expand_as(diff)
    else:
        mask_c = torch.ones_like(diff, dtype=torch.bool)

    diff = diff * mask_c

    # Horizontal gradient
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask_c[:, :, 1:] * mask_c[:, :, :-1]
    grad_x = grad_x * mask_x

    # Vertical gradient
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask_c[:, 1:, :] * mask_c[:, :-1, :]
    grad_y = grad_y * mask_y

    # Clamp
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Normalize
    total_valid = mask_c.sum()
    if total_valid < 1:
        return (pred * 0.0).sum()

    grad_loss = (grad_x.sum() + grad_y.sum()) / total_valid
    return grad_loss


def cosine_similarity_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    invalid_normal_thresh: float = 1e-3,
) -> torch.Tensor:
    """Cosine similarity loss for normals: mean(1 - cos(pred, gt)).

    Args:
        pred: [B, S, H, W, 3] predicted normals (should be unit-normalized)
        gt:   [B, S, H, W, 3] ground truth normals in [-1, 1] space
        mask: [B, S, H, W] optional boolean spatial mask
        invalid_normal_thresh: GT pixels whose squared L2 norm falls below
            this threshold are treated as invalid (e.g. Hypersim encodes
            invalid normals as RGB 127/255 ≈ 0.498 per channel, which maps
            to XYZ ≈ 0 after the [0,1]→[-1,1] decode).  Those pixels are
            excluded from loss computation and back-propagation.
    """
    # ── validity check on GT: exclude near-zero (invalid) normals ──────────
    gt_norm_sq = (gt ** 2).sum(dim=-1)             # [B, S, H, W]
    valid_gt = gt_norm_sq > invalid_normal_thresh   # [B, S, H, W] bool

    # Combine with caller-provided spatial mask
    if mask is not None:
        valid_mask = mask & valid_gt
    else:
        valid_mask = valid_gt

    if valid_mask.sum() < 1:
        return (pred * 0.0).sum()

    # Normalize both
    pred_norm = F.normalize(pred, p=2, dim=-1, eps=eps)
    gt_norm   = F.normalize(gt,   p=2, dim=-1, eps=eps)

    cos_sim = (pred_norm * gt_norm).sum(dim=-1)  # [B, S, H, W]
    loss = 1.0 - cos_sim                          # [B, S, H, W]

    return loss[valid_mask].mean()


# ============================================================================
# NEW: Render Loss
# ============================================================================

def render_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
) -> Optional[torch.Tensor]:
    """Render loss: MSE between (albedo * shading) and input image.

    Args:
        predictions: Must contain "albedo" and "shading", both [B, S, H, W, 3].
        batch: Must contain "images" [B, S, 3, H, W].

    Returns:
        Scalar loss tensor, or None if inputs are missing.
    """
    if "albedo" not in predictions or "shading" not in predictions:
        return None
    if "images" not in batch:
        return None

    albedo = predictions["albedo"]   # [B, S, H, W, 3]
    shading = predictions["shading"] # [B, S, H, W, 3]
    images = batch["images"]         # [B, S, 3, H, W]

    # Convert images to [B, S, H, W, 3]
    images_hwc = images.permute(0, 1, 3, 4, 2).contiguous()

    rendered = albedo * shading  # [B, S, H, W, 3]

    # Resize rendered to match image resolution if needed
    if rendered.shape[2:4] != images_hwc.shape[2:4]:
        B, S, H_r, W_r, C = rendered.shape
        H_i, W_i = images_hwc.shape[2], images_hwc.shape[3]
        rendered = rendered.permute(0, 1, 4, 2, 3).reshape(B * S, C, H_r, W_r)
        rendered = F.interpolate(rendered, size=(H_i, W_i), mode="bilinear", align_corners=False)
        rendered = rendered.reshape(B, S, C, H_i, W_i).permute(0, 1, 3, 4, 2)

    mask = None
    B, S = images_hwc.shape[0], images_hwc.shape[1]
    if "gt_mask" in batch:
        mask = batch["gt_mask"]
        if mask.ndim == 5 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1) # [B, S, H, W]
        
        # Resize mask if needed
        if mask.shape[2:4] != images_hwc.shape[2:4]:
            mask_flat = mask.reshape(B * S, 1, mask.shape[2], mask.shape[3])
            mask_flat = F.interpolate(mask_flat, size=images_hwc.shape[2:4], mode="nearest")
            mask = mask_flat.reshape(B, S, images_hwc.shape[2], images_hwc.shape[3])
            
        mask = mask > 0.5

    diff_sq = (rendered - images_hwc) ** 2
    if mask is not None:
        mask_bool = mask > 0.5  # [B, S, H, W]
        if mask_bool.sum() < 1:
            return (rendered * 0.0).sum()
        # Apply mask
        diff_sq = diff_sq[mask_bool]  # Now it's [N, 3] where N = sum(mask_bool)
        return diff_sq.mean()
    return diff_sq.mean()


# ============================================================================
# NEW: SSIM Loss
# ============================================================================

def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a Gaussian window for SSIM computation."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()
    return window


def _ssim_compute(
    pred: torch.Tensor,
    gt: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM between pred and gt images.

    Args:
        pred: [N, C, H, W]
        gt:   [N, C, H, W]

    Returns:
        SSIM map averaged over channels: [N, 1, H', W']
    """
    # 確保在 float32 下計算 SSIM，避免 bfloat16/fp16 平方相減時產生嚴重的精度遺失導致負數 variance (進而使梯度爆炸/全白圖)
    pred = pred.float()
    gt = gt.float()

    channels = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_gt = F.conv2d(gt, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred * mu_pred
    mu_gt_sq = mu_gt * mu_gt
    mu_pred_gt = mu_pred * mu_gt

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_gt_sq = F.conv2d(gt * gt, window, padding=padding, groups=channels) - mu_gt_sq
    sigma_pred_gt = F.conv2d(pred * gt, window, padding=padding, groups=channels) - mu_pred_gt

    # 必須加上 ReLU 避免因為浮點數誤差造成 sigma_sq 為負數，否則分母會趨近於整數 0 導致 loss collapse 產出純色圖片
    sigma_pred_sq = torch.relu(sigma_pred_sq)
    sigma_gt_sq = torch.relu(sigma_gt_sq)

    ssim_map = ((2 * mu_pred_gt + C1) * (2 * sigma_pred_gt + C2)) / \
               ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))

    return ssim_map.mean(dim=1, keepdim=True)  # average over channels


def ssim_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    window_size: int = 11,
) -> torch.Tensor:
    """SSIM loss: 1 - SSIM(pred, gt).

    Args:
        pred: [B, S, H, W, C]
        gt:   [B, S, H, W, C]
        mask: [B, S, H, W] optional (applied after SSIM computation)
        window_size: Gaussian window size for SSIM.

    Returns:
        Scalar loss tensor.
    """
    B, S, H, W, C = pred.shape

    # Reshape to [B*S, C, H, W] for conv2d operations
    pred_nchw = pred.reshape(B * S, H, W, C).permute(0, 3, 1, 2).contiguous()
    gt_nchw = gt.reshape(B * S, H, W, C).permute(0, 3, 1, 2).contiguous()

    # Compute SSIM (returns [B*S, 1, H', W'])
    ssim_map = _ssim_compute(pred_nchw, gt_nchw, window_size=window_size)

    # Loss = 1 - SSIM
    loss_map = 1.0 - ssim_map  # [B*S, 1, H', W']

    if mask is not None:
        # Resize mask to match SSIM output size
        mask_flat = mask.reshape(B * S, H, W).unsqueeze(1).float()  # [B*S, 1, H, W]
        if loss_map.shape[2:] != mask_flat.shape[2:]:
            mask_flat = F.interpolate(mask_flat, size=loss_map.shape[2:], mode="nearest")
        mask_bool = mask_flat > 0.5
        if mask_bool.sum() < 1:
            return (pred * 0.0).sum()
        return loss_map[mask_bool].mean()
    else:
        return loss_map.mean()


# ============================================================================
# NEW: Frequency Loss
# ============================================================================

def frequency_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    downsample_size: int = 256,
    high_freq_weight: float = 2.0,
) -> torch.Tensor:
    """Frequency loss: L1 difference of FFT magnitudes with high-frequency weighting.

    Uses torch.fft.rfft2 (real FFT) for efficiency (~50% less compute than full FFT).
    Downsamples inputs before FFT to reduce VRAM and computation time.

    Args:
        pred: [B, S, H, W, C]
        gt:   [B, S, H, W, C]
        mask: [B, S, H, W] optional (applied as spatial mask before FFT)
        downsample_size: Target size for downsampling before FFT.
        high_freq_weight: Multiplicative factor for high-frequency components.

    Returns:
        Scalar loss tensor.
    """
    B, S, H, W, C = pred.shape

    # Reshape to [B*S, C, H, W]
    pred_nchw = pred.reshape(B * S, H, W, C).permute(0, 3, 1, 2).contiguous()
    gt_nchw = gt.reshape(B * S, H, W, C).permute(0, 3, 1, 2).contiguous()

    # Apply mask as spatial zeroing before FFT (if available)
    if mask is not None:
        mask_nchw = mask.reshape(B * S, 1, H, W).expand_as(pred_nchw).float()
        pred_nchw = pred_nchw * mask_nchw
        gt_nchw = gt_nchw * mask_nchw

    # Downsample for efficiency
    target_h = min(downsample_size, H)
    target_w = min(downsample_size, W)
    if H > target_h or W > target_w:
        pred_nchw = F.interpolate(pred_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        gt_nchw = F.interpolate(gt_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)

    # Compute real FFT (rfft2 outputs half-spectrum, saving memory)
    # Cast to float32 for FFT stability
    pred_f = torch.fft.rfft2(pred_nchw.float())
    gt_f = torch.fft.rfft2(gt_nchw.float())

    # Magnitude spectrum
    pred_mag = torch.abs(pred_f)
    gt_mag = torch.abs(gt_f)

    # Create high-frequency weighting mask
    freq_h, freq_w = pred_mag.shape[-2], pred_mag.shape[-1]
    weight_mask = _make_freq_weight_mask(freq_h, freq_w, high_freq_weight, pred_mag.device)

    # Weighted L1 loss on magnitudes
    diff = torch.abs(pred_mag - gt_mag) * weight_mask

    return diff.mean().to(pred.dtype)


def _make_freq_weight_mask(
    h: int, w: int, high_freq_weight: float, device: torch.device,
) -> torch.Tensor:
    """Create a frequency weighting mask: higher weight for high frequencies.

    For rfft2 output, the frequency layout is:
    - rows: [0, 1, ..., h//2, -(h//2-1), ..., -1]  (full range)
    - cols: [0, 1, ..., w-1]  (non-negative only, since rfft2)

    We compute normalized radial distance from DC (0,0) and use it as weight.

    Returns:
        Weight mask [1, 1, h, w]
    """
    freq_y = torch.fft.fftfreq(h, device=device)       # [-0.5, 0.5]
    freq_x = torch.fft.rfftfreq(h, device=device)       # [0, 0.5] — note: use h for the rfft dim too
    # Actually rfftfreq uses the original signal length for the last dim
    freq_x = torch.linspace(0, 0.5, w, device=device)

    gy, gx = torch.meshgrid(freq_y, freq_x, indexing="ij")
    radius = torch.sqrt(gy ** 2 + gx ** 2)  # [h, w], range [0, ~0.707]

    # Normalize to [0, 1]
    radius = radius / (radius.max() + 1e-8)

    # Weight: 1.0 for DC, up to high_freq_weight for highest frequency
    weight = 1.0 + (high_freq_weight - 1.0) * radius

    return weight.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]


def _get_device(d: dict) -> torch.device:
    """Get device from the first tensor in a dict."""
    for v in d.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")

"""
diag_inverse_issues.py
======================
Non-training diagnostics for two failure modes:

  Q1  Albedo / shading artifacts on uniform regions (walls, sky, dark areas)
  Q2  Roughness / metallic collapse to all-black predictions

What this runs (single forward pass per scene, three variants each):

  (a) Pre-sigmoid logit stats per head      → detects "saturating logits"
  (b) Pre-sigmoid |logit| heatmap per head  → localizes the artifact spatially
  (c) Per-stage ResNeXt feature L2-norm map → checks ResNeXt as artifact source
  (d) Aggregator patch-token L2-norm map    → checks register-overflow pattern
  (e) Re-forward with ResNeXt fusion zeroed → does artifact disappear?
  (f) Prediction stats vs GT stats          → Q2 collapse quantification

Scenes diagnosed are the top-3 highest-loss scenes (per dataset/split/head)
that appear in ``loss_analysis/stats.json`` for albedo / shading / roughness /
metallic — i.e., the worst observed cases.

Outputs under ``diag_outputs/{dataset}/{split}/{scene}/``:
    raw_logit_stats.json
    pred_stats.json
    heatmap_logit_{head}.png         (jet colormap of |logit| per pixel)
    heatmap_resnext_layer{i}.png     (jet colormap of feature L2 norm)
    heatmap_agg_tokens.png           (jet colormap of last-layer LoRA patch-token norm)
    no_resnext/pred_{head}.png       (re-forward with fuse_projects ⇒ 0)
    normal/pred_{head}.png           (standard forward, same frame range)
    summary.json
"""

import os
import sys
import json
import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))

from hydra import initialize, compose

from eval_scene_loss import (
    HEAD_FILES, HEAD_NAMES, get_target_hw,
    load_scene_batch, _to_bgr_uint8, load_model, CKPT_PATH, NUM_FRAMES,
)
from eval_topk_all_preds import (
    LOSS_ANALYSIS_DIR, STATS_PATH, DATA_BASE,
    load_frame_indices_map, resolve_scene_dir, scene_avail_gts,
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "diag_outputs")

# Diagnose worst scenes for these (dataset, split, head) combinations.
DIAG_TARGETS = [
    ("hypersim",      "train", "albedo"),
    ("hypersim",      "train", "shading"),
    ("hypersim",      "test",  "albedo"),
    ("hypersim",      "test",  "shading"),
    ("structured3d",  "train", "albedo"),
    ("interiorverse", "train", "albedo"),
    ("interiorverse", "train", "roughness"),
    ("interiorverse", "train", "metallic"),
    ("matrixcity_normal", "train", "roughness"),
    ("matrixcity_normal", "train", "metallic"),
]
TOP_K_PER_TARGET = 3   # take max3 from each target

# Heads that route through dpt_res (ResNeXt fusion) per yaml; others are pure dpt.
DPT_RES_HEADS = {"albedo", "shading"}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Hook utilities
# ============================================================================

class HookRecorder:
    """Collect forward outputs from a list of (name, module) pairs."""
    def __init__(self):
        self.records: Dict[str, list] = {}
        self.handles = []

    def attach(self, name: str, module: nn.Module):
        def _hook(_m, _inp, out):
            self.records.setdefault(name, []).append(out.detach())
        self.handles.append(module.register_forward_hook(_hook))

    def clear(self):
        self.records = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================================
# Visualisation helpers
# ============================================================================

def heatmap_to_png(arr2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """[H,W] float → uint8 jet-colormap, resized to target_hw."""
    a = arr2d.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        norm = np.zeros_like(a, dtype=np.uint8)
    else:
        norm = ((a - lo) / (hi - lo) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    H, W = target_hw
    if color.shape[0] != H or color.shape[1] != W:
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
    return color


def save_pred_png(pred: torch.Tensor, head: str, path: str):
    """Save one head's prediction (single frame, [H,W,C]) as RGB png."""
    arr = pred.float().cpu().numpy()
    if head == "normal":
        arr = (arr + 1.0) / 2.0
    cv2.imwrite(path, _to_bgr_uint8(arr))


# ============================================================================
# ResNeXt-disabled forward (zeroes out fuse_projects contribution)
# ============================================================================

class _ZeroFuse(nn.Module):
    """Returns zeros with the same output shape as the wrapped fuse module.

    Accepts either:
        - bare nn.Conv2d (legacy fuse_projects)
        - nn.Sequential containing a Conv2d (new P2 fusion: Conv2d + GroupNorm)
    """
    def __init__(self, mod: nn.Module):
        super().__init__()
        if isinstance(mod, nn.Conv2d):
            self.out_channels = mod.out_channels
        elif isinstance(mod, nn.Sequential):
            convs = [m for m in mod if isinstance(m, nn.Conv2d)]
            if not convs:
                raise TypeError(f"No Conv2d found inside Sequential: {mod}")
            # The last Conv2d's out_channels determines fused output channels.
            self.out_channels = convs[-1].out_channels
        else:
            raise TypeError(f"Unsupported fuse module type: {type(mod)}")
    def forward(self, x):
        return torch.zeros(x.shape[0], self.out_channels,
                           x.shape[2], x.shape[3],
                           device=x.device, dtype=x.dtype)


def patch_fuse_to_zero(head_module) -> nn.ModuleList:
    """Replace head.fuse_projects with zero-returning wrappers; return original."""
    orig = head_module.fuse_projects
    head_module.fuse_projects = nn.ModuleList([_ZeroFuse(c) for c in orig])
    return orig


def restore_fuse(head_module, orig):
    head_module.fuse_projects = orig


# ============================================================================
# Stat helpers
# ============================================================================

def tensor_stats(t: torch.Tensor) -> dict:
    t_f = t.float()
    return {
        "min": float(t_f.amin().item()),
        "max": float(t_f.amax().item()),
        "mean": float(t_f.mean().item()),
        "std": float(t_f.std().item()),
        "abs_p99": float(torch.quantile(t_f.abs().flatten(), 0.99).item()),
        "abs_p999": float(torch.quantile(t_f.abs().flatten(), 0.999).item()),
        "nan": bool(torch.isnan(t_f).any().item()),
        "inf": bool(torch.isinf(t_f).any().item()),
        "shape": list(t.shape),
    }


def gt_stats_from_batch(batch: dict) -> Dict[str, dict]:
    """Compute per-head GT stats (mean/std/median + mask-aware) for the batch."""
    out = {}
    for head in HEAD_NAMES:
        gt_key, m_key = f"gt_{head}", f"mask_{head}"
        if gt_key not in batch:
            continue
        gt = batch[gt_key].float()  # [1,S,H,W,C]
        mask = batch[m_key]         # [1,S,H,W]
        if mask.sum().item() < 1:
            out[head] = {"valid_pixels": 0}
            continue
        m_exp = mask.unsqueeze(-1).expand_as(gt).bool()
        vals = gt[m_exp]
        if head == "normal":
            # report length stats of GT vector instead of channel-mixing mean
            vlen = vals.reshape(-1, 3).norm(dim=-1)
            out[head] = {
                "valid_pixels": int(mask.sum().item()),
                "vec_len_mean": float(vlen.mean().item()),
                "vec_len_std":  float(vlen.std().item()),
            }
        else:
            out[head] = {
                "valid_pixels": int(mask.sum().item()),
                "mean":   float(vals.mean().item()),
                "median": float(vals.median().item()),
                "std":    float(vals.std().item()),
                "frac_gt_0p05": float((vals > 0.05).float().mean().item()),
                "frac_gt_0p1":  float((vals > 0.1).float().mean().item()),
                "frac_gt_0p3":  float((vals > 0.3).float().mean().item()),
                "frac_gt_0p5":  float((vals > 0.5).float().mean().item()),
            }
    return out


def pred_stats_per_head(preds: dict) -> Dict[str, dict]:
    out = {}
    for head, p in preds.items():
        if head not in HEAD_NAMES:
            continue
        out[head] = tensor_stats(p)
    return out


# ============================================================================
# Per-scene diagnostic
# ============================================================================

def diagnose_scene(model, dataset: str, split: str, scene: str,
                   frame_indices: List[int], h: int, w: int,
                   reason: str) -> dict:
    sdir, fids = resolve_scene_dir(dataset, split, scene)
    avail = scene_avail_gts(sdir, fids[0])
    batch = load_scene_batch(sdir, fids, frame_indices, avail, h, w)
    # Pick the first frame as the "focus" frame for spatial heatmaps.
    focus = 0

    out_dir = os.path.join(OUTPUT_DIR, dataset, split,
                           f"{reason}__{scene}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "no_resnext"), exist_ok=True)

    # ---- Hooks ----
    rec = HookRecorder()
    # pre-sigmoid output of each head
    if model.inverse_heads is not None:
        for head_name in HEAD_NAMES:
            head_module = getattr(model.inverse_heads, f"{head_name}_head", None)
            if head_module is None:
                continue
            # output_conv2 is the last Conv before sigmoid/tanh
            rec.attach(f"logit_{head_name}", head_module.scratch.output_conv2)
    # ResNeXt layer outputs
    res_enc = getattr(model.inverse_heads, "res_encoder", None)
    if res_enc is not None:
        for i, layer in enumerate([res_enc.layer1, res_enc.layer2,
                                    res_enc.layer3, res_enc.layer4]):
            rec.attach(f"resnext_l{i+1}", layer)

    # Aggregator output is a tuple; hook on the aggregator itself
    agg_out_holder = {}
    def _agg_hook(_m, _inp, out):
        agg_out_holder["out"] = out  # tuple
    agg_handle = model.aggregator.register_forward_hook(_agg_hook)

    try:
        with torch.no_grad():
            preds = model(batch["images"])
    except torch.cuda.OutOfMemoryError:
        rec.remove(); agg_handle.remove()
        torch.cuda.empty_cache()
        return {"error": "OOM"}

    # ---- Pre-sigmoid logit stats + heatmap ----
    logit_stats: Dict[str, dict] = {}
    target_hw = (h, w)
    for head_name in HEAD_NAMES:
        key = f"logit_{head_name}"
        if key not in rec.records:
            continue
        # records[key] is list[Tensor[B*S, C, H', W']]; concat along batch
        cat = torch.cat(rec.records[key], dim=0)
        logit_stats[head_name] = tensor_stats(cat)
        # spatial L2 across channels for focus frame
        focus_t = cat[focus]                       # [C, H', W']
        per_pixel_l2 = focus_t.float().pow(2).sum(0).sqrt().cpu().numpy()
        cv2.imwrite(os.path.join(out_dir, f"heatmap_logit_{head_name}.png"),
                    heatmap_to_png(per_pixel_l2, target_hw))

    # ---- ResNeXt feature norm maps ----
    for i in range(1, 5):
        key = f"resnext_l{i}"
        if key not in rec.records:
            continue
        cat = torch.cat(rec.records[key], dim=0)   # [B*S, C, h, w]
        focus_t = cat[focus]
        l2 = focus_t.float().pow(2).sum(0).sqrt().cpu().numpy()
        cv2.imwrite(os.path.join(out_dir, f"heatmap_resnext_layer{i}.png"),
                    heatmap_to_png(l2, target_hw))

    # ---- Aggregator patch-token norm map (last intermediate from LoRA path) ----
    if "out" in agg_out_holder:
        agg_tuple = agg_out_holder["out"]
        aggregated_list, lora_list, p_start_idx, _light = agg_tuple
        # prefer LoRA path
        token_list = lora_list if lora_list is not None else aggregated_list
        if token_list:
            last_inter = token_list[-1]            # [B,S,P,2C]
            patches = last_inter[0, focus, p_start_idx:]   # [N,2C]
            N, _ = patches.shape
            # infer grid size: N = (H/14)*(W/14)
            ph = h // 14; pw = w // 14
            if N == ph * pw:
                patch_norms = patches.float().norm(dim=-1).reshape(ph, pw).cpu().numpy()
                cv2.imwrite(os.path.join(out_dir, "heatmap_agg_tokens.png"),
                            heatmap_to_png(patch_norms, target_hw))
            else:
                logger.warning(f"agg token grid mismatch: N={N} vs ph*pw={ph*pw}")
    agg_handle.remove()
    rec.remove()

    # ---- Save standard prediction (focus frame) for every head ----
    inp = batch["images"][0, focus].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    cv2.imwrite(os.path.join(out_dir, "input.png"), _to_bgr_uint8(inp))
    for head in HEAD_NAMES:
        if head in preds:
            save_pred_png(preds[head][0, focus], head,
                          os.path.join(out_dir, "normal", f"pred_{head}.png"))
        gt_key = f"gt_{head}"
        if gt_key in batch:
            gt_arr = batch[gt_key][0, focus].float().cpu().numpy()
            if head == "normal":
                gt_arr = (gt_arr + 1.0) / 2.0
            cv2.imwrite(os.path.join(out_dir, "normal", f"gt_{head}.png"),
                        _to_bgr_uint8(gt_arr))

    # ---- Re-forward with ResNeXt fusion zeroed (only affects dpt_res heads) ----
    saved_orig = {}
    for head_name in DPT_RES_HEADS:
        head_module = getattr(model.inverse_heads, f"{head_name}_head", None)
        if head_module is not None and hasattr(head_module, "fuse_projects"):
            saved_orig[head_name] = patch_fuse_to_zero(head_module)
    try:
        with torch.no_grad():
            preds_nores = model(batch["images"])
        for head in DPT_RES_HEADS:
            if head in preds_nores:
                save_pred_png(preds_nores[head][0, focus], head,
                              os.path.join(out_dir, "no_resnext", f"pred_{head}.png"))
        # also dump non-dpt-res heads (should be identical) for sanity check
        for head in HEAD_NAMES:
            if head in DPT_RES_HEADS:
                continue
            if head in preds_nores:
                save_pred_png(preds_nores[head][0, focus], head,
                              os.path.join(out_dir, "no_resnext", f"pred_{head}.png"))
    finally:
        for head_name, orig in saved_orig.items():
            restore_fuse(getattr(model.inverse_heads, f"{head_name}_head"), orig)

    # ---- Stats files ----
    with open(os.path.join(out_dir, "raw_logit_stats.json"), "w") as f:
        json.dump(logit_stats, f, indent=2)
    with open(os.path.join(out_dir, "pred_stats.json"), "w") as f:
        json.dump({
            "pred_normal_path":     pred_stats_per_head(preds),
            "pred_no_resnext_path": pred_stats_per_head(preds_nores),
            "gt_stats":             gt_stats_from_batch(batch),
        }, f, indent=2)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "dataset": dataset, "split": split, "scene": scene,
            "reason": reason, "focus_frame_idx": frame_indices[focus],
            "frame_indices": frame_indices,
            "available_gts": [k for k, v in avail.items() if v],
        }, f, indent=2)

    del batch, preds, preds_nores
    torch.cuda.empty_cache()
    return {"out_dir": out_dir, "logit_stats": logit_stats}


# ============================================================================
# Main
# ============================================================================

def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(0)

    with open(STATS_PATH) as f:
        stats = json.load(f)

    # Build the diagnose plan: each entry = (ds, split, head, rank, scene)
    plan = []
    for (ds, sp, head) in DIAG_TARGETS:
        entry = stats.get(ds, {}).get(sp, {}).get(head, {})
        if not entry:
            logger.warning(f"No stats for {ds}/{sp}/{head} — skipping")
            continue
        for i, e in enumerate(entry.get("max3", [])[:TOP_K_PER_TARGET]):
            plan.append({
                "dataset": ds, "split": sp, "head": head,
                "rank": i + 1, "scene": e["scene"], "loss": e["loss"],
            })

    # Dedupe by (ds, sp, scene) — we only need to forward each unique scene once
    by_scene: Dict[Tuple[str, str, str], dict] = {}
    for p in plan:
        key = (p["dataset"], p["split"], p["scene"])
        tag = f"{p['head']}{p['rank']}"
        if key not in by_scene:
            by_scene[key] = {"dataset": p["dataset"], "split": p["split"],
                              "scene": p["scene"], "tags": [tag]}
        else:
            by_scene[key]["tags"].append(tag)
    logger.info(f"{len(by_scene)} unique scenes to diagnose")

    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="inverse_rendering")
    logger.info(f"Loading checkpoint: {CKPT_PATH}")
    model = load_model(cfg)

    # Preload frame_indices lookup
    fi_lookup: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    needed = {(j["dataset"], j["split"]) for j in by_scene.values()}
    for (ds, sp) in needed:
        fi_lookup[(ds, sp)] = load_frame_indices_map(ds, sp)

    h, w = get_target_hw()
    summary = []
    for job in by_scene.values():
        ds, sp, scene = job["dataset"], job["split"], job["scene"]
        fi = fi_lookup.get((ds, sp), {}).get(scene)
        if fi is None:
            logger.warning(f"[{ds}/{sp}/{scene}] no frame_indices in CSV, skipping")
            continue
        reason = "_".join(sorted(job["tags"]))
        logger.info(f"→ {ds}/{sp}/{scene}   tags={job['tags']}   frames={fi}")
        try:
            res = diagnose_scene(model, ds, sp, scene, fi, h, w, reason)
            summary.append({"scene": scene, "dataset": ds, "split": sp,
                            "tags": job["tags"], **res})
        except Exception as e:
            logger.warning(f"  ✗ failed: {e}")
            summary.append({"scene": scene, "dataset": ds, "split": sp,
                            "tags": job["tags"], "error": str(e)})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "_index.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Done.  Outputs under: {OUTPUT_DIR}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

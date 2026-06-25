"""
compare_mv_consistency.py
=========================
Compare the **multi-view consistency** of two inverse-rendering models
(YOUR VGGT+Inverse vs MVInverse) from predictions produced by
``predict_mipnerf360.py``.

Protocol (the MVInverse "reprojection RMSE across views" metric, Tab. 2 / 4)
---------------------------------------------------------------------------
For an ordered pair of views (i, j):
  1. Back-project every pixel of view *i* to a 3D world point using the **shared
     VGGT geometry** (predicted depth + camera pose).
  2. Project that world point into view *j* → sub-pixel location (u_j, v_j).
  3. Keep the correspondence only if it is in-bounds AND not occluded
     (|z_proj − depth_j(u_j,v_j)| / z_proj < occ_thr) AND both endpoints pass a
     VGGT-confidence gate.  This valid mask is **identical for both methods**
     (it depends only on geometry), so the comparison is apples-to-apples.
  4. For each method, bilinearly sample its material map of view *j* at (u_j,v_j)
     and measure disagreement with view *i*'s prediction:
        • albedo / metallic / roughness / shading → RMSE over the map
        • normal → rotate both to world space, report mean angular error (deg)
                   and the fraction of pixels within 11.25° / 30°.

Lower RMSE / lower angular error = more multi-view consistent = better.

Inputs : the ``--pred_dir`` written by predict_mipnerf360.py.
Outputs: under ``--out_dir`` ::

    consistency_results.json     # per-scene + overall, per method per attr
    consistency_summary.csv      # flat table for quick diffing
    <scene>/errmaps/pair{i}_{j}_{attr}.png   # vggt|mvinverse error heatmaps

Usage::

    python compare_mv_consistency.py \
        --pred_dir mipnerf360_preds \
        --out_dir  mipnerf360_consistency \
        --occ_thr 0.05 --conf_percentile 10

NOTE: This script does NOT run any network — it only reads the dumped arrays.
"""

import os
import sys
import json
import csv
import argparse
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("compare_mv_consistency")

ATTRS = {"albedo": 3, "metallic": 1, "roughness": 1, "normal": 3, "shading": 3}
MATERIAL_ATTRS = ["albedo", "metallic", "roughness", "shading"]   # RMSE attrs
# Error-map color scale (max value mapped to red).
VMAX = {"albedo": 0.3, "metallic": 0.3, "roughness": 0.3, "shading": 0.3,
        "normal": 45.0}


# ==============================================================================
# Loading dumped arrays
# ==============================================================================

def load_geometry(scene_dir, s, device):
    d = np.load(os.path.join(scene_dir, "geometry", f"frame_{s:03d}.npz"))
    g = {
        "R": torch.from_numpy(d["extrinsic"][:, :3]).float().to(device),   # [3,3]
        "t": torch.from_numpy(d["extrinsic"][:, 3]).float().to(device),    # [3]
        "K": torch.from_numpy(d["intrinsic"]).float().to(device),          # [3,3]
        "depth": torch.from_numpy(d["depth"]).float().to(device),          # [H,W]
        "depth_conf": torch.from_numpy(d["depth_conf"]).float().to(device),# [H,W]
    }
    return g


def load_materials(scene_dir, method, s, device):
    f = os.path.join(scene_dir, method, f"frame_{s:03d}.npz")
    d = np.load(f)
    out = {}
    for a in ATTRS:
        if a in d.files:
            t = torch.from_numpy(d[a]).float().to(device)   # [H,W,C]
            out[a] = t.permute(2, 0, 1).contiguous()         # [C,H,W]
    return out


# ==============================================================================
# Geometry: correspondence between two views
# ==============================================================================

def build_correspondence(gi, gj, H, W, occ_thr, conf_thr_i, conf_thr_j, device):
    """Return (grid, valid) mapping view-i pixels → view-j sub-pixel coords.

    grid : [1,H,W,2] normalized for F.grid_sample (align_corners=True)
    valid: [H,W] bool mask (in-bounds, not occluded, confident on both ends)
    """
    eps = 1e-6
    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(us)
    pix = torch.stack([us, vs, ones], dim=-1)                # [H,W,3]

    # back-project view i  →  world
    Kinv_i = torch.inverse(gi["K"])                          # [3,3]
    ray = pix @ Kinv_i.transpose(0, 1)                       # [H,W,3]  (= Kinv_i @ pix)
    x_cam_i = gi["depth"].unsqueeze(-1) * ray                # [H,W,3]
    X_world = (x_cam_i - gi["t"]) @ gi["R"]                  # = R_i^T (x_cam - t)

    # project into view j
    x_cam_j = X_world @ gj["R"].transpose(0, 1) + gj["t"]    # = R_j X + t_j
    z_j = x_cam_j[..., 2]                                     # [H,W]
    uvw = x_cam_j @ gj["K"].transpose(0, 1)                  # = K_j x_cam_j
    u_j = uvw[..., 0] / (uvw[..., 2] + eps)
    v_j = uvw[..., 1] / (uvw[..., 2] + eps)

    in_bounds = (z_j > eps) & (u_j >= 0) & (u_j <= W - 1) & (v_j >= 0) & (v_j <= H - 1)

    # normalized grid for sampling view j
    gx = 2.0 * u_j / (W - 1) - 1.0
    gy = 2.0 * v_j / (H - 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)        # [1,H,W,2]

    # occlusion test: depth of view j at the projected location vs z_j
    depth_j = gj["depth"].unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
    depth_j_samp = F.grid_sample(depth_j, grid, mode="bilinear",
                                 padding_mode="border", align_corners=True)[0, 0]
    occ_ok = (torch.abs(z_j - depth_j_samp) / z_j.clamp(min=eps)) < occ_thr

    # confidence gate (VGGT geometry confidence), applied equally to both methods
    conf_ok = gi["depth_conf"] > conf_thr_i
    if conf_thr_j > 0:
        conf_j = gj["depth_conf"].unsqueeze(0).unsqueeze(0)
        conf_j_samp = F.grid_sample(conf_j, grid, mode="bilinear",
                                    padding_mode="border", align_corners=True)[0, 0]
        conf_ok = conf_ok & (conf_j_samp > conf_thr_j)

    valid = in_bounds & occ_ok & conf_ok
    return grid, valid


# ==============================================================================
# Per-attribute error
# ==============================================================================

def sample(map_chw, grid):
    """Bilinearly sample [C,H,W] at grid [1,H,W,2] → [C,H,W]."""
    out = F.grid_sample(map_chw.unsqueeze(0), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out[0]


def material_error_map(mat_i, mat_j, attr, grid):
    """Per-pixel Euclidean error [H,W] for a material attribute."""
    src = mat_i[attr]                       # [C,H,W]
    tgt = sample(mat_j[attr], grid)         # [C,H,W]
    diff = src - tgt
    err = torch.sqrt((diff ** 2).sum(dim=0).clamp(min=0))   # [H,W]
    return err, (diff ** 2).sum(dim=0)      # err map, and squared-sum for RMSE


def normal_error_map(mat_i, mat_j, grid, Ri, Rj, normal_space):
    """Per-pixel angular error [H,W] (degrees) between reprojected normals."""
    eps = 1e-6
    n_i = mat_i["normal"]                                  # [3,H,W]
    n_j = sample(mat_j["normal"], grid)                    # [3,H,W]
    # to [H,W,3]
    n_i = n_i.permute(1, 2, 0)
    n_j = n_j.permute(1, 2, 0)
    n_i = F.normalize(n_i, dim=-1, eps=eps)
    n_j = F.normalize(n_j, dim=-1, eps=eps)
    if normal_space == "camera":
        n_i = n_i @ Ri    # cam_i → world  (= R_i^T n)
        n_j = n_j @ Rj    # cam_j → world
    cos = (n_i * n_j).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.arccos(cos))                 # [H,W]
    return ang


# ==============================================================================
# Visualization
# ==============================================================================

def colorize(err_map, valid, vmax):
    e = err_map.detach().cpu().numpy().copy()
    m = valid.detach().cpu().numpy()
    e = np.clip(e / max(vmax, 1e-6), 0, 1)
    e[~m] = 0
    cm = cv2.applyColorMap((e * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cm[~m] = (0, 0, 0)   # invalid → black
    return cm


def save_pair_errmaps(out_scene, i, j, attr, err_vggt, err_mv, valid):
    folder = os.path.join(out_scene, "errmaps")
    os.makedirs(folder, exist_ok=True)
    vmax = VMAX[attr]
    a = colorize(err_vggt, valid, vmax)
    b = colorize(err_mv, valid, vmax)
    sep = np.full((a.shape[0], 4, 3), 255, np.uint8)
    canvas = np.concatenate([a, sep, b], axis=1)   # left=vggt, right=mvinverse
    cv2.imwrite(os.path.join(folder, f"pair{i:03d}_{j:03d}_{attr}.png"), canvas)


# ==============================================================================
# Aggregation containers
# ==============================================================================

def new_accumulator(methods):
    acc = {}
    for m in methods:
        acc[m] = {}
        for a in MATERIAL_ATTRS:
            acc[m][a] = {"sum_sq": 0.0, "elems": 0}
        acc[m]["normal"] = {"ang_sum": 0.0, "count": 0, "n1125": 0, "n30": 0}
    return acc


def merge(dst, src):
    for m in src:
        for a in MATERIAL_ATTRS:
            dst[m][a]["sum_sq"] += src[m][a]["sum_sq"]
            dst[m][a]["elems"] += src[m][a]["elems"]
        dst[m]["normal"]["ang_sum"] += src[m]["normal"]["ang_sum"]
        dst[m]["normal"]["count"] += src[m]["normal"]["count"]
        dst[m]["normal"]["n1125"] += src[m]["normal"]["n1125"]
        dst[m]["normal"]["n30"] += src[m]["normal"]["n30"]


def finalize(acc):
    """Convert accumulators → human-readable metrics dict."""
    out = {}
    for m in acc:
        out[m] = {}
        for a in MATERIAL_ATTRS:
            e = acc[m][a]
            out[m][a] = {
                "rmse": float(np.sqrt(e["sum_sq"] / e["elems"])) if e["elems"] else None,
                "num_px": int(e["elems"] // max(ATTRS[a], 1)),
            }
        nrm = acc[m]["normal"]
        c = nrm["count"]
        out[m]["normal"] = {
            "mean_angular_deg": float(nrm["ang_sum"] / c) if c else None,
            "pct_within_11.25": float(100.0 * nrm["n1125"] / c) if c else None,
            "pct_within_30": float(100.0 * nrm["n30"] / c) if c else None,
            "num_px": int(c),
        }
    return out


# ==============================================================================
# Per-scene processing
# ==============================================================================

def process_scene(scene_dir, out_scene, args, device):
    with open(os.path.join(scene_dir, "meta.json")) as f:
        meta = json.load(f)
    N, H, W = meta["num_views"], meta["H"], meta["W"]
    methods = meta.get("methods", ["vggt", "mvinverse"])
    normal_space = meta.get("normal_space", "camera")

    # view pairs
    if args.all_pairs:
        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    else:
        pairs = [(i, i + 1) for i in range(N - 1)]

    # pre-load geometry + materials for all frames
    geo = [load_geometry(scene_dir, s, device) for s in range(N)]
    mats = {m: [load_materials(scene_dir, m, s, device) for s in range(N)]
            for m in methods}

    # per-frame confidence thresholds (percentile gate)
    if args.conf_percentile > 0:
        conf_thr = [torch.quantile(geo[s]["depth_conf"].flatten(),
                                   args.conf_percentile / 100.0).item()
                    for s in range(N)]
    else:
        conf_thr = [0.0] * N

    acc = new_accumulator(methods)
    vis_done = 0

    for (i, j) in pairs:
        grid, valid = build_correspondence(
            geo[i], geo[j], H, W, args.occ_thr, conf_thr[i], conf_thr[j], device)
        n_valid = int(valid.sum().item())
        if n_valid < args.min_valid_px:
            continue
        vmask = valid

        save_this = (not args.all_pairs) and (vis_done < args.vis_pairs)

        for m in methods:
            mi, mj = mats[m][i], mats[m][j]
            # material attrs
            for a in MATERIAL_ATTRS:
                if a not in mi or a not in mj:
                    continue
                _, sq = material_error_map(mi, mj, a, grid)
                acc[m][a]["sum_sq"] += float(sq[vmask].sum().item())
                acc[m][a]["elems"] += n_valid * ATTRS[a]
            # normal
            if "normal" in mi and "normal" in mj:
                ang = normal_error_map(mi, mj, grid, geo[i]["R"], geo[j]["R"],
                                       normal_space)
                ang_v = ang[vmask]
                acc[m]["normal"]["ang_sum"] += float(ang_v.sum().item())
                acc[m]["normal"]["count"] += n_valid
                acc[m]["normal"]["n1125"] += int((ang_v < 11.25).sum().item())
                acc[m]["normal"]["n30"] += int((ang_v < 30.0).sum().item())

        # error-map visualization (vggt vs mvinverse, side by side)
        if save_this and len(methods) >= 2:
            ma, mb = methods[0], methods[1]
            for a in MATERIAL_ATTRS:
                if a in mats[ma][i] and a in mats[mb][i]:
                    ea, _ = material_error_map(mats[ma][i], mats[ma][j], a, grid)
                    eb, _ = material_error_map(mats[mb][i], mats[mb][j], a, grid)
                    save_pair_errmaps(out_scene, i, j, a, ea, eb, vmask)
            if "normal" in mats[ma][i] and "normal" in mats[mb][i]:
                ea = normal_error_map(mats[ma][i], mats[ma][j], grid,
                                      geo[i]["R"], geo[j]["R"], normal_space)
                eb = normal_error_map(mats[mb][i], mats[mb][j], grid,
                                      geo[i]["R"], geo[j]["R"], normal_space)
                save_pair_errmaps(out_scene, i, j, "normal", ea, eb, vmask)
            vis_done += 1

    return finalize(acc), acc, meta


# ==============================================================================
# Reporting
# ==============================================================================

def print_table(scene, metrics, methods):
    logger.info(f"───── {scene} ─────")
    header = f"{'attr':10s} | " + " | ".join(f"{m:>12s}" for m in methods) + " | winner"
    logger.info(header)
    for a in MATERIAL_ATTRS:
        vals = {m: metrics[m][a]["rmse"] for m in methods}
        cells = " | ".join(f"{(vals[m] if vals[m] is not None else float('nan')):12.5f}"
                           for m in methods)
        win = _winner(vals, lower_is_better=True)
        logger.info(f"{a+' RMSE':10s} | {cells} | {win}")
    vals = {m: metrics[m]["normal"]["mean_angular_deg"] for m in methods}
    cells = " | ".join(f"{(vals[m] if vals[m] is not None else float('nan')):12.5f}"
                       for m in methods)
    win = _winner(vals, lower_is_better=True)
    logger.info(f"{'normal deg':10s} | {cells} | {win}")


def _winner(vals, lower_is_better=True):
    items = [(m, v) for m, v in vals.items() if v is not None]
    if not items:
        return "n/a"
    best = min(items, key=lambda x: x[1]) if lower_is_better else max(items, key=lambda x: x[1])
    return best[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True,
                    help="Folder written by predict_mipnerf360.py")
    ap.add_argument("--out_dir", default=None,
                    help="Where to write results (default: <pred_dir>/_consistency)")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Subset of scenes to evaluate (default: all in pred_dir)")
    ap.add_argument("--occ_thr", type=float, default=0.05,
                    help="Relative depth diff for the occlusion test (smaller = "
                         "stricter visibility). Default 0.05 (5%%).")
    ap.add_argument("--conf_percentile", type=float, default=10.0,
                    help="Drop correspondences whose VGGT depth-confidence is "
                         "below this per-frame percentile. 0 = disabled.")
    ap.add_argument("--min_valid_px", type=int, default=500,
                    help="Skip a pair with fewer valid correspondences than this.")
    ap.add_argument("--all_pairs", action="store_true",
                    help="Use all ordered view pairs instead of adjacent only.")
    ap.add_argument("--vis_pairs", type=int, default=3,
                    help="How many adjacent pairs per scene to dump error maps for.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.pred_dir, "_consistency")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    scene_names = args.scenes or sorted(
        d for d in os.listdir(args.pred_dir)
        if os.path.isdir(os.path.join(args.pred_dir, d))
        and os.path.exists(os.path.join(args.pred_dir, d, "meta.json"))
    )
    logger.info(f"Scenes: {scene_names}")

    all_results = {}
    overall_acc = None
    methods_ref = None

    for scene in scene_names:
        scene_dir = os.path.join(args.pred_dir, scene)
        out_scene = os.path.join(out_dir, scene)
        os.makedirs(out_scene, exist_ok=True)
        try:
            metrics, acc, meta = process_scene(scene_dir, out_scene, args, device)
        except Exception as e:
            logger.exception(f"[{scene}] failed: {e}")
            continue
        all_results[scene] = metrics
        methods_ref = meta.get("methods", ["vggt", "mvinverse"])
        if overall_acc is None:
            overall_acc = new_accumulator(methods_ref)
        merge(overall_acc, acc)
        print_table(scene, metrics, methods_ref)

    if overall_acc is None:
        logger.error("No scenes processed.")
        return

    overall = finalize(overall_acc)
    all_results["_overall"] = overall

    logger.info("=" * 70)
    logger.info("OVERALL (pixel-weighted across all scenes & pairs)")
    print_table("OVERALL", overall, methods_ref)

    # ── write JSON ───────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, "consistency_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── write CSV (flat) ──────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "consistency_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "metric", "attr"] + methods_ref + ["winner"])
        for scene, metrics in all_results.items():
            for a in MATERIAL_ATTRS:
                row_vals = {m: metrics[m][a]["rmse"] for m in methods_ref}
                w.writerow([scene, "rmse", a]
                           + [row_vals[m] for m in methods_ref]
                           + [_winner(row_vals, True)])
            nvals = {m: metrics[m]["normal"]["mean_angular_deg"] for m in methods_ref}
            w.writerow([scene, "mean_angular_deg", "normal"]
                       + [nvals[m] for m in methods_ref]
                       + [_winner(nvals, True)])

    logger.info(f"Wrote:\n  {os.path.join(out_dir, 'consistency_results.json')}\n"
                f"  {csv_path}\n  {out_dir}/<scene>/errmaps/*.png")


if __name__ == "__main__":
    main()

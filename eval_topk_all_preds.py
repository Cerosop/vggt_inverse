"""
eval_topk_all_preds.py
======================
For the top-3 highest-loss scenes recorded in ``loss_analysis/stats.json``
under ``albedo`` and ``shading`` for {hypersim, structured3d, interiorverse},
re-run inference and dump **all five head predictions** (albedo, metallic,
roughness, normal, shading) — not just the head that was used to rank the
scene.

This lets us inspect, e.g., the worst-albedo scene in hypersim AND see how
all the other inverse-rendering outputs look on the same frames.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python eval_topk_all_preds.py
"""

import os
import sys
import csv
import json
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))

from hydra import initialize, compose

# Reuse helpers from eval_scene_loss.py (same directory).
from eval_scene_loss import (
    HEAD_FILES, HEAD_NAMES,
    get_target_hw, sample_frames_deterministic,
    load_scene_batch, _to_bgr_uint8,
    load_model, CKPT_PATH,
    NUM_FRAMES, TOP_K, SEED,
)

# ── what to dump ───────────────────────────────────────────────────────────────
DATA_BASE        = "/train-data-3-hdd/cerosop/vggt"
LOSS_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "loss_analysis")
STATS_PATH       = os.path.join(LOSS_ANALYSIS_DIR, "stats.json")
OUTPUT_DIR       = os.path.join(PROJECT_ROOT, "loss_analysis_topk_all_preds")

DATASETS_OF_INTEREST = ["hypersim", "structured3d", "interiorverse"]
HEADS_TO_RANK_BY     = ["albedo", "shading"]   # scenes are picked from max3 of these
SPLITS               = ["train", "test"]

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Helpers
# ==============================================================================

def collect_target_scenes(stats: dict) -> List[dict]:
    """Build list of unique (dataset, split, scene) jobs from stats.json.

    Each job records: which (head, rank, loss) entries it appeared in, so the
    output folder name reflects that ranking.
    """
    # key = (dataset, split, scene)  ->  job dict
    jobs: Dict[Tuple[str, str, str], dict] = {}
    for ds in DATASETS_OF_INTEREST:
        for sp in SPLITS:
            for head in HEADS_TO_RANK_BY:
                entry = stats.get(ds, {}).get(sp, {}).get(head, {})
                if not entry:
                    continue
                for i, e in enumerate(entry.get("max3", [])):
                    sname, loss = e["scene"], e["loss"]
                    key = (ds, sp, sname)
                    rank_info = {"head": head, "rank": i + 1, "loss": loss}
                    if key not in jobs:
                        jobs[key] = {
                            "dataset": ds, "split": sp, "scene": sname,
                            "ranks": [rank_info],
                        }
                    else:
                        jobs[key]["ranks"].append(rank_info)
    return list(jobs.values())


def load_frame_indices_map(dataset: str, split: str) -> Dict[str, List[int]]:
    """Parse per_scene_{dataset}_{split}.csv → {scene_name: [frame_indices]}.

    The CSV was written by eval_scene_loss.py with column "frame_indices"
    containing ``";"``-joined integers (e.g. ``"6;7;8;9;10;11"``).  Using this
    lookup guarantees that the frames we render here are *exactly* the ones
    used to compute the loss recorded in stats.json.
    """
    csv_path = os.path.join(LOSS_ANALYSIS_DIR,
                            f"per_scene_{dataset}_{split}.csv")
    out: Dict[str, List[int]] = {}
    if not os.path.exists(csv_path):
        logger.warning(f"CSV not found: {csv_path}")
        return out
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sname = row.get("scene", "").strip()
            fi_str = row.get("frame_indices", "").strip()
            if not sname or not fi_str:
                continue
            try:
                out[sname] = [int(x) for x in fi_str.split(";") if x != ""]
            except ValueError:
                logger.warning(f"Bad frame_indices for {dataset}/{split}/{sname}: {fi_str}")
    return out


def scene_avail_gts(scene_dir: str, frame_id_str: str) -> Dict[str, bool]:
    """Return per-head availability dict for a scene using its first frame."""
    fd = os.path.join(scene_dir, frame_id_str)
    return {h: os.path.exists(os.path.join(fd, HEAD_FILES[h][0]))
            for h in HEAD_NAMES}


def resolve_scene_dir(dataset: str, split: str, scene: str) -> Tuple[str, List[str]]:
    """Return (scene_dir, sorted_frame_ids) for a scene."""
    sdir = os.path.join(DATA_BASE, f"processed_data_{dataset}", split, scene)
    if not os.path.isdir(sdir):
        raise FileNotFoundError(sdir)
    fids = sorted(
        [d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d))],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    if not fids:
        raise RuntimeError(f"No frames in {sdir}")
    return sdir, fids


def save_all_predictions(out_folder: str, batch: dict, preds: dict,
                         frame_indices: List[int]):
    """Write input + all-head GT/pred for every frame in the batch."""
    os.makedirs(out_folder, exist_ok=True)
    S = batch["images"].shape[1]
    for s in range(S):
        fi = frame_indices[s] if s < len(frame_indices) else s

        # input
        inp = batch["images"][0, s].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(out_folder, f"frame{fi:04d}_input.png"),
                    _to_bgr_uint8(inp))

        # all head predictions (always, even if no GT)
        for head in HEAD_NAMES:
            if head in preds:
                p = preds[head][0, s].float().cpu()
                if head == "normal":
                    p = (p + 1.0) / 2.0
                cv2.imwrite(
                    os.path.join(out_folder, f"frame{fi:04d}_pred_{head}.png"),
                    _to_bgr_uint8(p.numpy()),
                )

            # GT only if available
            gt_key = f"gt_{head}"
            if gt_key in batch:
                g = batch[gt_key][0, s].float().cpu()
                if head == "normal":
                    g = (g + 1.0) / 2.0
                cv2.imwrite(
                    os.path.join(out_folder, f"frame{fi:04d}_gt_{head}.png"),
                    _to_bgr_uint8(g.numpy()),
                )


# ==============================================================================
# Main
# ==============================================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29510")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(0)

    # ── load stats ─────────────────────────────────────────────────────────────
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"Expected stats.json at: {STATS_PATH}")
    with open(STATS_PATH) as f:
        stats = json.load(f)
    jobs = collect_target_scenes(stats)
    if not jobs:
        logger.error("No matching scenes found in stats.json — nothing to do.")
        return
    logger.info(f"Will process {len(jobs)} unique scenes (deduped across heads).")

    # ── load model (same path as eval_scene_loss.py) ──────────────────────────
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="inverse_rendering")
    logger.info(f"Loading checkpoint: {CKPT_PATH}")
    model = load_model(cfg)

    h, w = get_target_hw()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── preload frame_indices lookup from per_scene CSVs ──────────────────────
    # Python's hash() is salted per-process, so re-seeding by hash() won't
    # reproduce eval_scene_loss.py's frames.  The CSV is the source of truth.
    fi_lookup: Dict[Tuple[str, str], Dict[str, List[int]]] = {}
    for ds in DATASETS_OF_INTEREST:
        for sp in SPLITS:
            fi_lookup[(ds, sp)] = load_frame_indices_map(ds, sp)

    # ── process each scene ─────────────────────────────────────────────────────
    for job in jobs:
        ds, sp, scene = job["dataset"], job["split"], job["scene"]
        try:
            sdir, fids = resolve_scene_dir(ds, sp, scene)
        except Exception as e:
            logger.warning(f"[{ds}/{sp}/{scene}] cannot resolve scene dir: {e}")
            continue

        # Pull the EXACT frame indices used in eval_scene_loss.py (recorded in
        # the per-scene CSV).  Only fall back to a regenerated seed if missing.
        frame_indices = fi_lookup.get((ds, sp), {}).get(scene)
        if frame_indices is None:
            logger.warning(f"[{ds}/{sp}/{scene}] no CSV entry — falling back "
                           f"to deterministic resample (frames may differ).")
            seed = abs(hash((ds, sp, scene))) % (2**31)
            frame_indices = sample_frames_deterministic(fids, NUM_FRAMES, seed)
        avail = scene_avail_gts(sdir, fids[0])

        try:
            batch = load_scene_batch(sdir, fids, frame_indices, avail, h, w)
        except Exception as e:
            logger.warning(f"[{ds}/{sp}/{scene}] load failed: {e}")
            continue

        try:
            with torch.no_grad():
                preds = model(batch["images"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning(f"[{ds}/{sp}/{scene}] OOM, skipping")
            continue

        # Folder name encodes the head(s) and rank(s) it was top-K for, e.g.:
        #   high_albedo01_loss0.0827__shading03_loss0.0654__ai_029_003-cam_00
        # so a single scene shared across heads keeps both labels visible.
        ranks_sorted = sorted(job["ranks"], key=lambda r: (r["head"], r["rank"]))
        tag = "__".join(
            f"{r['head']}{r['rank']:02d}_loss{r['loss']:.4f}"
            for r in ranks_sorted
        )
        safe_scene = scene.replace("/", "_")
        out_folder = os.path.join(OUTPUT_DIR, ds, sp, f"high_{tag}__{safe_scene}")
        save_all_predictions(out_folder, batch, preds, frame_indices)

        with open(os.path.join(out_folder, "meta.json"), "w") as f:
            json.dump({
                "dataset": ds, "split": sp, "scene": scene,
                "frame_indices": frame_indices,
                "available_gts": [h_ for h_, v in avail.items() if v],
                "ranked_by": ranks_sorted,
                "ckpt": CKPT_PATH,
            }, f, indent=2)

        logger.info(
            f"[{ds}/{sp}] {scene}  →  "
            f"{', '.join(r['head']+'#'+str(r['rank']) for r in ranks_sorted)}  "
            f"saved {batch['images'].shape[1]} frames × 5 heads"
        )

        del batch, preds
        torch.cuda.empty_cache()

    logger.info(f"Done. All predictions saved under: {OUTPUT_DIR}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""
eval_scene_loss.py
==================
Evaluate per-scene inverse-rendering loss for a given checkpoint over
**all six active datasets** listed in `training/config/inverse_rendering.yaml`,
covering every scene in both ``train`` and ``test`` splits.

For each ``(dataset, split, head)`` combination we report:
    * mean / variance over all scenes
    * top-3 highest-loss scenes (with scene name + loss value)
    * top-3 lowest-loss scenes  (with scene name + loss value)

Heads evaluated (only computed when the head's GT file exists in the scene):
    albedo, metallic, roughness, normal, shading.

Outputs (under ``loss_analysis/``):
    stats.json
        Full statistics: per (dataset, split, head) {mean, var, count,
        max3 [(scene, loss), ...], min3 [(scene, loss), ...]}
    per_scene_{dataset}_{split}.csv
        One row per scene, columns = scene + each head's loss (NaN if N/A)
    {dataset}/{split}/{head}/{rank}_{scene}_loss{val}/
        Image triplets (input / gt / pred) for top-3 and bottom-3 scenes.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python eval_scene_loss.py
"""

import os
import sys
import csv
import json
import math
import random
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

# ── project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))

from hydra import initialize, compose
from hydra.utils import instantiate

# ── constants ──────────────────────────────────────────────────────────────────
CKPT_PATH   = "/train-data-3-hdd/cerosop/vggt/vggt_origin/vggt/logs_0610/inverse_rendering/ckpts/checkpoint.pt"
DATA_BASE   = "/train-data-3-hdd/cerosop/vggt"
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "loss_analysis_0610")
IMG_SIZE    = 518
PATCH_SIZE  = 14
NUM_FRAMES  = 6           # fixed frame count per scene (deterministic)
TOP_K       = 3
SEED        = 42
SAVE_VISUALS = True       # set False to skip image triplets (faster)
MAX_SCENES_PER_SPLIT = 500  # cap per (dataset, split); set to None for "all scenes"

# Datasets active in training/config/inverse_rendering.yaml (commented ones excluded).
DATASETS: List[str] = [
    "structured3d",
    "olbedo",
    "hypersim",
    "interiorverse",
    "matrixcity_normal",
    "matrixcity_unnormal",
]

# Heads to evaluate.  filename used to (a) check GT presence, (b) load GT.
HEAD_FILES: Dict[str, Tuple[str, int]] = {
    "albedo":    ("albedo.png",    3),
    "metallic":  ("metallic.png",  1),
    "roughness": ("roughness.png", 1),
    "normal":    ("normal.png",    3),
    "shading":   ("shading.png",   3),
}
HEAD_NAMES = list(HEAD_FILES.keys())

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Dataset I/O helpers (mirror training/data/datasets/inverse_rendering_dataset.py)
# ==============================================================================

def discover_scenes(split_dir: str) -> List[Tuple[str, str, List[str], Dict[str, bool]]]:
    """Return list of (scene_name, scene_dir, frame_ids, available_gts_dict).

    A scene is included if it has at least one frame and at least one of the
    head GT files exists in its first frame.
    """
    scenes = []
    if not os.path.isdir(split_dir):
        return scenes
    for sname in sorted(os.listdir(split_dir)):
        sdir = os.path.join(split_dir, sname)
        if not os.path.isdir(sdir):
            continue
        fids = sorted(
            [d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d))],
            key=lambda x: int(x) if x.isdigit() else x,
        )
        if not fids:
            continue
        first_dir = os.path.join(sdir, fids[0])
        avail = {h: os.path.exists(os.path.join(first_dir, HEAD_FILES[h][0]))
                 for h in HEAD_NAMES}
        if not any(avail.values()):
            continue
        scenes.append((sname, sdir, fids, avail))
    return scenes


def get_target_hw(img_size: int = IMG_SIZE, patch_size: int = PATCH_SIZE,
                  aspect: float = 1.0) -> Tuple[int, int]:
    short = int(img_size * aspect)
    if short % patch_size != 0:
        short = (short // patch_size) * patch_size
    return short, img_size  # H, W


def sample_frames_deterministic(fids: List[str], n: int, seed: int) -> List[int]:
    """Sample n consecutive frame indices, deterministic given seed.

    Mirrors `InverseRenderingDataset.get_data`'s frame-sampling logic: pads by
    repetition if scene has fewer than n frames.
    """
    rng = random.Random(seed)
    num_avail = len(fids)
    if num_avail >= n:
        start = rng.randint(0, num_avail - n)
        return list(range(start, start + n))
    base = list(range(num_avail))
    repeated = [rng.choice(base) for _ in range(n - num_avail)]
    return sorted(base + repeated)


def _load_rgb(path: str, h: int, w: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img  # uint8 [0,255]


def _load_gt_png(path: str, h: int, w: int, channels: int,
                 is_normal: bool = False) -> np.ndarray:
    if channels == 1:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return np.zeros((h, w, 1), dtype=np.float32)
        g = g.astype(np.float32) / 255.0
        g = g[:, :, np.newaxis]
    else:
        g = cv2.imread(path)
        if g is None:
            return np.zeros((h, w, channels), dtype=np.float32)
        g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if is_normal:
        g = g * 2.0 - 1.0
    if g.shape[0] != h or g.shape[1] != w:
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        if g.ndim == 2:
            g = g[:, :, np.newaxis]
    return g.astype(np.float32)


def load_scene_batch(scene_dir: str, fids: List[str], frame_indices: List[int],
                     avail: Dict[str, bool], h: int, w: int) -> dict:
    """Load a GPU batch for one scene.

    Returns a dict with:
        images:      [1, S, 3, H, W]   float32 [0,1]
        gt_<head>:   [1, S, H, W, C]   float32  (only for heads with GT)
        mask_<head>: [1, S, H, W]      bool     (True = valid pixel)
        frame_dirs:  list[str]
    """
    imgs, frame_dirs = [], []
    gts: Dict[str, list] = {h_: [] for h_ in HEAD_NAMES if avail[h_]}
    masks: Dict[str, list] = {h_: [] for h_ in HEAD_NAMES if avail[h_]}

    for fi in frame_indices:
        fd = os.path.join(scene_dir, fids[fi])
        frame_dirs.append(fd)
        imgs.append(_load_rgb(os.path.join(fd, "rgb.png"), h, w).astype(np.float32) / 255.0)

        # per-frame base mask (mask.png if exists, else all ones)
        mask_path = os.path.join(fd, "mask.png")
        if os.path.exists(mask_path):
            base_mask = _load_gt_png(mask_path, h, w, 1)  # [H,W,1] [0,1]
            base_mask = (base_mask[..., 0] > 0.5)         # [H,W] bool
        else:
            base_mask = np.ones((h, w), dtype=bool)

        for head_name in gts.keys():
            gt_file, ch = HEAD_FILES[head_name]
            gt_path = os.path.join(fd, gt_file)
            if os.path.exists(gt_path):
                gt_arr = _load_gt_png(gt_path, h, w, ch,
                                      is_normal=(head_name == "normal"))
                gts[head_name].append(gt_arr)
                masks[head_name].append(base_mask)
            else:
                # Pad with zeros and zero mask so this frame contributes nothing.
                gts[head_name].append(np.zeros((h, w, ch), dtype=np.float32))
                masks[head_name].append(np.zeros((h, w), dtype=bool))

    imgs_np = np.stack(imgs, axis=0)                                # [S,H,W,3]
    imgs_t  = torch.from_numpy(imgs_np).permute(0, 3, 1, 2)\
                .unsqueeze(0).cuda()                                # [1,S,3,H,W]
    batch = {"images": imgs_t, "frame_dirs": frame_dirs}

    for head_name, lst in gts.items():
        arr = np.stack(lst, axis=0)                                 # [S,H,W,C]
        batch[f"gt_{head_name}"] = torch.from_numpy(arr).unsqueeze(0).cuda()
        m = np.stack(masks[head_name], axis=0)                      # [S,H,W]
        batch[f"mask_{head_name}"] = torch.from_numpy(m).unsqueeze(0).cuda()
    return batch


# ==============================================================================
# Per-scene losses
# ==============================================================================

def _resize_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Bilinear-resize pred [B,S,Hp,Wp,C] -> [B,S,Hg,Wg,C] to match gt."""
    if pred.shape == gt.shape:
        return pred
    B, S, Hp, Wp, C = pred.shape
    _, _, Hg, Wg, _ = gt.shape
    p = pred.reshape(B * S, Hp, Wp, C).permute(0, 3, 1, 2)
    p = F.interpolate(p, size=(Hg, Wg), mode="bilinear", align_corners=False)
    return p.permute(0, 2, 3, 1).reshape(B, S, Hg, Wg, C)


def compute_scene_losses(preds: dict, batch: dict) -> Dict[str, float]:
    """For each head with GT present in the batch, compute:
        normal  -> mean(1 - cos(pred, gt))            (mask-aware)
        others  -> masked MSE
    Returns dict {head: float}.  Head is omitted if mask sum is zero.
    """
    out: Dict[str, float] = {}
    for head in HEAD_NAMES:
        if head not in preds:
            continue
        gt_key, m_key = f"gt_{head}", f"mask_{head}"
        if gt_key not in batch:
            continue
        pred = _resize_pred_to_gt(preds[head], batch[gt_key])
        gt   = batch[gt_key]
        mask = batch[m_key]                                  # [B,S,H,W] bool

        if mask.sum().item() < 1:
            continue

        if head == "normal":
            p_n = F.normalize(pred, p=2, dim=-1, eps=1e-8)
            g_n = F.normalize(gt,   p=2, dim=-1, eps=1e-8)
            cos = (p_n * g_n).sum(dim=-1)                    # [B,S,H,W]
            val = (1.0 - cos)[mask].mean()
        else:
            diff_sq = (pred - gt) ** 2                       # [B,S,H,W,C]
            m_exp = mask.unsqueeze(-1).expand_as(diff_sq)
            val = diff_sq[m_exp].mean()

        out[head] = float(val.item())
    return out


# ==============================================================================
# Visualisation
# ==============================================================================

def _to_bgr_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_scene_visuals(dataset: str, split: str, head: str, rank_str: str,
                       scene_name: str, loss_val: float, batch: dict,
                       preds: dict, frame_indices: List[int]):
    safe = scene_name.replace("/", "_")
    folder = os.path.join(OUTPUT_DIR, dataset, split, head,
                          f"{rank_str}_{safe}_loss{loss_val:.4f}")
    os.makedirs(folder, exist_ok=True)
    S = batch["images"].shape[1]
    for s in range(S):
        fi = frame_indices[s] if s < len(frame_indices) else s
        # input
        inp = batch["images"][0, s].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(folder, f"frame{fi:04d}_input.png"),
                    _to_bgr_uint8(inp))
        # gt
        gt_key = f"gt_{head}"
        if gt_key in batch:
            gt_t = batch[gt_key][0, s].float().cpu()
            if head == "normal":
                gt_t = (gt_t + 1.0) / 2.0
            cv2.imwrite(os.path.join(folder, f"frame{fi:04d}_gt_{head}.png"),
                        _to_bgr_uint8(gt_t.numpy()))
        # pred
        if head in preds:
            pred_t = preds[head][0, s].float().cpu()
            if head == "normal":
                pred_t = (pred_t + 1.0) / 2.0
            cv2.imwrite(os.path.join(folder, f"frame{fi:04d}_pred_{head}.png"),
                        _to_bgr_uint8(pred_t.numpy()))
    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump({"dataset": dataset, "split": split, "head": head,
                   "rank": rank_str, "scene": scene_name, "loss": loss_val,
                   "frame_indices": frame_indices}, f, indent=2)


# ==============================================================================
# Model loading (auto-detect LoRA ranks from checkpoint)
# ==============================================================================

def _infer_lora_ranks_from_ckpt(state_dict: dict, base_lora_rank: int = 16):
    lora_global_base_rank = base_lora_rank
    for k, t in state_dict.items():
        if "lora_global_blocks" in k and "lora_qkv.lora_A.weight" in k:
            lora_global_base_rank = t.shape[0]
            break
    frame_ranks: Dict[int, int] = {}
    for k, t in state_dict.items():
        if "lora_frame_blocks" in k and "lora_qkv.lora_A.weight" in k:
            parts = k.split(".")
            try:
                idx = int(parts[parts.index("lora_frame_blocks") + 1])
            except (ValueError, IndexError):
                continue
            frame_ranks[idx] = t.shape[0]
    if frame_ranks:
        all_ranks = sorted(set(frame_ranks.values()))
        if len(all_ranks) == 1:
            return all_ranks[0], lora_global_base_rank, 0, all_ranks[0]
        return min(all_ranks), lora_global_base_rank, \
               sum(1 for r in frame_ranks.values() if r == max(all_ranks)), \
               max(all_ranks)
    return base_lora_rank, lora_global_base_rank, 6, 64


def load_model(cfg):
    cfg.model.enable_light_token = False  # eval-only; SG outputs unused
    logger.info(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt["model"]
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    lr, gr, tl, tr = _infer_lora_ranks_from_ckpt(clean_sd, cfg.model.lora_rank)
    logger.info(f"Detected LoRA ranks → frame={lr}, global={gr}, "
                f"tail_layers={tl}, tail_rank={tr}")
    cfg.model.lora_rank             = lr
    cfg.model.lora_global_base_rank = gr
    cfg.model.lora_tail_layers      = tl
    cfg.model.lora_tail_rank        = tr

    model = instantiate(cfg.model, _recursive_=False)
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    logger.info(f"Checkpoint loaded — missing={len(missing)}, "
                f"unexpected={len(unexpected)}")
    return model.cuda().eval()


# ==============================================================================
# Per-dataset evaluation
# ==============================================================================

def evaluate_dataset_split(model, dataset: str, split: str,
                           scenes, h: int, w: int) -> List[dict]:
    """Run model on every scene; return list of per-scene records.
    Record schema: {scene, scene_dir, fids, frame_indices, losses{head:val}}
    """
    records: List[dict] = []
    for i, (sname, sdir, fids, avail) in enumerate(
            tqdm(scenes, desc=f"[{dataset}/{split}]")):
        # Deterministic frame sampling per (dataset, split, scene)
        seed = abs(hash((dataset, split, sname))) % (2**31)
        frame_indices = sample_frames_deterministic(fids, NUM_FRAMES, seed)
        try:
            batch = load_scene_batch(sdir, fids, frame_indices, avail, h, w)
        except Exception as e:
            logger.warning(f"[{dataset}/{split}] Skip {sname}: {e}")
            continue

        try:
            with torch.no_grad():
                preds = model(batch["images"])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning(f"[{dataset}/{split}] OOM on {sname}, skipping.")
            continue

        losses = compute_scene_losses(preds, batch)
        records.append({
            "scene": sname,
            "scene_dir": sdir,
            "fids": fids,
            "frame_indices": frame_indices,
            "losses": losses,
        })
        # release tensors
        del batch, preds
        if (i + 1) % 200 == 0:
            torch.cuda.empty_cache()
    return records


# ==============================================================================
# Stats and CSV
# ==============================================================================

def stats_with_topk(records: List[dict], head: str, k: int = TOP_K) -> dict:
    pairs = [(r["scene"], r["losses"][head])
             for r in records if head in r["losses"]]
    if not pairs:
        return {}
    vals = np.array([v for _, v in pairs], dtype=np.float64)
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    bottom = pairs_sorted[:k]
    top    = pairs_sorted[-k:][::-1]
    return {
        "count": int(vals.size),
        "mean":  float(vals.mean()),
        "var":   float(vals.var()),
        "max3":  [{"scene": s, "loss": float(v)} for s, v in top],
        "min3":  [{"scene": s, "loss": float(v)} for s, v in bottom],
    }


def write_per_scene_csv(records: List[dict], dataset: str, split: str):
    if not records:
        return
    path = os.path.join(OUTPUT_DIR, f"per_scene_{dataset}_{split}.csv")
    fieldnames = ["scene"] + HEAD_NAMES + ["frame_indices"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            row = {"scene": r["scene"],
                   "frame_indices": ";".join(str(i) for i in r["frame_indices"])}
            for h in HEAD_NAMES:
                row[h] = r["losses"].get(h, "")
            w.writerow(row)
    logger.info(f"  CSV saved → {path}")


# ==============================================================================
# Visualise top/bottom K
# ==============================================================================

def visualise_topk(model, dataset: str, split: str, head: str,
                   entries: List[dict], rank_prefix: str, h: int, w: int):
    for i, entry in enumerate(entries):
        rec = entry["rec"]
        loss_val = entry["loss"]
        rank_str = f"{rank_prefix}{i+1:02d}"
        try:
            avail = {head_: (os.path.exists(os.path.join(
                rec["scene_dir"], rec["fids"][rec["frame_indices"][0]],
                HEAD_FILES[head_][0]))) for head_ in HEAD_NAMES}
            batch = load_scene_batch(rec["scene_dir"], rec["fids"],
                                     rec["frame_indices"], avail, h, w)
            with torch.no_grad():
                preds = model(batch["images"])
        except Exception as e:
            logger.warning(f"[vis] skip {rec['scene']}: {e}")
            continue
        save_scene_visuals(dataset, split, head, rank_str, rec["scene"],
                           loss_val, batch, preds, rec["frame_indices"])
        del batch, preds


# ==============================================================================
# Main
# ==============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # init DDP-style process group (some VGGT internals expect it)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29509")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(0)

    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="inverse_rendering")
    model = load_model(cfg)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h, w = get_target_hw()

    # Aggregate results: results[dataset][split] = list[records]
    results: Dict[str, Dict[str, List[dict]]] = {}

    for ds in DATASETS:
        ds_dir = os.path.join(DATA_BASE, f"processed_data_{ds}")
        results[ds] = {}
        for split in ["train", "test"]:
            split_dir = os.path.join(ds_dir, split)
            scenes = discover_scenes(split_dir)
            logger.info(f"[{ds}/{split}] {len(scenes)} scenes discovered "
                        f"in {split_dir}")
            if not scenes:
                results[ds][split] = []
                continue
            # Cap to MAX_SCENES_PER_SPLIT with deterministic shuffling
            if MAX_SCENES_PER_SPLIT is not None and len(scenes) > MAX_SCENES_PER_SPLIT:
                rng = random.Random(abs(hash((ds, split, SEED))) % (2**31))
                scenes_shuffled = scenes[:]
                rng.shuffle(scenes_shuffled)
                scenes = sorted(scenes_shuffled[:MAX_SCENES_PER_SPLIT],
                                key=lambda x: x[0])
                logger.info(f"[{ds}/{split}] sampled down to {len(scenes)} scenes "
                            f"(cap={MAX_SCENES_PER_SPLIT})")
            records = evaluate_dataset_split(model, ds, split, scenes, h, w)
            results[ds][split] = records
            write_per_scene_csv(records, ds, split)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STATISTICS  (mean / var / top-3 / bottom-3 per dataset × split × head)")
    print("=" * 80)
    stats: Dict[str, dict] = {}
    for ds in DATASETS:
        stats[ds] = {}
        for split in ["train", "test"]:
            recs = results.get(ds, {}).get(split, [])
            stats[ds][split] = {}
            if not recs:
                continue
            print(f"\n[{ds} / {split}]  (#scenes = {len(recs)})")
            for h_ in HEAD_NAMES:
                s = stats_with_topk(recs, h_, TOP_K)
                stats[ds][split][h_] = s
                if not s:
                    continue
                print(f"  {h_:10s}  n={s['count']:5d}  "
                      f"mean={s['mean']:.5f}  var={s['var']:.6f}")
                print(f"    max3: " + "  ".join(
                    f"{e['scene']}={e['loss']:.5f}" for e in s["max3"]))
                print(f"    min3: " + "  ".join(
                    f"{e['scene']}={e['loss']:.5f}" for e in s["min3"]))

    stats_path = os.path.join(OUTPUT_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats JSON → {stats_path}")

    # ── Visualise top-3 / bottom-3 ────────────────────────────────────────────
    if SAVE_VISUALS:
        print("\n" + "=" * 80)
        print(f"SAVING TOP-{TOP_K} / BOTTOM-{TOP_K} VISUALS")
        print("=" * 80)
        for ds in DATASETS:
            for split in ["train", "test"]:
                recs = results.get(ds, {}).get(split, [])
                if not recs:
                    continue
                # Pre-build scene -> record map for lookup by name
                name_map = {r["scene"]: r for r in recs}
                for h_ in HEAD_NAMES:
                    s = stats[ds][split].get(h_, {})
                    if not s:
                        continue
                    top    = [{"rec": name_map[e["scene"]], "loss": e["loss"]}
                              for e in s["max3"]]
                    bottom = [{"rec": name_map[e["scene"]], "loss": e["loss"]}
                              for e in s["min3"]]
                    logger.info(f"[vis] {ds}/{split}/{h_}: top-{TOP_K}")
                    visualise_topk(model, ds, split, h_, top,
                                   rank_prefix="high_", h=h, w=w)
                    logger.info(f"[vis] {ds}/{split}/{h_}: bottom-{TOP_K}")
                    visualise_topk(model, ds, split, h_, bottom,
                                   rank_prefix="low_", h=h, w=w)

    print("\nDone.  Outputs in:", OUTPUT_DIR)
    print("  stats.json                            — full numerical stats")
    print("  per_scene_{dataset}_{split}.csv       — per-scene loss tables")
    print("  {dataset}/{split}/{head}/...          — top/bottom-3 visuals")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

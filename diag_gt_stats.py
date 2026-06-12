"""
diag_gt_stats.py
================
Q2 diagnostic: are roughness/metallic GTs actually concentrated near 0?

Scans roughness.png / metallic.png in datasets that *should* have these GTs
(interiorverse, matrixcity_normal, matrixcity_unnormal) and prints:
    mean, median, std,
    fraction > {0.05, 0.1, 0.3, 0.5},
    coarse histogram (10 bins).

If GT mean is ≤ ~0.1 and median ≈ 0, the model collapsing to "all black"
is partly a data-distribution problem (most pixels really are near-zero
for indoor scenes).  If GT shows healthy mid-range mass, "all black"
predictions are pure model collapse.

Usage:  python diag_gt_stats.py
"""

import os
import cv2
import random
import numpy as np

DATA_BASE = "/train-data-3-hdd/cerosop/vggt"
DATASETS_WITH_MR_GT = ["interiorverse", "matrixcity_normal", "matrixcity_unnormal"]
SCENES_PER_SPLIT = 200      # random sample
FRAMES_PER_SCENE = 5
SEED = 42

random.seed(SEED)


def gather(ds: str, split: str, filename: str):
    root = f"{DATA_BASE}/processed_data_{ds}/{split}"
    if not os.path.isdir(root):
        return None
    scenes = sorted(os.listdir(root))
    if len(scenes) > SCENES_PER_SPLIT:
        scenes = random.Random(SEED).sample(scenes, SCENES_PER_SPLIT)
    vals = []
    for s in scenes:
        sdir = f"{root}/{s}"
        if not os.path.isdir(sdir):
            continue
        fids = sorted(os.listdir(sdir))
        if not fids:
            continue
        take = fids if len(fids) <= FRAMES_PER_SCENE \
               else random.Random(hash((s, filename)) & 0xffff).sample(fids, FRAMES_PER_SCENE)
        for fid in take:
            p = f"{sdir}/{fid}/{filename}"
            if os.path.exists(p):
                im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    vals.append(im.astype(np.float32) / 255.0)
    if not vals:
        return None
    return np.concatenate([v.flatten() for v in vals])


def summarize(arr, label):
    if arr is None or len(arr) == 0:
        print(f"  {label:35s}  <no data>")
        return
    pct = lambda t: float((arr > t).mean())
    print(f"  {label:35s}  "
          f"n_px={len(arr):>10d}  "
          f"mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"std={arr.std():.4f}  "
          f">0.05={pct(0.05):.3f}  >0.1={pct(0.1):.3f}  "
          f">0.3={pct(0.3):.3f}  >0.5={pct(0.5):.3f}")
    hist, _ = np.histogram(arr, bins=np.linspace(0, 1, 11))
    hist_pct = hist / hist.sum()
    bar = "  hist [0->1]: " + " ".join(f"{p*100:5.1f}%" for p in hist_pct)
    print(bar)


def main():
    print("=" * 110)
    print("GT distribution health-check for roughness / metallic")
    print(f"Sampled {SCENES_PER_SPLIT} scenes × up to {FRAMES_PER_SCENE} frames per (dataset, split)")
    print("=" * 110)
    for ds in DATASETS_WITH_MR_GT:
        for split in ["train", "test"]:
            print(f"\n[{ds} / {split}]")
            for f in ["roughness.png", "metallic.png"]:
                arr = gather(ds, split, f)
                summarize(arr, f)


if __name__ == "__main__":
    main()

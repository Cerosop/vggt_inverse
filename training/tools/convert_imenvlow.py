#!/usr/bin/env python3
"""
Convert OpenRooms `imenvlow` HDR → compact per-pixel environment GT (correctly parsed).
=======================================================================================

WHY THIS EXISTS
---------------
OpenRooms `*_imenvlow_*.hdr` is a *spatially-varying* lighting map stored as a TILED grid:
    HDR shape (960, 2560, 3) = (120*8, 160*16, 3)
        spatial grid = 120 x 160   (one env map per image pixel)
        each tile     = 8 x 16      (envHeight x envWidth, the local hemisphere env)

The previous processing was WRONG:
  - `populate_openroomsff_env_maps.py`: `cv2.resize` the whole tiled grid to 1024x512
        -> blends across tile boundaries -> garbage (neither a panorama nor valid SVL).
  - `hdr_to_sg.py`: fit a single SG by treating the (960,2560) tiled grid as ONE
        equirectangular panorama -> fits to a meaningless "panorama".

This tool parses the tiles CORRECTLY into per-pixel env maps:
    (960, 2560, 3) -> (Hs, Ws, 8, 16, 3)      [Hs=120, Ws=160]
and saves them compactly (fp16 + zlib).  Each tile = the incoming light at that pixel,
in the OpenRooms local frame (pole ≈ surface normal; upper hemisphere).

OUTPUT (.npz, np.savez_compressed):
    env      : float16, shape (Hs, Ws, 8, 16, 3)   (radiance, or log1p(radiance) if --log1p)
    log1p    : bool scalar   (whether `env` is log1p-encoded)
    env_hw   : (8, 16)       (angular resolution)
    note     : str

USAGE
-----
Single file (quick test):
    python training/tools/convert_imenvlow.py \
        --input /train-data-3-hdd/cerosop/vggt/1_imenvlow_1.hdr \
        --output /tmp/env_pixel.npz --verify

Batch (walk processed dataset, mirror populate path mapping):
    python training/tools/convert_imenvlow.py \
        --openrooms-root /train-data-3-hdd/cerosop/vggt/OpenRooms_FF \
        --processed-root /train-data-3-hdd/cerosop/vggt/processed_data_openroomsff \
        --splits train test --workers 8 \
        [--downsample 2] [--log1p] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow EXR/HDR reading without OpenCL surprises.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

ENV_H, ENV_W = 8, 16  # OpenRooms per-pixel env angular resolution (elevation x azimuth)

_SAMPLE_RE = re.compile(r"^(?P<source>.+)_scene(?P<scene>\d{4})_(?P<sub>\d{2})_(?P<view>\d+)$")
_HDR_RE = re.compile(r"^(?P<view>\d+)_imenvlow_(?P<idx>\d+)\.hdr$")


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def load_hdr_rgb(path: str) -> np.ndarray:
    hdr = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if hdr is None:
        raise RuntimeError(f"Failed to read HDR: {path}")
    return cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB).astype(np.float32)


def parse_per_pixel_env(hdr: np.ndarray) -> np.ndarray:
    """(H, W, 3) tiled imenvlow -> (Hs, Ws, ENV_H, ENV_W, 3) per-pixel env.

    Layout: row = sh*ENV_H + eh,  col = sw*ENV_W + ew.
    """
    H, W, C = hdr.shape
    if C != 3 or H % ENV_H != 0 or W % ENV_W != 0:
        raise ValueError(
            f"Unexpected imenvlow shape {hdr.shape}; expected (k*{ENV_H}, m*{ENV_W}, 3)."
        )
    Hs, Ws = H // ENV_H, W // ENV_W
    # (Hs, ENV_H, Ws, ENV_W, 3) -> (Hs, Ws, ENV_H, ENV_W, 3)
    env = hdr.reshape(Hs, ENV_H, Ws, ENV_W, 3).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(env)


def spatial_downsample(env: np.ndarray, factor: int) -> np.ndarray:
    """Block-average over the spatial grid only (each block element is a full env).

    env: (Hs, Ws, ENV_H, ENV_W, 3) -> (Hs//f, Ws//f, ENV_H, ENV_W, 3).
    Averaging neighbouring pixels' envs = a coarser (valid) SVL; never mixes angular dims.
    """
    if factor <= 1:
        return env
    Hs, Ws = env.shape[:2]
    if Hs % factor != 0 or Ws % factor != 0:
        raise ValueError(f"spatial {Hs}x{Ws} not divisible by downsample factor {factor}")
    Hs2, Ws2 = Hs // factor, Ws // factor
    env = env.reshape(Hs2, factor, Ws2, factor, ENV_H, ENV_W, 3).mean(axis=(1, 3))
    return np.ascontiguousarray(env)


def convert_one(
    hdr_path: str,
    out_path: str,
    downsample: int = 1,
    log1p: bool = False,
    clip_percentile: float = 0.0,
) -> Tuple[Tuple[int, ...], float]:
    """Convert a single imenvlow HDR to compact per-pixel env .npz. Returns (shape, max)."""
    hdr = load_hdr_rgb(hdr_path)
    env = parse_per_pixel_env(hdr)                       # (Hs, Ws, 8, 16, 3) float32

    # Sanitize corrupt HDR pixels (we observed inf/nan in a few OpenRooms frames).
    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
    env = np.clip(env, 0.0, None)

    if clip_percentile and clip_percentile > 0:
        cap = float(np.percentile(env, clip_percentile))
        env = np.clip(env, 0.0, cap)

    env = spatial_downsample(env, downsample)

    if log1p:
        env = np.log1p(env)                              # HDR-friendly + compresses better

    env16 = env.astype(np.float16)
    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        env=env16,
        log1p=np.array(bool(log1p)),
        env_hw=np.array([ENV_H, ENV_W], dtype=np.int32),
        note=np.array("per-pixel OpenRooms imenvlow env; local hemisphere (pole~normal)"),
    )
    return env16.shape, float(env.max())


def load_env_pixel(path: str) -> np.ndarray:
    """Loader helper. Returns (Hs, Ws, ENV_H, ENV_W, 3) float32 radiance (decodes log1p)."""
    with np.load(path, allow_pickle=False) as d:
        env = d["env"].astype(np.float32)
        if bool(d["log1p"]):
            env = np.expm1(env)
    return env


# ---------------------------------------------------------------------------
# Batch over processed dataset (mirrors populate_openroomsff_env_maps.py mapping)
# ---------------------------------------------------------------------------
def parse_sample_name(name: str) -> Optional[Tuple[str, str, str, str]]:
    m = _SAMPLE_RE.match(name)
    return (m.group("source"), m.group("scene"), m.group("sub"), m.group("view")) if m else None


def sorted_hdr_candidates(scene_dir: Path, view_id: str) -> dict:
    out = {}
    prefix = f"{view_id}_imenvlow_"
    if not scene_dir.is_dir():
        return out
    for p in scene_dir.iterdir():
        if p.is_file() and p.name.startswith(prefix) and p.name.endswith(".hdr"):
            m = _HDR_RE.match(p.name)
            if m:
                out[int(m.group("idx"))] = p
    return out


def process_sample(sample_dir: Path, openrooms_root: Path, out_name: str,
                   downsample: int, log1p: bool, clip_p: float,
                   overwrite: bool, dry_run: bool) -> dict:
    res = {"wrote": 0, "skipped": 0, "no_src": 0, "warn": None}
    parsed = parse_sample_name(sample_dir.name)
    if parsed is None:
        return res
    source_set, scene_id, sub_id, view_id = parsed
    scene_dir = openrooms_root / source_set / f"scene{scene_id}_{sub_id}"
    cand = sorted_hdr_candidates(scene_dir, view_id)
    if not cand:
        res["no_src"] = 1
        return res

    for frame_dir in sorted([c for c in sample_dir.iterdir() if c.is_dir() and c.name.isdigit()],
                            key=lambda p: int(p.name)):
        out_path = frame_dir / out_name
        if out_path.exists() and not overwrite:
            res["skipped"] += 1
            continue
        hdr_idx = int(frame_dir.name) + 1          # frame k -> imenvlow index k+1
        if hdr_idx not in cand:
            continue
        if dry_run:
            res["wrote"] += 1
            continue
        try:
            convert_one(str(cand[hdr_idx]), str(out_path), downsample, log1p, clip_p)
            res["wrote"] += 1
        except Exception as exc:
            res["warn"] = f"[WARN] {out_path}: {exc}"
    return res


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Convert OpenRooms imenvlow -> compact per-pixel env GT")
    # single-file mode
    ap.add_argument("--input", type=str, help="single imenvlow .hdr")
    ap.add_argument("--output", type=str, help="output .npz (single-file mode)")
    ap.add_argument("--verify", action="store_true", help="print parsed output stats")
    # batch mode
    ap.add_argument("--openrooms-root", type=Path,
                    default=Path("/train-data-3-hdd/cerosop/vggt/OpenRooms_FF"))
    ap.add_argument("--processed-root", type=Path,
                    default=Path("/train-data-3-hdd/cerosop/vggt/processed_data_openroomsff"))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--output-name", default="env_pixel.npz")
    # conversion options
    ap.add_argument("--downsample", type=int, default=1, help="spatial block-average factor (1=keep 120x160)")
    ap.add_argument("--log1p", action="store_true", help="store log1p(radiance) (HDR-friendly, smaller)")
    ap.add_argument("--clip-percentile", type=float, default=0.0, help="clip values above this percentile (0=off)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    # ---- single-file mode ----
    if args.input:
        assert args.output, "--output required in single-file mode"
        shape, mx = convert_one(args.input, args.output, args.downsample, args.log1p, args.clip_percentile)
        sz = osp.getsize(args.output) / 1e6
        print(f"[OK] {args.input} -> {args.output}  shape={shape}  max={mx:.2f}  size={sz:.2f} MB")
        if args.verify:
            env = load_env_pixel(args.output)
            print(f"  decoded env: shape={env.shape} min={env.min():.3f} max={env.max():.3f} mean={env.mean():.3f}")
            print(f"  one pixel env[0,0] (8x16 luminance):")
            print(np.round(env[0, 0].sum(-1), 2))
        return

    # ---- batch mode ----
    totals = {"wrote": 0, "skipped": 0, "no_src": 0}
    for split in args.splits:
        split_dir = args.processed_root / split
        if not split_dir.is_dir():
            print(f"[WARN] missing split dir: {split_dir}")
            continue
        samples = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
        print(f"[INFO] split={split} samples={len(samples)} workers={args.workers}", flush=True)

        def _run(s):
            return process_sample(s, args.openrooms_root, args.output_name,
                                  args.downsample, args.log1p, args.clip_percentile,
                                  args.overwrite, args.dry_run)

        if args.workers <= 1:
            it = (_run(s) for s in samples)
            for i, r in enumerate(it, 1):
                for k in totals: totals[k] += r[k]
                if r["warn"]: print(r["warn"], flush=True)
                if i % 200 == 0 or i == len(samples):
                    print(f"[PROGRESS] {split} {i}/{len(samples)} wrote={totals['wrote']}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_run, s) for s in samples]
                for i, f in enumerate(as_completed(futs), 1):
                    r = f.result()
                    for k in totals: totals[k] += r[k]
                    if r["warn"]: print(r["warn"], flush=True)
                    if i % 200 == 0 or i == len(samples):
                        print(f"[PROGRESS] {split} {i}/{len(samples)} wrote={totals['wrote']}", flush=True)

    print("=== convert_imenvlow summary ===", flush=True)
    for k, v in totals.items():
        print(f"{k}={v}", flush=True)


if __name__ == "__main__":
    main()

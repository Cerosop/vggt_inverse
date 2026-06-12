"""
Hypersim diffuse_illumination.hdf5 → diffuse_illumination.npy
==============================================================

Offline preprocessor for Hypersim dataset.
Reads per-frame "diffuse_illumination" HDF5 files (stored at scene level in
original Hypersim, e.g. frame.0000.diffuse_illumination.hdf5) and converts
them to per-frame numpy files placed in the processed frame directories:

    processed_data_hypersim/train/SCENE/FRAME/diffuse_illumination.npy

Usage (batch):
    python training/tools/hypersim_extract_diffuse.py \
        --raw_data_dir  /path/to/original_hypersim_data \
        --processed_dir /train-data-3-hdd/cerosop/vggt/processed_data_hypersim

Hypersim raw layout assumed:
    <raw_data_dir>/
      ai_001_001/
        images/
          scene_cam_00_final_hdf5/
            frame.0000.diffuse_illumination.hdf5
            frame.0001.diffuse_illumination.hdf5
            ...
        _detail/
          cam_00/
            ...

The script also handles the case where .hdf5 files are already placed directly
inside the frame directories of the processed dataset.
"""

import argparse
import os
import os.path as osp
import numpy as np
import h5py
import cv2
from tqdm import tqdm


def load_hdf5_rgb(path: str) -> np.ndarray:
    """Load a 3-channel HDF5 image as float32 [H, W, 3]."""
    with h5py.File(path, "r") as f:
        data = f["dataset"][:]  # usually key 'dataset', shape [H, W, 3]
    return data.astype(np.float32)


def resize_to(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if arr.shape[0] == target_h and arr.shape[1] == target_w:
        return arr
    return cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def process_from_raw(raw_data_dir: str, processed_dir: str, overwrite: bool):
    """Extract diffuse_illumination from original Hypersim layout."""
    for split in ["train", "test"]:
        proc_split = osp.join(processed_dir, split)
        if not osp.isdir(proc_split):
            continue

        scene_names = sorted(os.listdir(proc_split))
        print(f"\n=== {split}: {len(scene_names)} scenes ===")

        for scene_cam in tqdm(scene_names):
            # scene_cam format: "ai_001_001-cam_00"
            parts = scene_cam.split("-")
            if len(parts) != 2:
                continue
            scene_id, cam_tag = parts            # e.g. "ai_001_001", "cam_00"
            cam_id = cam_tag                     # e.g. "cam_00"

            # Locate the HDF5 directory in raw data
            # Possible paths:
            hdf5_dir = osp.join(raw_data_dir, scene_id, "images",
                                 f"scene_{cam_id}_final_hdf5")
            if not osp.isdir(hdf5_dir):
                continue

            proc_scene = osp.join(proc_split, scene_cam)
            frame_dirs = sorted([
                d for d in os.listdir(proc_scene)
                if osp.isdir(osp.join(proc_scene, d))
            ], key=lambda x: int(x) if x.isdigit() else x)

            for frame_dir_name in frame_dirs:
                frame_dir = osp.join(proc_scene, frame_dir_name)
                out_path  = osp.join(frame_dir, "diffuse_illumination.npy")

                if osp.exists(out_path) and not overwrite:
                    continue

                # frame_dir_name is typically "0000", "0001", etc.
                frame_idx = frame_dir_name.zfill(4)
                hdf5_path = osp.join(hdf5_dir,
                                     f"frame.{frame_idx}.diffuse_illumination.hdf5")
                if not osp.exists(hdf5_path):
                    continue

                # Load the reference rgb.png to know the target size
                rgb_path = osp.join(frame_dir, "rgb.png")
                ref_shape = None
                if osp.exists(rgb_path):
                    ref = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
                    if ref is not None:
                        ref_shape = (ref.shape[0], ref.shape[1])

                try:
                    arr = load_hdf5_rgb(hdf5_path)     # [H, W, 3] float32
                    if ref_shape is not None:
                        arr = resize_to(arr, ref_shape[0], ref_shape[1])
                    np.save(out_path, arr)
                except Exception as e:
                    print(f"\n  [WARN] {hdf5_path}: {e}")


def process_already_placed(processed_dir: str, overwrite: bool):
    """Handle case where .hdf5 files are already in frame directories."""
    for split in ["train", "test"]:
        proc_split = osp.join(processed_dir, split)
        if not osp.isdir(proc_split):
            continue

        for scene_name in sorted(os.listdir(proc_split)):
            scene_dir = osp.join(proc_split, scene_name)
            if not osp.isdir(scene_dir):
                continue
            for frame_name in sorted(os.listdir(scene_dir)):
                frame_dir = osp.join(scene_dir, frame_name)
                if not osp.isdir(frame_dir):
                    continue
                out_path = osp.join(frame_dir, "diffuse_illumination.npy")
                if osp.exists(out_path) and not overwrite:
                    continue
                hdf5_path = osp.join(frame_dir, "diffuse_illumination.hdf5")
                if not osp.exists(hdf5_path):
                    continue
                try:
                    arr = load_hdf5_rgb(hdf5_path)
                    np.save(out_path, arr)
                    print(f"  Converted: {hdf5_path}")
                except Exception as e:
                    print(f"  [WARN] {hdf5_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Hypersim diffuse_illumination to numpy arrays")
    parser.add_argument("--processed_dir", type=str, required=True,
                        help="Root of processed Hypersim dataset")
    parser.add_argument("--raw_data_dir",  type=str, default=None,
                        help="Root of original Hypersim dataset (if HDF5 not already placed)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.raw_data_dir and osp.isdir(args.raw_data_dir):
        print(f"Mode: extract from raw Hypersim at {args.raw_data_dir}")
        process_from_raw(args.raw_data_dir, args.processed_dir, args.overwrite)
    else:
        print("Mode: convert .hdf5 files already in frame directories")
        process_already_placed(args.processed_dir, args.overwrite)


if __name__ == "__main__":
    main()

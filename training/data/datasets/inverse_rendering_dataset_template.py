# Inverse Rendering Dataset for training MVInverse-style heads on VGGT.
#
# Expected data layout:
#   DATA_DIR/
#     scene_001/
#       images/          # RGB images (*.png, *.jpg, *.exr)
#       albedo/          # GT albedo maps (optional, *.png or *.exr)
#       metallic/        # GT metallic maps (optional, *.png or *.exr)
#       roughness/       # GT roughness maps (optional, *.png or *.exr)
#       normal/          # GT normal maps (optional, *.png or *.exr)
#       shading/         # GT diffuse shading (optional, *.png or *.exr)
#     scene_002/
#       ...
#
# Each scene folder must have an 'images/' subfolder. The GT subfolders are
# optional — if a GT is missing, the corresponding loss is skipped.
#
# Images and GTs should be named identically (e.g., 0001.png) so they can be
# matched by filename stem.

import os
import os.path as osp
import logging
import random
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


GT_SUBFOLDERS = {
    "albedo": ("gt_albedo", 3),      # (batch_key, num_channels)
    "metallic": ("gt_metallic", 1),
    "roughness": ("gt_roughness", 1),
    "normal": ("gt_normal", 3),
    "shading": ("gt_shading", 3),
}


class InverseRenderingDataset(Dataset):
    """Dataset for inverse rendering with multi-view images and material GTs.

    Supports incomplete GTs: if a scene doesn't have e.g. 'metallic/' subfolder,
    the batch will simply not contain 'gt_metallic'.

    Args:
        common_conf: Shared config (img_size, patch_size, etc.)
        split: 'train' or 'val'
        DATA_DIR: Root directory of the inverse rendering dataset
        min_num_images: Minimum images per scene
        len_train: Virtual epoch length for training
        len_test:  Virtual epoch length for validation
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = None,
        min_num_images: int = 4,
        len_train: int = 50000,
        len_test: int = 5000,
    ):
        super().__init__()

        if DATA_DIR is None:
            raise ValueError("DATA_DIR must be specified")

        self.data_dir = DATA_DIR
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.training = getattr(common_conf, 'training', split == 'train')
        self.min_num_images = min_num_images
        self.len_train = len_train if self.training else len_test

        # Discover scenes
        self.scenes = []
        self.scene_data = {}

        if not osp.isdir(DATA_DIR):
            logging.warning(f"InverseRenderingDataset: DATA_DIR does not exist: {DATA_DIR}")
            return

        for scene_name in sorted(os.listdir(DATA_DIR)):
            scene_dir = osp.join(DATA_DIR, scene_name)
            img_dir = osp.join(scene_dir, "images")

            if not osp.isdir(img_dir):
                continue

            # Collect image files
            img_files = sorted([
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))
            ])

            if len(img_files) < min_num_images:
                continue

            # Check which GTs are available
            available_gts = {}
            for gt_name, (batch_key, channels) in GT_SUBFOLDERS.items():
                gt_dir = osp.join(scene_dir, gt_name)
                if osp.isdir(gt_dir):
                    available_gts[gt_name] = gt_dir

            self.scenes.append(scene_name)
            self.scene_data[scene_name] = {
                "img_dir": img_dir,
                "img_files": img_files,
                "available_gts": available_gts,
            }

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: InverseRenderingDataset size: {len(self.scenes)} scenes")
        logging.info(f"{status}: InverseRenderingDataset epoch length: {self.len_train}")

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Args:
            idx_N: Tuple of (seq_index, img_per_seq, aspect_ratio) from DynamicTorchDataset.
        """
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index,
            img_per_seq=img_per_seq,
            aspect_ratio=aspect_ratio,
        )

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """Load multi-view images and corresponding material GTs.

        Returns:
            dict with keys:
                - 'images': List of [H, W, 3] numpy arrays (RGB, float32, [0, 1])
                - 'seq_name': str
                - 'frame_num': int
                - 'gt_albedo', 'gt_metallic', etc.: List of [H, W, C] numpy arrays (optional)
                - 'available_gts': List of GT names present in this batch
        """
        if seq_index is None:
            seq_index = random.randint(0, len(self.scenes) - 1)
        else:
            seq_index = seq_index % len(self.scenes)

        scene_name = seq_name or self.scenes[seq_index]
        scene = self.scene_data[scene_name]

        img_files = scene["img_files"]
        num_available = len(img_files)

        if img_per_seq is None:
            img_per_seq = min(8, num_available)

        # Sample image indices
        if ids is None:
            replace = num_available < img_per_seq
            ids = np.random.choice(num_available, img_per_seq, replace=replace)

        # Determine target shape
        target_h, target_w = self._get_target_shape(aspect_ratio)

        images = []
        gt_data = {gt_name: [] for gt_name in scene["available_gts"]}

        for idx in ids:
            fname = img_files[idx]
            stem = osp.splitext(fname)[0]

            # Load image
            img_path = osp.join(scene["img_dir"], fname)
            img = self._load_image(img_path, target_h, target_w)
            images.append(img)

            # Load available GTs
            for gt_name, gt_dir in scene["available_gts"].items():
                gt_path = self._find_gt_file(gt_dir, stem)
                if gt_path is not None:
                    _, channels = GT_SUBFOLDERS[gt_name]
                    gt_map = self._load_gt(gt_path, target_h, target_w, channels)
                    gt_data[gt_name].append(gt_map)
                else:
                    # If a specific frame's GT is missing, use zeros
                    _, channels = GT_SUBFOLDERS[gt_name]
                    gt_data[gt_name].append(np.zeros((target_h, target_w, channels), dtype=np.float32))

        batch = {
            "seq_name": f"inverse_{scene_name}",
            "ids": ids,
            "frame_num": len(images),
            "images": images,
            "available_gts": list(scene["available_gts"].keys()),
        }

        # Add GT data to batch
        for gt_name, gt_list in gt_data.items():
            batch_key, _ = GT_SUBFOLDERS[gt_name]
            batch[batch_key] = gt_list

        return batch

    def _get_target_shape(self, aspect_ratio):
        """Calculate target (H, W) that is patch-aligned."""
        short_size = int(self.img_size * aspect_ratio)
        if short_size % self.patch_size != 0:
            short_size = (short_size // self.patch_size) * self.patch_size
        return short_size, self.img_size

    def _load_image(self, path, target_h, target_w):
        """Load and resize an image to [H, W, 3] in [0, 255]."""
        if path.endswith('.exr'):
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise FileNotFoundError(f"Failed to load: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Failed to load: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return img

    def _load_gt(self, path, target_h, target_w, channels):
        """Load a GT map to [H, W, C] float32 in [0, 1] (or [-1, 1] for normals with EXR)."""
        if path.endswith('.exr'):
            gt = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if gt is None:
                return np.zeros((target_h, target_w, channels), dtype=np.float32)
            if channels == 3:
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            elif channels == 1 and gt.ndim == 3:
                gt = gt[:, :, 0:1]
        else:
            if channels == 1:
                gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    return np.zeros((target_h, target_w, channels), dtype=np.float32)
                gt = gt.astype(np.float32) / 255.0
                gt = gt[:, :, np.newaxis]  # [H, W, 1]
            else:
                gt = cv2.imread(path)
                if gt is None:
                    return np.zeros((target_h, target_w, channels), dtype=np.float32)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if gt.shape[0] != target_h or gt.shape[1] != target_w:
            gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if gt.ndim == 2:
                gt = gt[:, :, np.newaxis]

        return gt.astype(np.float32)

    def _find_gt_file(self, gt_dir, stem):
        """Find a GT file matching the given stem in gt_dir."""
        for ext in ['.png', '.jpg', '.exr', '.jpeg']:
            candidate = osp.join(gt_dir, stem + ext)
            if osp.exists(candidate):
                return candidate
        return None

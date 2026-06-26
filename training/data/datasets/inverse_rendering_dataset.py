# Inverse Rendering Dataset for processed OpenRoomsFF data.
#
# Expected data layout:
#   DATA_DIR/
#     train/
#       mainDiffLight_xml1_scene0001_00_0/
#         0/
#           rgb.png          # Input image (RGB)
#           albedo.png       # GT albedo (RGB)
#           roughness.png    # GT roughness (grayscale)
#           normal.png       # GT normal (RGB)
#           depth.npy        # GT depth
#           pose.npy         # Camera extrinsic (4x4)
#           intrinsics.npy   # Camera intrinsic (3x3)
#           shading.png      # Shading info (physically integrated)
#           position.npy     # Position info
#         1/
#           ...
#       mainDiffLight_xml1_scene0001_00_1/
#         ...
#     test/
#       ...
#
# Each scene has numbered frame subfolders. Each frame has rgb.png + material GTs.
# Not all GTs need to exist — missing ones are skipped in loss.

import os
import os.path as osp
import logging
import random

import cv2
import numpy as np
from torch.utils.data import Dataset


# Per-pixel env GT shape produced by training/tools/convert_imenvlow.py.
# (Hs, Ws, env_h, env_w, 3) — must match the converter's --downsample/env grid.
# Default = OpenRooms imenvlow downsample 2 (120x160 -> 60x80), 8x16 env.
ENV_PIXEL_SHAPE = (60, 80, 8, 16, 3)

# TEST-ONLY fallback: when env_pixel.npz is not yet populated per-frame, set the
# env var VGGT_ENV_PIXEL_FALLBACK=/path/to/env_pixel.npz to feed that single file
# as gt_env_pixel for EVERY frame. Decoded once and (if needed) spatially
# block-averaged to ENV_PIXEL_SHAPE. Leave the env var unset for normal training.
_ENV_PIXEL_FALLBACK_CACHE = {}


def _load_env_pixel_fallback():
    """Return the cached fallback env tile (ENV_PIXEL_SHAPE) or None if unset."""
    path = os.environ.get("VGGT_ENV_PIXEL_FALLBACK", "")
    if not path:
        return None
    if path in _ENV_PIXEL_FALLBACK_CACHE:
        return _ENV_PIXEL_FALLBACK_CACHE[path]
    with np.load(path, allow_pickle=False) as _d:
        ep = _d["env"].astype(np.float32)
        if "log1p" in _d and bool(_d["log1p"]):
            ep = np.expm1(ep)
    # Block-average the spatial dims to ENV_PIXEL_SHAPE if they differ by an int factor.
    Hs, Ws = ep.shape[:2]
    tH, tW = ENV_PIXEL_SHAPE[:2]
    if (Hs, Ws) != (tH, tW) and Hs % tH == 0 and Ws % tW == 0:
        fh, fw = Hs // tH, Ws // tW
        ep = ep.reshape(tH, fh, tW, fw, *ep.shape[2:]).mean(axis=(1, 3))
    ep = np.ascontiguousarray(ep.astype(np.float32))
    logging.warning(f"[env_pixel] using FALLBACK {path} for all frames; shape {ep.shape}")
    _ENV_PIXEL_FALLBACK_CACHE[path] = ep
    return ep

# Maps: GT filename -> (batch_key, num_channels)
GT_FILES = {
    "albedo.png":    ("gt_albedo", 3),
    "roughness.png": ("gt_roughness", 1),
    "normal.png":    ("gt_normal", 3),
}

# Optional GT files that may not exist in all scenes
OPTIONAL_GT_FILES = {
    "metallic.png":  ("gt_metallic", 1),
    "shading.png":  ("gt_shading", 3),
    "mask.png":     ("gt_mask", 1),
    "sg.npy":       ("gt_sg", -1),           # SG GT: per-scene [num_lobes, 7]
    "env_map.npy":  ("gt_env_map", 3),       # Per-frame environment map for SG-vs-env supervision
    "diffuse_illumination.npy": ("gt_diffuse_illumination", 3),  # Hypersim per-frame diffuse irradiance
    "env_pixel.npz": ("gt_env_pixel", -1),   # Per-pixel env GT (Hs,Ws,8,16,3) for the d4rt lighting branch
}


class InverseRenderingDataset(Dataset):
    """Dataset for processed OpenRoomsFF inverse rendering data.

    Each scene folder contains numbered frame subfolders, each with
    rgb.png and material GT files.

    Args:
        common_conf: Shared config (img_size, patch_size, etc.)
        split: 'train' or 'test'
        DATA_DIR: Root directory containing train/ and test/ splits
        min_num_frames: Minimum frames per scene
        len_train: Virtual epoch length for training
        len_test:  Virtual epoch length for validation
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DATA_DIR: str = "/mnt/train-data-5-hdd/cerosop/vggt/processed_data_openroomsff",
        min_num_frames: int = 4,
        len_train: int = 50000,
        len_test: int = 5000,
    ):
        super().__init__()

        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.training = getattr(common_conf, 'training', split == 'train')
        if hasattr(common_conf, 'img_nums') and common_conf.img_nums:
            self.min_num_frames = min(common_conf.img_nums)
        else:
            self.min_num_frames = min_num_frames
        self.length = len_train if self.training else len_test
        self.dataset_name = osp.basename(DATA_DIR.rstrip('/')).replace('processed_data_', '')

        # Resolve split directory
        split_dir = osp.join(DATA_DIR, split)
        if not osp.isdir(split_dir):
            logging.warning(f"InverseRenderingDataset: split dir not found: {split_dir}")
            self.scenes = []
            self.scene_data = {}
            return

        # Discover scenes
        self.scenes = []
        self.scene_data = {}

        for scene_name in sorted(os.listdir(split_dir)):
            scene_dir = osp.join(split_dir, scene_name)
            if not osp.isdir(scene_dir):
                continue

            # Discover frame subfolders (named 0, 1, 2, ...)
            frame_ids = sorted([
                d for d in os.listdir(scene_dir)
                if osp.isdir(osp.join(scene_dir, d))
            ], key=lambda x: int(x) if x.isdigit() else x)

            if len(frame_ids) < min_num_frames:
                continue

            # Check which GT files are available (check first frame)
            first_frame_dir = osp.join(scene_dir, frame_ids[0])
            available_gts = {}
            for gt_file, (batch_key, channels) in {**GT_FILES, **OPTIONAL_GT_FILES}.items():
                if osp.exists(osp.join(first_frame_dir, gt_file)):
                    available_gts[gt_file] = (batch_key, channels)

            self.scenes.append(scene_name)
            self.scene_data[scene_name] = {
                "scene_dir": scene_dir,
                "frame_ids": frame_ids,
                "available_gts": available_gts,
            }

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: InverseRenderingDataset DATA_DIR: {DATA_DIR}")
        logging.info(f"{status}: InverseRenderingDataset scenes: {len(self.scenes)}")
        logging.info(f"{status}: InverseRenderingDataset epoch length: {self.length}")

    def __len__(self):
        return self.length

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
                - 'gt_albedo', 'gt_roughness', 'gt_normal', etc.: Lists of [H, W, C] arrays
                - 'available_gts': List of GT names present
        """
        if len(self.scenes) == 0:
            raise RuntimeError(f"InverseRenderingDataset: No scenes found for split index calculation. Check your DATA_DIR and split settings.")

        if seq_index is None:
            seq_index = random.randint(0, len(self.scenes) - 1)
        else:
            seq_index = seq_index % len(self.scenes)

        scene_name = seq_name or self.scenes[seq_index]
        scene = self.scene_data[scene_name]
        frame_ids = scene["frame_ids"]
        num_available = len(frame_ids)

        if img_per_seq is None:
            img_per_seq = min(8, num_available)

        # Sample frame indices
        if ids is None:
            if num_available >= img_per_seq:
                # 連續採樣 (continuous sampling)，確保每個片段被選中的機率相等
                start_idx = random.randint(0, num_available - img_per_seq)
                selected = list(range(start_idx, start_idx + img_per_seq))
            else:
                # 視角數不足 img_per_seq（但已經 >= img_nums 最小數字），這時取全部，再重複隨機採樣補足
                base_selected = list(range(num_available))
                num_needed = img_per_seq - num_available
                repeated = [random.choice(base_selected) for _ in range(num_needed)]
                selected = base_selected + repeated
                selected.sort()
        else:
            selected = ids

        # Target shape
        target_h, target_w = self._get_target_shape(aspect_ratio)

        images = []
        ALL_GTS = {**GT_FILES, **OPTIONAL_GT_FILES}
        gt_data = {gt_file: [] for gt_file in ALL_GTS}
        
        HEADS = {
            "albedo": "albedo.png",
            "roughness": "roughness.png",
            "normal": "normal.png",
            "metallic": "metallic.png",
            "shading": "shading.png",
        }
        mask_data = {f"mask_{h}": [] for h in HEADS}

        for idx in selected:
            frame_dir = osp.join(scene["scene_dir"], frame_ids[idx])

            # Load RGB image
            rgb_path = osp.join(frame_dir, "rgb.png")
            img = self._load_image(rgb_path, target_h, target_w)
            images.append(img)
            
            # Load base mask
            base_mask_path = osp.join(frame_dir, "mask.png")
            if "mask.png" in scene["available_gts"] and osp.exists(base_mask_path):
                base_mask = self._load_gt(base_mask_path, target_h, target_w, 1, "mask.png")
            else:
                base_mask = np.ones((target_h, target_w, 1), dtype=np.float32)
                
            # Populate head masks explicitly
            for head_name, gt_filename in HEADS.items():
                if gt_filename in scene["available_gts"] and osp.exists(osp.join(frame_dir, gt_filename)):
                    mask_data[f"mask_{head_name}"].append(base_mask)
                else:
                    mask_data[f"mask_{head_name}"].append(np.zeros((target_h, target_w, 1), dtype=np.float32))

            # Load GTs
            for gt_file, (batch_key, channels) in ALL_GTS.items():
                # sg.npy: per-frame SG parameters [24, 7]
                if gt_file == "sg.npy":
                    sg_path = osp.join(frame_dir, gt_file)
                    if gt_file in scene["available_gts"] and osp.exists(sg_path):
                        sg_val = np.load(sg_path).astype(np.float32)
                    else:
                        sg_val = np.zeros((24, 7), dtype=np.float32)
                    gt_data[gt_file].append(sg_val)
                    continue

                # diffuse_illumination.npy: per-frame pixel map
                if gt_file == "diffuse_illumination.npy":
                    di_path = osp.join(frame_dir, gt_file)
                    if gt_file in scene["available_gts"] and osp.exists(di_path):
                        di = np.load(di_path).astype(np.float32)
                        if di.ndim == 2:
                            di = di[:, :, np.newaxis]
                        # Resize to target
                        if di.shape[0] != target_h or di.shape[1] != target_w:
                            di = cv2.resize(di, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        if di.ndim == 2:
                            di = di[:, :, np.newaxis]
                        gt_data[gt_file].append(di)
                    else:
                        gt_data[gt_file].append(np.zeros((target_h, target_w, 3), dtype=np.float32))
                    continue

                # env_pixel.npz: per-pixel env GT (Hs,Ws,8,16,3), fp16 [+ log1p].
                if gt_file == "env_pixel.npz":
                    fallback = _load_env_pixel_fallback()
                    if fallback is not None:
                        gt_data[gt_file].append(fallback)
                        continue
                    ep_path = osp.join(frame_dir, gt_file)
                    if gt_file in scene["available_gts"] and osp.exists(ep_path):
                        with np.load(ep_path, allow_pickle=False) as _d:
                            ep = _d["env"].astype(np.float32)
                            if "log1p" in _d and bool(_d["log1p"]):
                                ep = np.expm1(ep)
                        if ep.shape != ENV_PIXEL_SHAPE:
                            ep = np.zeros(ENV_PIXEL_SHAPE, dtype=np.float32)  # shape mismatch -> skip
                    else:
                        ep = np.zeros(ENV_PIXEL_SHAPE, dtype=np.float32)
                    gt_data[gt_file].append(ep)
                    continue

                if gt_file in scene["available_gts"]:
                    gt_path = osp.join(frame_dir, gt_file)
                    if osp.exists(gt_path):
                        gt = self._load_gt(gt_path, target_h, target_w, channels, gt_file)
                        gt_data[gt_file].append(gt)
                    else:
                        gt_data[gt_file].append(
                            np.zeros((target_h, target_w, channels), dtype=np.float32)
                        )
                else:
                    gt_data[gt_file].append(
                        np.zeros((target_h, target_w, channels), dtype=np.float32)
                    )

        # Build available GT name list (e.g., ["albedo", "roughness", "normal"])
        available_gt_names = []
        for gt_file, (batch_key, _) in scene["available_gts"].items():
            name = gt_file.replace(".png", "").replace(".npy", "")
            available_gt_names.append(name)

        batch = {
            "seq_name":    f"{self.dataset_name}_{scene_name}",
            "dataset_name": self.dataset_name,  # e.g. "openroomsff" / "hypersim"
            "ids":         selected,
            "frame_num":   len(images),
            "images":      images,
            "available_gts": ",".join(available_gt_names),
        }

        # Add GT data
        for gt_file, gt_list in gt_data.items():
            batch_key, _ = ALL_GTS[gt_file]
            batch[batch_key] = gt_list
            
        for mask_key, mask_list in mask_data.items():
            batch[mask_key] = mask_list

        return batch

    def _get_target_shape(self, aspect_ratio):
        """Calculate target (H, W) that is patch-aligned."""
        short_size = int(self.img_size * aspect_ratio)
        if short_size % self.patch_size != 0:
            short_size = (short_size // self.patch_size) * self.patch_size
        return short_size, self.img_size

    def _load_image(self, path, target_h, target_w):
        """Load RGB image to [H, W, 3] in [0, 255]."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] != target_h or img.shape[1] != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return img

    def _load_gt(self, path, target_h, target_w, channels, gt_filename):
        """Load a GT map to [H, W, C] float32.

        - PNG images: loaded as uint8 and normalized to [0, 1]
        - normal.png: loaded as RGB [0, 1], but OpenRooms normals can be in [0, 1]
          range where 0.5 = zero; they will be converted to [-1, 1] range.
        """
        if path.endswith('.npy'):
            gt = np.load(path).astype(np.float32)
            if gt.ndim == 2:
                gt = gt[:, :, np.newaxis]
        elif path.endswith('.hdr'):
            # Load HDR environment map
            gt = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if gt is None:
                return np.zeros((target_h, target_w, channels), dtype=np.float32)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32)
            # We don't necessarily resize the HDR map here if we want to handle it in the loss,
            # but for simplicity, let's keep it consistent with other GTs for now.
            # However, if target_h/target_w are derived from img_size (518), 
            # it might be overkill for an env map.
        else:
            if channels == 1:
                gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    return np.zeros((target_h, target_w, channels), dtype=np.float32)
                gt = gt.astype(np.float32) / 255.0
                gt = gt[:, :, np.newaxis]
            else:
                gt = cv2.imread(path)
                if gt is None:
                    return np.zeros((target_h, target_w, channels), dtype=np.float32)
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # For normal maps: convert from [0,1] to [-1,1] if stored as PNG
        if gt_filename == "normal.png" and not path.endswith('.npy'):
            gt = gt * 2.0 - 1.0  # [0,1] -> [-1,1]

        if gt.shape[0] != target_h or gt.shape[1] != target_w:
            gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            if gt.ndim == 2:
                gt = gt[:, :, np.newaxis]

        return gt.astype(np.float32)

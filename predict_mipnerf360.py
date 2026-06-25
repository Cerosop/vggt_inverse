"""
predict_mipnerf360.py
=====================
Run BOTH inverse-rendering models on Mip-NeRF 360 scenes and dump their
predictions (+ the geometry needed for a multi-view consistency comparison)
to a folder that ``compare_mv_consistency.py`` later reads.

The two models compared:
  1. YOUR model  — VGGT + LoRA + InverseHeads (this repo, weight/checkpoint.pt)
  2. MVInverse   — /train-data-3-hdd/cerosop/vggt/mvinverse  (HF: maddog241/mvinverse)

Why this is a *fair* comparison
-------------------------------
* Both models receive the **identical** pre-processed image tensor (W=518,
  H patch-aligned to 14, range [0,1]).  Their outputs therefore live on the
  same pixel grid, so no resampling is needed before reprojection.
* The geometry used for reprojection (camera extrinsics/intrinsics + per-pixel
  depth) is taken **entirely from YOUR VGGT model** (it predicts depth + camera
  pose for free in eval mode).  Because the *same* geometry is used to warp both
  methods' material maps, any geometry error affects both equally — the
  consistency numbers reflect only the *materials'* cross-view agreement.
  This is exactly the "reprojection RMSE across views" protocol MVInverse uses
  (their Tab. 2 / Tab. 4).

Mip-NeRF 360 layout expected (standard release)::

    <MIPNERF360_ROOT>/
        bicycle/  bonsai/  counter/  garden/  kitchen/  room/  stump/ ...
            images/      images_2/   images_4/   images_8/
            poses_bounds.npy   sparse/0/ ...

We only need the RGB images (geometry comes from VGGT), so COLMAP poses are not
parsed.  Use ``--image_subdir images_4`` to save VRAM on the 4K originals.

Output layout produced here::

    <OUT_DIR>/<scene>/
        meta.json                         # frames, H, W, normal_space, ...
        input/frame_000.png ...           # the exact tensor fed to both models
        geometry/frame_000.npz ...        # extrinsic(3x4) intrinsic(3x3) depth(HxW)
                                          #   world_points(HxWx3) depth_conf point_conf
        vggt/frame_000.npz ...            # albedo metallic roughness normal shading (float)
        vggt_vis/frame_000_<attr>.png ... # quick-look PNGs
        mvinverse/frame_000.npz ...
        mvinverse_vis/frame_000_<attr>.png ...

Usage (single GPU, run from the repo root)::

    CUDA_VISIBLE_DEVICES=0 python predict_mipnerf360.py \
        --mipnerf360_root /path/to/360_v2 \
        --scenes bonsai room counter kitchen \
        --image_subdir images_4 \
        --num_views 8 --stride 10 --start 0 \
        --out_dir mipnerf360_preds

NOTE: This script only WRITES predictions; the consistency metrics are computed
by ``compare_mv_consistency.py`` on the output folder.
"""

import os
import sys
import json
import glob
import argparse
import logging

import cv2
import numpy as np
import torch
import torch.distributed as dist

# ── project paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MVINVERSE_ROOT = "/train-data-3-hdd/cerosop/vggt/mvinverse"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
sys.path.insert(0, MVINVERSE_ROOT)

from hydra import initialize, compose
from hydra.utils import instantiate

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("predict_mipnerf360")

# Attributes (heads) and their channel counts.
ATTRS = {"albedo": 3, "metallic": 1, "roughness": 1, "normal": 3, "shading": 3}


# ==============================================================================
# Image loading / preprocessing
# ==============================================================================

def patch_align(x: int, patch: int = 14) -> int:
    return max(patch, (int(round(x)) // patch) * patch)


def list_scene_images(scene_dir: str, image_subdir: str):
    """Return a sorted list of image paths for one scene."""
    img_dir = os.path.join(scene_dir, image_subdir)
    if not os.path.isdir(img_dir):
        # Fall back to a bare 'images' dir or the scene dir itself.
        for alt in ("images", ""):
            cand = os.path.join(scene_dir, alt)
            if os.path.isdir(cand):
                img_dir = cand
                break
    exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(img_dir, e)))
    return sorted(files)


def sample_indices(n_total: int, num_views: int, stride: int, start: int):
    """Pick `num_views` frame indices with a fixed stride, clamped to range."""
    if n_total == 0:
        return []
    # Shrink stride if the requested span overflows the available frames.
    while stride > 1 and start + (num_views - 1) * stride >= n_total:
        stride -= 1
    idxs = [min(start + i * stride, n_total - 1) for i in range(num_views)]
    # De-dup while preserving order (in case of clamping collisions).
    seen, out = set(), []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def load_images_tensor(paths, width: int = 518):
    """Load images → a single [N,3,H,W] float tensor in [0,1].

    W is fixed to `width`; H = patch-aligned(width * h0/w0).  The aspect ratio of
    the first image defines (H, W) for the whole scene so all views share a grid.
    """
    first = cv2.imread(paths[0])
    if first is None:
        raise FileNotFoundError(paths[0])
    h0, w0 = first.shape[:2]
    W = patch_align(width)
    H = patch_align(width * h0 / w0)
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32) / 255.0)
    arr = np.stack(imgs, axis=0)                      # [N,H,W,3]
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # [N,3,H,W]
    return t, H, W


# ==============================================================================
# Model loading
# ==============================================================================

def _infer_lora_ranks_from_ckpt(state_dict: dict, base_lora_rank: int = 16):
    """Auto-detect LoRA ranks from a checkpoint (mirrors eval_scene_loss.py)."""
    lora_global_base_rank = base_lora_rank
    for k, t in state_dict.items():
        if "lora_global_blocks" in k and "lora_qkv.lora_A.weight" in k:
            lora_global_base_rank = t.shape[0]
            break
    frame_ranks = {}
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
        return (min(all_ranks), lora_global_base_rank,
                sum(1 for r in frame_ranks.values() if r == max(all_ranks)),
                max(all_ranks))
    return base_lora_rank, lora_global_base_rank, 6, 64


def load_user_model(ckpt_path: str, device: str = "cuda"):
    """Instantiate YOUR VGGT+Inverse model and load the checkpoint.

    Mirrors eval_scene_loss.py: auto-detects LoRA ranks, disables the light
    token (SG outputs unused here), loads with strict=False.
    """
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="inverse_rendering")

    cfg.model.enable_light_token = False
    logger.info(f"[user] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Adapt instantiation to how the checkpoint was actually trained.
    #   • LoRA checkpoint   → enable LoRA, use the detected ranks.
    #   • full-finetune ckpt → no LoRA blocks (frame/global blocks were trained
    #     directly). Building identity-LoRA would only double inference memory.
    has_lora = any(("lora_frame_blocks" in k) or ("lora_global_blocks" in k)
                   for k in clean_sd)
    cfg.model.enable_lora = has_lora
    if has_lora:
        lr, gr, tl, tr = _infer_lora_ranks_from_ckpt(clean_sd, cfg.model.lora_rank)
        logger.info(f"[user] LoRA checkpoint → frame={lr}, global={gr}, "
                    f"tail_layers={tl}, tail_rank={tr}")
        cfg.model.lora_rank = lr
        cfg.model.lora_global_base_rank = gr
        cfg.model.lora_tail_layers = tl
        cfg.model.lora_tail_rank = tr
    else:
        logger.info("[user] full-finetune checkpoint (no LoRA keys) → enable_lora=False")

    model = instantiate(cfg.model, _recursive_=False)
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    logger.info(f"[user] loaded — missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device).eval()


def load_mvinverse_model(ckpt: str, device: str = "cuda"):
    """Load MVInverse from a local checkpoint path or a HF repo id."""
    from mvinverse.models.mvinverse import MVInverse
    if os.path.exists(ckpt):
        logger.info(f"[mvinverse] loading local checkpoint: {ckpt}")
        model = MVInverse().to(device).eval()
        weight = torch.load(ckpt, map_location=device, weights_only=False)
        weight = weight["model"] if "model" in weight else weight
        missing, unused = model.load_state_dict(weight, strict=False)
        logger.info(f"[mvinverse] loaded — missing={len(missing)}, unused={len(unused)}")
    else:
        logger.info(f"[mvinverse] loading from HF hub: {ckpt}")
        model = MVInverse.from_pretrained(ckpt).to(device).eval()
    return model


def load_geom_model(ckpt_path: str, device: str = "cuda"):
    """Load the ORIGINAL pretrained VGGT, used purely for reprojection geometry.

    IMPORTANT: the inverse-rendering checkpoint was full-finetuned on the VGGT
    backbone, which COLLAPSES its camera-pose / depth estimation (poses → ~0
    translation, depth → a narrow band). Using that geometry for reprojection
    is invalid. So we run the untouched pretrained VGGT to get reliable
    pose+depth in a single consistent frame, and warp BOTH methods' materials
    with it.
    """
    from vggt.models.vggt import VGGT
    model = VGGT(enable_camera=True, enable_depth=True, enable_point=True,
                 enable_track=True, enable_inverse=False, enable_lora=False,
                 enable_light_token=False)
    logger.info(f"[geom] loading pretrained VGGT: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[geom] loaded — missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device).eval()


# ==============================================================================
# Saving helpers
# ==============================================================================

def _to_uint8_rgb(arr: np.ndarray, is_normal: bool = False) -> np.ndarray:
    """[H,W,C] float → [H,W,3] uint8 BGR for cv2.imwrite (quick-look only)."""
    a = arr.copy()
    if is_normal:
        a = (a + 1.0) / 2.0
    a = np.clip(a, 0.0, 1.0)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    a = (a * 255).astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def save_material_npz(folder: str, frame_tag: str, preds_hwc: dict):
    """Save per-frame material maps (float32) into one .npz."""
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(
        os.path.join(folder, f"{frame_tag}.npz"),
        **{a: preds_hwc[a].astype(np.float32) for a in ATTRS if a in preds_hwc},
    )


def save_material_vis(folder: str, frame_tag: str, preds_hwc: dict):
    os.makedirs(folder, exist_ok=True)
    for a in ATTRS:
        if a not in preds_hwc:
            continue
        cv2.imwrite(os.path.join(folder, f"{frame_tag}_{a}.png"),
                    _to_uint8_rgb(preds_hwc[a], is_normal=(a == "normal")))


def extract_materials_bshwc(preds: dict) -> dict:
    """Pull the 5 material heads from a model output dict, as [S,H,W,C] numpy."""
    out = {}
    for a in ATTRS:
        if a in preds and preds[a] is not None:
            out[a] = preds[a][0].detach().float().cpu().numpy()  # [S,H,W,C]
    return out


# ==============================================================================
# Main per-scene routine
# ==============================================================================

def process_scene(scene_name, scene_dir, args, user_model, mv_model, geom_model, device):
    paths_all = list_scene_images(scene_dir, args.image_subdir)
    if len(paths_all) == 0:
        logger.warning(f"[{scene_name}] no images found under {args.image_subdir}; skipping")
        return
    idxs = sample_indices(len(paths_all), args.num_views, args.stride, args.start)
    paths = [paths_all[i] for i in idxs]
    logger.info(f"[{scene_name}] {len(paths_all)} imgs available → using "
                f"{len(paths)} views (idx={idxs})")

    imgs, H, W = load_images_tensor(paths, width=args.width)   # [N,3,H,W] in [0,1]
    imgs = imgs.to(device)
    N = imgs.shape[0]

    out_scene = os.path.join(args.out_dir, scene_name)
    os.makedirs(out_scene, exist_ok=True)

    # ── input dump (the exact tensor both models see) ────────────────────────
    in_dir = os.path.join(out_scene, "input")
    os.makedirs(in_dir, exist_ok=True)
    for s in range(N):
        rgb = imgs[s].permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(os.path.join(in_dir, f"frame_{s:03d}.png"),
                    _to_uint8_rgb(rgb))

    # ── Geometry model (pretrained VGGT) — reliable pose+depth ───────────────
    # The finetuned user model's geometry is collapsed, so we get geometry from
    # the untouched pretrained VGGT when available; otherwise fall back to the
    # user model (with a warning) for backward compatibility.
    geo_src = None
    if geom_model is not None:
        logger.info(f"[{scene_name}] running GEOMETRY model (pretrained VGGT) ...")
        with torch.no_grad():
            geo_src = geom_model(imgs[None])

    # ── YOUR model: materials ────────────────────────────────────────────────
    logger.info(f"[{scene_name}] running USER model (materials) ...")
    with torch.no_grad():
        preds = user_model(imgs[None])   # add batch dim → [1,N,3,H,W]

    user_mats = extract_materials_bshwc(preds)
    for s in range(N):
        save_material_npz(os.path.join(out_scene, "vggt"), f"frame_{s:03d}",
                          {a: user_mats[a][s] for a in user_mats})
        if args.save_vis:
            save_material_vis(os.path.join(out_scene, "vggt_vis"), f"frame_{s:03d}",
                              {a: user_mats[a][s] for a in user_mats})

    if geo_src is None:
        logger.warning(f"[{scene_name}] no geometry model → using (collapsed) "
                       f"user-model geometry. Reprojection will be unreliable.")
        geo_src = preds
    # geometry (depth head + camera head + point head)
    if "pose_enc" not in geo_src:
        raise RuntimeError(
            f"[{scene_name}] geometry output has no 'pose_enc' — geometry heads "
            f"did not run. Ensure eval() mode and camera/depth heads enabled.")
    extr, intr = pose_encoding_to_extri_intri(geo_src["pose_enc"], (H, W))  # [1,N,3,4],[1,N,3,3]
    extr = extr[0].detach().float().cpu().numpy()      # [N,3,4]
    intr = intr[0].detach().float().cpu().numpy()      # [N,3,3]
    depth = geo_src["depth"][0, ..., 0].detach().float().cpu().numpy()       # [N,H,W]
    wpts = geo_src["world_points"][0].detach().float().cpu().numpy()         # [N,H,W,3]
    dconf = (geo_src["depth_conf"][0].detach().float().cpu().numpy()
             if "depth_conf" in geo_src else np.ones((N, H, W), np.float32))
    pconf = (geo_src["world_points_conf"][0].detach().float().cpu().numpy()
             if "world_points_conf" in geo_src else np.ones((N, H, W), np.float32))

    geo_dir = os.path.join(out_scene, "geometry")
    os.makedirs(geo_dir, exist_ok=True)
    for s in range(N):
        np.savez_compressed(
            os.path.join(geo_dir, f"frame_{s:03d}.npz"),
            extrinsic=extr[s].astype(np.float32),     # world→cam [R|t], OpenCV
            intrinsic=intr[s].astype(np.float32),     # pixels, 3x3
            depth=depth[s].astype(np.float32),        # z-depth [H,W]
            world_points=wpts[s].astype(np.float32),  # [H,W,3]
            depth_conf=dconf[s].astype(np.float32),
            point_conf=pconf[s].astype(np.float32),
        )

    del preds, geo_src
    torch.cuda.empty_cache()

    # ── MVInverse: materials only ────────────────────────────────────────────
    logger.info(f"[{scene_name}] running MVInverse ...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            mv_preds = mv_model(imgs[None])   # [1,N,3,H,W]
    mv_mats = extract_materials_bshwc(mv_preds)
    for s in range(N):
        save_material_npz(os.path.join(out_scene, "mvinverse"), f"frame_{s:03d}",
                          {a: mv_mats[a][s] for a in mv_mats})
        if args.save_vis:
            save_material_vis(os.path.join(out_scene, "mvinverse_vis"), f"frame_{s:03d}",
                              {a: mv_mats[a][s] for a in mv_mats})
    del mv_preds
    torch.cuda.empty_cache()

    # ── meta ─────────────────────────────────────────────────────────────────
    meta = {
        "scene": scene_name,
        "scene_dir": scene_dir,
        "image_subdir": args.image_subdir,
        "frame_files": [os.path.basename(p) for p in paths],
        "frame_indices": idxs,
        "num_views": N,
        "H": int(H), "W": int(W),
        "attrs": list(ATTRS.keys()),
        "methods": ["vggt", "mvinverse"],
        # Both models predict normals in CAMERA space (MVInverse paper states
        # camera-space; OpenRooms-trained heads likewise). The comparison script
        # rotates them to world space before measuring agreement.
        "normal_space": "camera",
        "geometry_source": args.geom_source,   # "pretrained" (reliable) or "user" (collapsed)
        "extrinsic_convention": "world2cam_opencv_3x4",
    }
    with open(os.path.join(out_scene, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"[{scene_name}] done → {out_scene}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mipnerf360_root", required=True,
                    help="Root dir containing Mip-NeRF 360 scene folders.")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Scene names to process (default: all sub-dirs that "
                         "contain an image folder).")
    ap.add_argument("--image_subdir", default="images_4",
                    help="Which resolution sub-dir to read (images / images_2 / "
                         "images_4 / images_8). Default images_4.")
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "mipnerf360_preds"))
    ap.add_argument("--user_ckpt",
                    default=os.path.join(PROJECT_ROOT, "weight", "checkpoint.pt"))
    ap.add_argument("--mvinverse_ckpt", default="maddog241/mvinverse",
                    help="Local path OR HF repo id for MVInverse weights.")
    ap.add_argument("--geom_source", choices=["pretrained", "user"], default="pretrained",
                    help="Where reprojection geometry comes from. 'pretrained' "
                         "(default) = untouched VGGT (reliable). 'user' = the "
                         "finetuned model (geometry is collapsed → unreliable).")
    ap.add_argument("--geom_ckpt",
                    default=os.path.join(PROJECT_ROOT, "weight", "vggt_1b_pretrained.pt"),
                    help="Pretrained VGGT checkpoint used for geometry.")
    ap.add_argument("--num_views", type=int, default=8)
    ap.add_argument("--stride", type=int, default=10,
                    help="Frame stride for sampling consecutive-ish views.")
    ap.add_argument("--start", type=int, default=0, help="First frame index.")
    ap.add_argument("--width", type=int, default=518,
                    help="Target image width (height derives from aspect, "
                         "patch-aligned to 14). 518 = VGGT native.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_vis", action="store_true", default=True,
                    help="Also dump PNG quick-look images.")
    ap.add_argument("--no_save_vis", dest="save_vis", action="store_false")
    args = ap.parse_args()

    # DDP single-process init (some VGGT internals expect a process group).
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29511")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    if not dist.is_initialized():
        try:
            dist.init_process_group("nccl")
        except Exception as e:
            logger.warning(f"dist init failed ({e}); continuing without it.")
    if args.device.startswith("cuda"):
        torch.cuda.set_device(0)

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve scenes
    root = args.mipnerf360_root
    if args.scenes:
        scene_names = args.scenes
    else:
        scene_names = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        )
    logger.info(f"Scenes to process: {scene_names}")

    # Load models once
    user_model = load_user_model(args.user_ckpt, args.device)
    mv_model = load_mvinverse_model(args.mvinverse_ckpt, args.device)
    geom_model = (load_geom_model(args.geom_ckpt, args.device)
                  if args.geom_source == "pretrained" else None)

    for scene_name in scene_names:
        scene_dir = os.path.join(root, scene_name)
        if not os.path.isdir(scene_dir):
            logger.warning(f"[{scene_name}] not a directory; skipping")
            continue
        try:
            process_scene(scene_name, scene_dir, args, user_model, mv_model,
                          geom_model, args.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error(f"[{scene_name}] OOM — try fewer --num_views or a smaller "
                         f"--image_subdir / --width. Skipping.")
        except Exception as e:
            logger.exception(f"[{scene_name}] failed: {e}")

    logger.info("All scenes done.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

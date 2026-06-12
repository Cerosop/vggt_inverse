"""
HDR Environment Map → Spherical Gaussian (SG) Fitting Tool
===========================================================

Offline preprocessor for OpenRoomsFF dataset.

Usage (single file):
    python training/tools/hdr_to_sg.py \
        --input /path/to/scene_env.hdr \
        --output /path/to/scene_dir/sg.npy \
        --lobes 24

Usage (batch — process all scenes in DATA_DIR):
    python training/tools/hdr_to_sg.py \
        --data_dir /train-data-3-hdd/cerosop/vggt/processed_data_openroomsff \
        --hdr_name imenvlow.hdr \
        --lobes 24 \
        --device cuda

Output:
    sg.npy  — shape [24, 7], float32
               [:, :3] directions (L2-normalized)
               [:, 3]  sharpness  (1~1000)
               [:, 4:] amplitude  (>= 0, RGB)
"""

import argparse
import os
import os.path as osp
import glob
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Spherical Gaussian helpers
# ---------------------------------------------------------------------------

def equirect_to_directions(H, W, device="cpu"):
    """Return [H*W, 3] unit direction vectors for an equirectangular map."""
    theta = torch.linspace(0, torch.pi, H, device=device)          # polar
    phi   = torch.linspace(0, 2 * torch.pi, W, device=device)      # azimuth
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")          # [H, W]

    x = torch.sin(theta) * torch.cos(phi)
    y = torch.cos(theta)
    z = torch.sin(theta) * torch.sin(phi)
    dirs = torch.stack([x, y, z], dim=-1)                          # [H, W, 3]
    return dirs.reshape(-1, 3)                                      # [H*W, 3]


def sg_eval(dirs, sg_params):
    """Evaluate SG lobes at given directions.

    Args:
        dirs:      [N, 3] unit direction vectors
        sg_params: [K, 7] — dir3, sharpness1, amplitude3

    Returns:
        [N, 3] RGB radiance
    """
    K = sg_params.shape[0]
    mu  = F.normalize(sg_params[:, :3], dim=-1)          # [K, 3]
    lam = sg_params[:, 3:4]                               # [K, 1]  sharpness
    col = sg_params[:, 4:]                                # [K, 3]  amplitude

    # dot product: [N, K]
    dot = dirs @ mu.T                                     # [N, K]
    w   = torch.exp(lam.T * (dot - 1.0))                 # [N, K]

    radiance = (w.unsqueeze(-1) * col.unsqueeze(0)).sum(dim=1)  # [N, 3]
    return radiance


def fit_sg_from_hdr(
    hdr: np.ndarray,
    num_lobes: int = 24,
    num_iters: int = 2000,
    lr: float = 0.05,
    device: str = "cuda",
    verbose: bool = False,
):
    """Fit SG lobes to an equirectangular HDR environment map.

    Args:
        hdr:       [H, W, 3] float32 HDR image (linear, any scale)
        num_lobes: Number of SG lobes
        num_iters: Gradient descent iterations
        lr:        Adam learning rate
        device:    "cuda" or "cpu"

    Returns:
        sg_np: [num_lobes, 7] float32 numpy array with fitted SG params
    """
    H, W = hdr.shape[:2]
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    # Solid-angle weighting: sin(theta) for equirectangular projection
    theta = torch.linspace(0, torch.pi, H, device=device)
    sin_weights = torch.sin(theta)[:, None].expand(H, W).reshape(-1)  # [H*W]
    sin_weights = sin_weights / sin_weights.sum()  # normalize

    dirs   = equirect_to_directions(H, W, device=device)  # [H*W, 3]
    target = torch.from_numpy(hdr.reshape(-1, 3)).to(device)          # [H*W, 3]

    # Initialize SG parameters
    # directions: uniformly spread on sphere (Fibonacci lattice)
    idx = torch.arange(num_lobes, device=device, dtype=torch.float32)
    golden = (1 + 5 ** 0.5) / 2
    theta0 = torch.acos(1 - 2 * idx / num_lobes)
    phi0   = (2 * torch.pi * idx / golden) % (2 * torch.pi)
    init_dirs = torch.stack([
        torch.sin(theta0) * torch.cos(phi0),
        torch.cos(theta0),
        torch.sin(theta0) * torch.sin(phi0),
    ], dim=-1)  # [K, 3]

    # Raw (unconstrained) parameters
    raw_dir   = init_dirs.clone().requires_grad_(True)
    raw_lam   = torch.full((num_lobes, 1), 2.0, device=device, requires_grad=True)   # log sharpness
    raw_col   = torch.zeros((num_lobes, 3), device=device, requires_grad=True)       # log amplitude

    optimizer = torch.optim.Adam([raw_dir, raw_lam, raw_col], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=lr/100)

    for i in range(num_iters):
        optimizer.zero_grad()

        # Constrained parameters
        mu  = F.normalize(raw_dir, dim=-1)
        lam = 1.0 + 999.0 * torch.sigmoid(raw_lam)   # [1, 1000]
        col = F.softplus(raw_col)                     # non-negative

        sg_params = torch.cat([mu, lam, col], dim=-1)  # [K, 7]
        rendered  = sg_eval(dirs, sg_params)            # [H*W, 3]

        # Weighted MSE (weight by solid angle)
        diff = (rendered - target) ** 2               # [H*W, 3]
        loss = (diff.mean(dim=-1) * sin_weights).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_dir, raw_lam, raw_col], 1.0)
        optimizer.step()
        scheduler.step()

        if verbose and (i % 200 == 0 or i == num_iters - 1):
            print(f"  iter {i:4d} | loss={loss.item():.6f}")

    with torch.no_grad():
        mu  = F.normalize(raw_dir, dim=-1)
        lam = 1.0 + 999.0 * torch.sigmoid(raw_lam)
        col = F.softplus(raw_col)
        sg_final = torch.cat([mu, lam, col], dim=-1)  # [K, 7]

    return sg_final.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_hdr(path: str) -> np.ndarray:
    """Load HDR file as float32 [H, W, 3] RGB."""
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load HDR: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Clip extreme outliers (e.g. sun disk) to avoid domination
    p99 = np.percentile(img, 99.5)
    img = np.clip(img, 0, p99)
    return img


# ---------------------------------------------------------------------------
# Entry Points
# ---------------------------------------------------------------------------

def process_single(args):
    print(f"Loading HDR: {args.input}")
    hdr = load_hdr(args.input)
    print(f"  Shape: {hdr.shape}, max={hdr.max():.2f}")
    print(f"Fitting {args.lobes} SG lobes...")
    sg = fit_sg_from_hdr(hdr, num_lobes=args.lobes, num_iters=args.iters, lr=args.lr,
                         device=args.device, verbose=True)
    np.save(args.output, sg)
    print(f"Saved SG GT: {args.output}  shape={sg.shape}")


def process_batch(args):
    """Walk DATA_DIR looking for .hdr files recursively.
    Fits an sg.npy in the same directory for each HDR found.
    Handles per-frame (or per-view) HDRs robustly.
    """
    from pathlib import Path
    data_path = Path(args.data_dir)
    
    # Try finding specified name first
    hdr_files = list(data_path.rglob(args.hdr_name))
    if not hdr_files:
        # Fallback to any .hdr
        hdr_files = list(data_path.rglob("*.hdr"))

    print(f"\n=== Found {len(hdr_files)} HDR files to process ===")
    skip = 0
    for hdr_path in tqdm(hdr_files):
        out_path = hdr_path.parent / "sg.npy"
        if out_path.exists() and not args.overwrite:
            skip += 1
            continue

        try:
            hdr = load_hdr(str(hdr_path))
            sg  = fit_sg_from_hdr(hdr, num_lobes=args.lobes, num_iters=args.iters,
                                  lr=args.lr, device=args.device)
            np.save(str(out_path), sg)
        except Exception as e:
            print(f"\n  [WARN] {hdr_path}: {e}")

    print(f"  Skipped (already done): {skip}")


def main():
    parser = argparse.ArgumentParser(description="Fit Spherical Gaussians from HDR env map")
    parser.add_argument("--input",    type=str, help="Path to a single .hdr file")
    parser.add_argument("--output",   type=str, help="Output path for sg.npy (single-file mode)")
    parser.add_argument("--data_dir", type=str, help="Root data dir (batch mode)")
    parser.add_argument("--hdr_name", type=str, default="imenvlow.hdr",
                        help="Filename of env map within each scene dir (batch mode)")
    parser.add_argument("--lobes",    type=int, default=24)
    parser.add_argument("--iters",    type=int, default=2000)
    parser.add_argument("--lr",       type=float, default=0.05)
    parser.add_argument("--device",   type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sg.npy")
    args = parser.parse_args()

    if args.input:
        assert args.output, "--output is required in single-file mode"
        process_single(args)
    elif args.data_dir:
        process_batch(args)
    else:
        parser.error("Either --input or --data_dir must be provided")


if __name__ == "__main__":
    main()

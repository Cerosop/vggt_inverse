#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# NOTE: Since OpenRooms imenvlow specifies cv2 HDR format reading
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class BatchedSGFitter(nn.Module):
    def __init__(self, batch_size, num_lobes=24, device='cuda'):
        super().__init__()
        self.num_lobes = num_lobes
        self.batch_size = batch_size
        
        # Fibonacci sphere for reasonable distributed initialization
        indices = torch.arange(0, num_lobes, dtype=torch.float32) + 0.5
        phi = torch.acos(1 - 2 * indices / num_lobes)
        theta = math.pi * (1 + 5**0.5) * indices
        x = torch.cos(theta) * torch.sin(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(phi)
        dirs = torch.stack([x, y, z], dim=1).float()
        
        # Expand for batch
        dirs = dirs.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        self.raw_dirs = nn.Parameter(dirs)
        self.raw_sharpness = nn.Parameter(torch.zeros(batch_size, num_lobes, 1, device=device)) 
        self.raw_amplitude = nn.Parameter(torch.ones(batch_size, num_lobes, 3, device=device) * 0.1)
        
    def forward(self, eval_dirs):
        # eval_dirs: [N, 3] representing coordinates on sphere
        dirs = F.normalize(self.raw_dirs, dim=-1) # [B, 24, 3]
        sharpness = torch.sigmoid(self.raw_sharpness) * 999.0 + 1.0 # [B, 24, 1]
        amplitude = F.softplus(self.raw_amplitude) # [B, 24, 3]
        
        # Efficient dot product broadcasting:
        # eval_dirs is [N, 3], dirs is [B, 24, 3]. Output [B, N, 24]
        cos_theta = torch.einsum('ni,bji->bnj', eval_dirs, dirs)
        
        exp_term = torch.exp(sharpness.transpose(1, 2) * (cos_theta - 1.0)) # [B, N, 24]
        # Element-wise and sum over lobes
        res = (exp_term.unsqueeze(-1) * amplitude.unsqueeze(1)).sum(dim=2) # [B, N, 3]
        return res
    
    def get_sg_params(self):
        with torch.no_grad():
            dirs = F.normalize(self.raw_dirs, dim=-1)
            sharpness = torch.sigmoid(self.raw_sharpness) * 999.0 + 1.0
            amplitude = F.softplus(self.raw_amplitude)
            return torch.cat([dirs, sharpness, amplitude], dim=-1).cpu().numpy() # [B, 24, 7]

def get_equirectangular_dirs(target_shape):
    # Returns [H*W, 3] spherical coordinates unit vectors and [H*W, 1] solid angle factors (sin(theta))
    H, W = target_shape
    theta = torch.linspace(0, math.pi, H + 2)[1:-1] # Avoid pure poles
    phi = torch.linspace(0, 2 * math.pi, W)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    x = torch.sin(theta) * torch.sin(phi)
    y = torch.cos(theta)
    z = torch.sin(theta) * torch.cos(phi)

    dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3).float()
    sin_theta = torch.sin(theta).reshape(-1, 1).float()
    return dirs, sin_theta

def batched_fit_sg(target_rgb_batch, target_shape, num_lobes=24, device='cuda', iters=2000):
    # target_rgb_batch: [B, H*W, 3]
    B = target_rgb_batch.shape[0]
    target_rgb_batch = target_rgb_batch.to(device)
    dirs, sin_theta = get_equirectangular_dirs(target_shape)
    dirs = dirs.to(device)
    sin_theta = sin_theta.to(device)
    
    # Normalize Solid Angle weights to sum to H*W
    weights = sin_theta * (target_shape[0] * target_shape[1] / sin_theta.sum())
    weights = weights.unsqueeze(0) # [1, N, 1]

    fitter = BatchedSGFitter(B, num_lobes, device)
    optimizer = torch.optim.Adam(fitter.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-4)

    for i in range(iters):
        optimizer.zero_grad()
        pred_rgb = fitter(dirs) # [B, N, 3]
        
        # MSE weighted by sin(theta) solid angle
        loss = (((pred_rgb - target_rgb_batch)**2) * weights).mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step()

    return fitter.get_sg_params()

def read_and_downsample_envmap(envmap_path, target_res):
    try:
        env = cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED)
        if env is None:
            return None
            
        env = env[:, :, ::-1].astype(np.float32) # BGR to RGB
        # (120*8, 160*16, 3) -> (120, 8, 160, 16, 3) 
        env = env.reshape(120, 8, 160, 16, 3)
        # Average spatial dims (axis=0 and 2), extracting purely angular global map -> (8, 16, 3)
        global_env = env.mean(axis=(0, 2))
        
        # Up-res it
        global_env_hr = cv2.resize(global_env, (target_res[1], target_res[0]), interpolation=cv2.INTER_LINEAR)
        global_env_hr = torch.from_numpy(global_env_hr).reshape(-1, 3)

        # Ensure no invalid pixels
        global_env_hr = torch.nan_to_num(global_env_hr, nan=0.0, posinf=0.0, neginf=0.0)
        return global_env_hr
    except Exception as e:
        logger.error(f"Error processing {envmap_path}: {e}")
        return None

def process_scene(scene_name: str, version: str, low_base: Path, processed_base: Path, split: str, iters=2000):
    parts = scene_name.split("_")
    
    if scene_name.startswith("mainDiffLight_"):
        version = "mainDiffLight_xml" if "xml_" in scene_name and not "xml1_" in scene_name else "mainDiffLight_xml1"
        rest = scene_name[len("mainDiffLight_xml_"):] if "_xml_" in scene_name else scene_name[len("mainDiffLight_xml1_"):]
    elif scene_name.startswith("mainDiffMat_"):
        version = "mainDiffMat_xml" if "xml_" in scene_name and not "xml1_" in scene_name else "mainDiffMat_xml1"
        rest = scene_name[len("mainDiffMat_xml_"):] if "_xml_" in scene_name else scene_name[len("mainDiffMat_xml1_"):]
    elif scene_name.startswith("main_xml1_"):
        version = "main_xml1"
        rest = scene_name[len("main_xml1_"):]
    elif scene_name.startswith("main_xml_"):
        version = "main_xml"
        rest = scene_name[len("main_xml_"):]
    else:
        return 0
        
    rest_parts = rest.rsplit("_", 1)
    if len(rest_parts) != 2:
        return 0
        
    original_scene = rest_parts[0]
    img_idx = rest_parts[1]
    
    envmap_dir = low_base / version / original_scene
    processed_scene_dir = processed_base / split / scene_name
    if not envmap_dir.exists() or not processed_scene_dir.exists():
        return 0
        
    paths_to_process = []
    for view_idx in range(9):
        view_num = view_idx + 1  
        envmap_path = envmap_dir / f"{img_idx}_imenvlow_{view_num}.hdr"
        output_dir = processed_scene_dir / str(view_idx)
        output_path = output_dir / "sg.npy"
        
        if envmap_path.exists() and output_dir.exists() and not output_path.exists():
            paths_to_process.append((view_idx, envmap_path, output_path))
            
    if not paths_to_process:
        return 0

    target_res = (64, 128)
    loaded_data = []
    
    # Extract IO parallelization
    with ThreadPoolExecutor(max_workers=min(9, len(paths_to_process))) as executor:
        future_to_view = {executor.submit(read_and_downsample_envmap, p[1], target_res): p for p in paths_to_process}
        for future in as_completed(future_to_view):
            p = future_to_view[future]
            res = future.result()
            if res is not None:
                loaded_data.append((p[0], p[2], res))
                
    if not loaded_data:
        return 0

    loaded_data.sort(key=lambda x: x[0])
    target_batch = torch.stack([x[2] for x in loaded_data]) # [B, N, 3]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sg_params_batch = batched_fit_sg(target_batch, target_res, num_lobes=24, device=device, iters=iters)
    
    for idx, (_, output_path, _) in enumerate(loaded_data):
        np.save(str(output_path), sg_params_batch[idx].astype(np.float32))

    return len(loaded_data)

def main():
    parser = argparse.ArgumentParser(description="Fit 24 SG components from OpenRooms EnvMaps (Batched Parallel V2)")
    parser.add_argument("--low_dir", type=str, default="/train-data-3-hdd/cerosop/vggt/OpenRooms_FF")
    parser.add_argument("--processed_dir", type=str, default="/train-data-3-hdd/cerosop/vggt/processed_data_openroomsff")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--iters", type=int, default=2000)
    args = parser.parse_args()

    low_dir = Path(args.low_dir)
    processed_dir = Path(args.processed_dir)

    total = 0
    for split in ["train", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists(): continue
        
        scenes = [d.name for d in split_dir.iterdir() if d.is_dir()]
        logger.info(f"Processing {len(scenes)} scenes in {split}")
        
        if args.num_workers > 1:
            torch.multiprocessing.set_start_method('spawn', force=True)
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {}
                for scene_name in scenes:
                    future = executor.submit(process_scene, scene_name, "", low_dir, processed_dir, split, args.iters)
                    futures[future] = scene_name
                
                with tqdm(total=len(futures)) as pbar:
                    for future in as_completed(futures):
                        total += future.result()
                        pbar.update(1)
        else:
            with tqdm(total=len(scenes)) as pbar:
                for scene_name in scenes:
                    total += process_scene(scene_name, "", low_dir, processed_dir, split, args.iters)
                    pbar.update(1)

    logger.info(f"Total processed SG views: {total}")

if __name__ == "__main__":
    main()

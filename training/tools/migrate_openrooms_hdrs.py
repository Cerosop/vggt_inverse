import os
import os.path as osp
import numpy as np
import cv2
from cv2 import COLOR_BGR2RGB  # explicit import for clarity in some environments
from tqdm import tqdm
import argparse
from pathlib import Path

def migrate_scene(scene_path, source_root, target_res=(128, 256), dry_run=False, delete_original=False):
    """
    scene_path: Path to processed scene folder, e.g. .../train/main_xml1_scene0291_01_4
    """
    scene_name = osp.basename(scene_path)
    parts = scene_name.split('_')
    
    if len(parts) < 4:
        return 0, 0 # Skip
    
    # Logic: last is env_id, previous two are scene_id
    env_id = parts[-1]
    scene_id = "_".join(parts[-3:-1])
    dir_type = "_".join(parts[:-3])
    
    source_dir = osp.join(source_root, dir_type, scene_id)
    if not osp.isdir(source_dir):
        # print(f"Source dir not found: {source_dir}")
        return 0, 0
    
    # Find frame subfolders in scene_path
    frame_dirs = [d for d in os.listdir(scene_path) if osp.isdir(osp.join(scene_path, d)) and d.isdigit()]
    
    migrated = 0
    skipped = 0
    
    for frame_id in frame_dirs:
        # Pattern: {frame_id}_imenvlow_{env_id}.hdr
        hdr_filename = f"{frame_id}_imenvlow_{env_id}.hdr"
        source_hdr = osp.join(source_dir, hdr_filename)
        
        target_npy = osp.join(scene_path, frame_id, "env_map.npy")
        
        if osp.exists(target_npy):
            if delete_original and osp.exists(source_hdr):
                os.remove(source_hdr)
                migrated += 1
            else:
                skipped += 1
            continue
            
        if not osp.exists(source_hdr):
            # Try alternate names if any, but standard is this
            skipped += 1
            continue
            
        if dry_run:
            migrated += 1
            continue
            
        try:
            # Load
            img = cv2.imread(source_hdr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, COLOR_BGR2RGB).astype(np.float32)
            
            # Clip extreme outliers (e.g. sun disk) to avoid float16 overflow and dominated loss
            p99 = np.percentile(img, 99.5)
            img = np.clip(img, 0, p99)
            
            # Resize
            img_small = cv2.resize(img, (target_res[1], target_res[0]), interpolation=cv2.INTER_LINEAR)
            
            # Save as npy float16
            np.save(target_npy, img_small.astype(np.float16))
            
            # Delete original
            if delete_original:
                os.remove(source_hdr)
                
            migrated += 1
        except Exception as e:
            print(f"Error processing {source_hdr}: {e}")
            
    return migrated, skipped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, default="/train-data-3-hdd/cerosop/vggt/processed_data_openroomsff")
    parser.add_argument("--source_root", type=str, default="/train-data-3-hdd/cerosop/vggt/OpenRooms_FF")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--delete", action="store_true", help="Delete original HDRs after migration")
    args = parser.parse_args()
    
    processed_root = Path(args.processed_root)
    total_migrated = 0
    total_skipped = 0
    
    for split in ["train", "test"]:
        split_dir = processed_root / split
        if not split_dir.exists():
            continue
            
        scenes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"Processing split '{split}', {len(scenes)} scenes...")
        
        for scene_path in tqdm(scenes):
            m, s = migrate_scene(str(scene_path), args.source_root, dry_run=args.dry_run, delete_original=args.delete)
            total_migrated += m
            total_skipped += s
            
    print(f"\nDone! Migrated: {total_migrated}, Skipped/Already exist: {total_skipped}")

if __name__ == "__main__":
    main()

import os
import sys
from omegaconf import DictConfig

# Mock logging to see everything
import logging
logging.basicConfig(level=logging.INFO)

# Add paths to sys.path
sys.path.append(os.getcwd())

from data.datasets.inverse_rendering_dataset import InverseRenderingDataset

def check_dataset():
    common_conf = DictConfig({
        "img_size": 518,
        "patch_size": 14,
        "training": True
    })
    
    data_dir = "/train-data-3-hdd/cerosop/vggt/processed_data_matrixcity"
    print(f"Checking DATA_DIR: {data_dir}")
    
    ds = InverseRenderingDataset(
        common_conf=common_conf,
        split="train",
        DATA_DIR=data_dir
    )
    
    print(f"Number of scenes found: {len(ds.scenes)}")
    if len(ds.scenes) > 0:
        print(f"First scene: {ds.scenes[0]}")
        print(f"Number of frames in first scene: {len(ds.scene_data[ds.scenes[0]]['frame_ids'])}")
    else:
        print("SCENES LIST IS EMPTY!")
        # Check why
        split_dir = os.path.join(data_dir, "train")
        print(f"Split dir exists? {os.path.isdir(split_dir)}")
        if os.path.isdir(split_dir):
            members = os.listdir(split_dir)
            print(f"Members of split dir: {members}")
            if members:
                scene_dir = os.path.join(split_dir, members[0])
                print(f"Checking scene dir {scene_dir}")
                frame_ids = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]
                print(f"Frame IDs found: {frame_ids}")

if __name__ == "__main__":
    check_dataset()

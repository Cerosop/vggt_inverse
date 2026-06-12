import sys
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

sys.path.append(os.path.join(os.path.dirname(__file__), "training"))
from hydra import initialize, compose
from hydra.utils import instantiate

def main():
    # Force single-node distributed mode since DataLoader requires DDP group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29510"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    import torch.distributed as dist
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)

    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="inverse_rendering")
        
    dataset = instantiate(cfg.data.val, _recursive_=False)
    loader = dataset.get_loader(epoch=0)
    
    expected_datasets = ["hypersim", "interiorverse", "matrixcity_normal"]
    dataset_counts = {}
    
    for i, batch in enumerate(loader):
        d_names = batch.get('dataset_name')
        if d_names is None:
            continue
        d_name = d_names[0]
        seq_name = batch.get('seq_name')[0]
        ids = batch.get('ids')[0]
        
        if d_name not in dataset_counts:
            dataset_counts[d_name] = 0
            
        if d_name in expected_datasets and dataset_counts[d_name] == 0:
            dataset_counts[d_name] += 1
            print(f"--- MATCH ---")
            print(f"Dataset: {d_name}")
            print(f"Original Scene: {seq_name}")
            print(f"Sampled IDs: {ids}")
            print(f"View 0 corresponds to frame idx: {ids[0]}")
            
        done = True
        for ed in expected_datasets:
            if dataset_counts.get(ed, 0) == 0:
                done = False
        if done:
            break

if __name__ == "__main__":
    main()

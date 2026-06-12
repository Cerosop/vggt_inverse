import sys
import os
import torch
import torch.distributed as dist
from torchvision.utils import save_image
import glob
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "training"))
from hydra import initialize, compose
from hydra.utils import instantiate

def main():
    # Force single-node distributed mode since DataLoader requires DDP group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29507"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    
    with initialize(version_base=None, config_path="training/config"):
        cfg = compose(config_name="inverse_rendering")
        
    # Identify checkpoint
    ckpt_path = "/train-data-3-hdd/cerosop/vggt/vggt_origin/vggt/logs_0519/inverse_rendering/ckpts/checkpoint.pt"
    # print the loaded checkpoint clearly
    print(f"=====================================")
    print(f"Loading specific checkpoint: {ckpt_path}")
    print(f"=====================================")
    
    cfg.model.enable_light_token = False
    cfg.model.lora_global_base_rank = 16
    cfg.model.lora_tail_layers = 6
    cfg.model.lora_tail_rank = 64
    model = instantiate(cfg.model, _recursive_=False)
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # Clean state dict keys if they have "module." due to DDP
    state_dict = checkpoint["model"]
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # Load model weights
    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    # If light token is missing in state dict but expected in model, it would error if strict=True
    # But here we disabled it in model, so it should match better.
    
    model.cuda()
    model.eval()
    
    output_dir = "demo_outputs_0519"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Predictions and GTs will be saved to: {output_dir}/")
    
    # Target: 3 scenes per dataset per split
    scenes_per_dataset = 3
    
    heads = ["albedo", "roughness", "normal", "metallic", "shading"]
    
    # Splits to process
    splits = [
        ("test", cfg.data.val),
        ("train", cfg.data.train)
    ]
    
    for split_name, split_cfg in splits:
        print(f"\n=====================================")
        print(f"Processing SPLIT: {split_name}")
        print(f"=====================================")
        
        # Setup dataset
        print(f"Loading {split_name} Dataset...")
        dataset = instantiate(split_cfg, _recursive_=False)
        loader = dataset.get_loader(epoch=0)
        
        dataset_counts = {}
        
        # Identify expected datasets from config for better progress tracking
        expected_datasets = []
        if hasattr(split_cfg.dataset, 'dataset_configs'):
            for ds_cfg in split_cfg.dataset.dataset_configs:
                d_dir = getattr(ds_cfg, 'DATA_DIR', '')
                if d_dir:
                    name = os.path.basename(d_dir.rstrip('/')).replace('processed_data_', '')
                    if name not in expected_datasets:
                        expected_datasets.append(name)
        
        print(f"Targeting these datasets in {split_name}: {expected_datasets}")

        with torch.no_grad():
            for i, batch in enumerate(loader):
                # dataset_name is a list (batch) of strings
                d_names = batch.get('dataset_name')
                if d_names is None:
                    dataset_name = "unknown"
                else:
                    dataset_name = d_names[0]
                
                if dataset_name not in dataset_counts:
                    dataset_counts[dataset_name] = 0
                
                if dataset_counts[dataset_name] >= scenes_per_dataset:
                    # If we have enough for this dataset, check if we need more for others
                    all_done = True
                    for ed in expected_datasets:
                        if dataset_counts.get(ed, 0) < scenes_per_dataset:
                            all_done = False
                            break
                    if all_done:
                        print(f"All datasets in {split_name} covered!")
                        break
                    continue 
                    
                # Send batch to GPU
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda()
                        
                print(f"[{split_name}][{i}] Processing {dataset_name} scene {dataset_counts[dataset_name]+1}/{scenes_per_dataset} ...")
                preds = model(batch['images'])
                
                # shape: [B, S, 3, H, W]
                B, S = batch['images'].shape[:2]
                
                # Save 3 views
                num_views_to_save = min(3, S)
                
                # Setup scene folder: split/dataset/scene_X
                scene_dir = os.path.join(output_dir, split_name, dataset_name, f"scene_{dataset_counts[dataset_name]}")
                os.makedirs(scene_dir, exist_ok=True)
                
                for img_idx in range(num_views_to_save):
                    view_dir = os.path.join(scene_dir, f"view_{img_idx}")
                    os.makedirs(view_dir, exist_ok=True)
                    
                    # Save Input Image
                    img = batch['images'][0, img_idx].clone()
                    if img.dim() == 4:
                        img = img.squeeze(0)
                    if img.shape[-1] == 3:
                        img = img.permute(2, 0, 1)

                    # Restore 0-1 range (ComposedDataset divides by 255)
                    img = img
                    img = torch.clamp(img, 0.0, 1.0)
                    save_image(img, os.path.join(view_dir, "input.png"))
                    
                    # Save Predictions and GTs
                    for head in heads:
                        # 1. Prediction
                        if head in preds:
                            pred_t = preds[head][0, img_idx].clone()
                            if pred_t.dim() == 4:
                                pred_t = pred_t.squeeze(0)
                            if pred_t.shape[-1] in [1, 3]:
                                pred_t = pred_t.permute(2, 0, 1)
                            if pred_t.shape[0] == 1:
                                pred_t = pred_t.repeat(3, 1, 1)
                            if head == "normal" and pred_t.shape[0] == 3:
                                pred_t = (pred_t + 1.0) / 2.0
                            pred_t = torch.clamp(pred_t, 0, 1)
                            save_image(pred_t, os.path.join(view_dir, f"pred_{head}.png"))
                        
                        # 2. GT
                        gt_key = f"gt_{head}"
                        if gt_key in batch:
                            gt_t = batch[gt_key][0, img_idx].clone()
                            if gt_t.dim() == 4:
                                gt_t = gt_t.squeeze(0)
                            if gt_t.shape[-1] in [1, 3]:
                                gt_t = gt_t.permute(2, 0, 1)
                            if gt_t.shape[0] == 1:
                                gt_t = gt_t.repeat(3, 1, 1)
                            if head == "normal" and gt_t.shape[0] == 3:
                                gt_t = (gt_t + 1.0) / 2.0
                            gt_t = torch.clamp(gt_t, 0, 1)
                            save_image(gt_t, os.path.join(view_dir, f"gt_{head}.png"))
                
                dataset_counts[dataset_name] += 1
                
                # Print current status
                status_str = ", ".join([f"{k}: {v}" for k, v in dataset_counts.items()])
                print(f"Current Progress ({split_name}): {status_str}")
                
                if sum(dataset_counts.values()) >= 50: # Max 50 scenes total per split
                    print(f"Reached maximum scene limit (50) for {split_name}. Stopping split.")
                    break

    print("\nDone! All images saved.")

if __name__ == "__main__":
    main()

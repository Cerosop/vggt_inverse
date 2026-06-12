import torch
from data.composed_dataset import ComposedDataset
from omegaconf import DictConfig

def test_sampling():
    # Mock datasets
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, name, length):
            self.name = name
            self.length = length
        def __len__(self):
            return self.length
        def __getitem__(self, idx_tuple):
            return {"seq_name": f"{self.name}_{idx_tuple[0]}", "images": [], "ids": []}

    # Common config
    common_config = DictConfig({
        "augs": {"cojitter": False, "cojitter_ratio": 0.5, "color_jitter": None, "gray_scale": False, "gau_blur": False},
        "fix_img_num": -1,
        "fix_aspect_ratio": 1.0,
        "load_track": False,
        "track_num": 0,
        "training": True,
        "inside_random": True
    })

    # Manual instantiation logic since we can't easily mock Hydra instantiate here
    ds1 = MockDataset("A", 100)
    ds2 = MockDataset("B", 1000000) # Dataset B is much larger

    from data.composed_dataset import TupleConcatDataset
    
    # Test 1: 50/50 weights
    weights = [0.5, 0.5]
    concat = TupleConcatDataset([ds1, ds2], common_config, weights=weights)
    
    counts = {"A": 0, "B": 0}
    for _ in range(1000):
        item = concat[(0, 5, 1.0)]
        if item["seq_name"].startswith("A"):
            counts["A"] += 1
        else:
            counts["B"] += 1
    
    print(f"Results with 50/50 weights (A=100, B=1M): {counts}")
    # Without weights, B would be selected 99.9% of the time. 
    # With weights, it should be close to 500/500.

    # Test 2: 90/10 weights
    weights = [0.9, 0.1]
    concat = TupleConcatDataset([ds1, ds2], common_config, weights=weights)
    counts = {"A": 0, "B": 0}
    for _ in range(1000):
        item = concat[(0, 5, 1.0)]
        if item["seq_name"].startswith("A"):
            counts["A"] += 1
        else:
            counts["B"] += 1
    print(f"Results with 90/10 weights (A=100, B=1M): {counts}")

if __name__ == "__main__":
    test_sampling()

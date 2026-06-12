import torch
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from vggt.utils.geometry import closed_form_inverse_se3 # to warm up imports

conf = OmegaConf.load('training/config/inverse_rendering.yaml')
conf.data.train.num_workers = 0

dataloader_conf = conf.data.train
train_loader = instantiate(dataloader_conf)
print("Dataloader length:", len(train_loader))

for idx, batch in enumerate(train_loader):
    print("Batch type:", type(batch))
    if isinstance(batch, list):
         print("This is a list of size", len(batch))
         print("First element keys:", batch[0].keys())
    else:
         print("Batch keys:", list(batch.keys()))
    if idx > 0:
         break


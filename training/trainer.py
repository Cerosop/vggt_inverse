# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os


# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Check if we should auto-resume from the folder
        auto_resume = getattr(self.checkpoint_conf, "auto_resume", True)
        
        if not auto_resume and self.rank == 0:
            import shutil
            import logging as py_logging
            try:
                if os.path.exists(self.checkpoint_conf.save_dir):
                    shutil.rmtree(self.checkpoint_conf.save_dir)
                    py_logging.info(f"Deleted entire checkpoint folder because auto_resume=False: {self.checkpoint_conf.save_dir}")
                os.makedirs(self.checkpoint_conf.save_dir, exist_ok=True)
            except OSError as e:
                py_logging.warning(f"Could not fully delete the checkpoint folder: {e}")

        # Prioritize resuming from previous crashed/stopped training
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir) if auto_resume else None
        
        if ckpt_path is not None:
            self._load_resuming_checkpoint(ckpt_path)
            # Use Python's built-in logging module instead of the 'logging' argument shadow
            import logging as py_logging
            py_logging.info(f"Auto-resumed from previous training checkpoint: {ckpt_path}")
        elif getattr(self.checkpoint_conf, "resume_checkpoint_path", None) is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
            import logging as py_logging
            py_logging.info(f"Loaded starting weights from: {self.checkpoint_conf.resume_checkpoint_path}")

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # Filter out shape mismatches (e.g. from changing LoRA ranks)
        current_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        for k, v in model_state_dict.items():
            if k in current_state_dict and current_state_dict[k].shape != v.shape:
                logging.warning(f"Shape mismatch for {k}: checkpoint has {v.shape}, model has {current_state_dict[k].shape}. Skipping this key (rank {self.rank}).")
                continue
            filtered_state_dict[k] = v

        missing, unexpected = self.model.load_state_dict(
            filtered_state_dict, strict=self.checkpoint_conf.strict
        )
        # if self.rank == 0:
        #     logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        load_weights_only = getattr(self.checkpoint_conf, "load_weights_only", False)
        
        if not load_weights_only:
            if "optimizer" in checkpoint:
                logging.info(f"Loading optimizer state dict (rank {self.rank})")
                # self.optims.optimizer.load_state_dict(checkpoint["optimizer"])
                opt_state = checkpoint["optimizer"]
                if isinstance(opt_state, list):
                    for i, state in enumerate(opt_state):
                        self.optims[i].optimizer.load_state_dict(state)
                else:
                    self.optims[0].optimizer.load_state_dict(opt_state)

            # Load training progress
            if "epoch" in checkpoint:
                self.epoch = checkpoint["epoch"]
            elif "prev_epoch" in checkpoint:
                self.epoch = checkpoint["prev_epoch"]

            self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

            # Load AMP scaler state if available
            if self.optim_conf.amp.enabled and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
        else:
            logging.info(f"Skipping optimizer and epoch recovery because load_weights_only=True")

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            # "prev_epoch": epoch,
            "epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            batch["phase_ratio"] = float(self.epoch) / max(1.0, float(self.max_epochs))
            batch["phase_ratio"] = float(self.epoch) / max(1.0, float(self.max_epochs))

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict, y_hat = self._step(
                        batch, self.model, phase, loss_meters
                    )
            
            if data_iter < getattr(self.logging_conf, "num_demo_scenes", 3) and self.rank == 0:
                import os
                from torchvision.utils import save_image
                out_dir = os.path.join(self.checkpoint_conf.save_dir, "val_outputs", f"epoch_{self.epoch + 1}", f"scene_{data_iter}")
                os.makedirs(out_dir, exist_ok=True)
                
                img_idx = 0
                img = batch['images'][0, img_idx].clone()
                if img.dim() == 4: img = img.squeeze(0)
                if img.shape[-1] == 3: img = img.permute(2, 0, 1)
                save_image(torch.clamp(img, 0., 1.), os.path.join(out_dir, "input.png"))

                # d4rt: re-run the frozen backbone on just the demo frame ONCE; shared by
                # the full material maps + the render preview below. (_tk/_psi/_di or None.)
                _m = self.model.module if hasattr(self.model, "module") else self.model
                _d4rt = getattr(_m, "d4rt_heads", None)
                _tk = _psi = _di = None
                if _d4rt is not None and _d4rt.enable_material and "material_uv" in y_hat:
                    try:
                        with torch.no_grad():
                            _di = batch["images"][:1, img_idx:img_idx + 1]   # [1,1,3,H,W]
                            _agg, _lora, _psi, _ = _m.aggregator(_di)
                            _tk = _lora if _lora is not None else _agg
                    except Exception as e:
                        logging.warning(f"[val demo] backbone re-run failed: {e}")
                        _tk = None

                mat_full = None      # full per-pixel material maps for the demo frame ([1,1,H,W,c])
                if _tk is not None:
                    try:
                        mat_full = _d4rt.predict_material_dense(_tk, _di, _psi)
                    except Exception as e:
                        logging.warning(f"[val demo] dense material query failed: {e}")

                for head in ["albedo", "metallic", "roughness", "normal", "shading"]:
                    pred_t = None
                    if mat_full is not None and head in mat_full:
                        pred_t = mat_full[head][0, 0].clone()          # [H,W,c]
                    elif "material_uv" not in y_hat and head in y_hat:  # dense path (num_material_samples=0)
                        pred_t = y_hat[head][0, img_idx].clone()
                    if pred_t is not None:
                        if pred_t.dim() == 4: pred_t = pred_t.squeeze(0)
                        if pred_t.shape[-1] in [1, 3]: pred_t = pred_t.permute(2, 0, 1)
                        if pred_t.shape[0] == 1: pred_t = pred_t.repeat(3, 1, 1)
                        if head == "normal" and pred_t.shape[0] == 3: pred_t = (pred_t + 1.0) / 2.0
                        save_image(torch.clamp(pred_t, 0.0, 1.0), os.path.join(out_dir, f"pred_{head}.png"))

                    gt_key = f"gt_{head}"
                    if gt_key in batch and batch[gt_key] is not None:
                        gt_t = batch[gt_key][0, img_idx].clone()
                        if gt_t.dim() == 4: gt_t = gt_t.squeeze(0)
                        if gt_t.shape[-1] in [1, 3]: gt_t = gt_t.permute(2, 0, 1)
                        if gt_t.shape[0] == 1: gt_t = gt_t.repeat(3, 1, 1)
                        if head == "normal" and gt_t.shape[0] == 3: gt_t = (gt_t + 1.0) / 2.0
                        save_image(torch.clamp(gt_t, 0.0, 1.0), os.path.join(out_dir, f"gt_{head}.png"))

                # --- d4rt render preview (ALWAYS, regardless of render-loss flag) ---
                # Smart full-res render: full-res material (reuse mat_full) × lighting on a
                # (H/f, W/f) coarse grid upsampled by f (render_demo_upsample). sRGB, full res.
                if _tk is not None and _d4rt.enable_lighting and mat_full is not None:
                    try:
                        with torch.no_grad():
                            pm = cam = None
                            if getattr(_m, "per_pixel_render_specular", False) and _m.point_head is not None:
                                from vggt.heads.brdf_renderer import compute_camera_positions
                                _pose = _m.camera_head(_tk)
                                cam = compute_camera_positions(_pose[-1])
                                pm, _ = _m.point_head(_tk, images=_di, patch_start_idx=_psi)
                            rendered = _d4rt.render_demo(_tk, _di, _psi, material=mat_full,
                                                         point_map=pm, camera_pos=cam)  # [1,1,H,W,3]
                            rimg = rendered[0, 0].permute(2, 0, 1)                      # [3,H,W] full res
                            save_image(torch.clamp(rimg, 0., 1.), os.path.join(out_dir, "pred_render.png"))
                    except Exception as e:
                        logging.warning(f"[val demo] render preview failed: {e}")

                # --- Light token demos: SG environment map + BRDF-rendered image ---
                # SG params are hard to visualize directly, so we render the SG lobes
                # into an equirectangular env map. We also render the predicted
                # materials + SG lighting through the BRDF renderer. pose_enc /
                # world_points are present because the VGGT geometry heads run in val.
                with torch.no_grad():
                    sg = y_hat.get("sg_params")
                    if sg is not None:
                        try:
                            from sg_loss import render_env_map_from_sg
                            env = render_env_map_from_sg(
                                sg[:1, img_idx:img_idx + 1].float(), height=128, width=256
                            )  # [1, 1, 128, 256, 3], HDR radiance
                            env_t = env[0, 0]
                            env_t = env_t / (env_t + 1.0)  # Reinhard tone-map -> [0, 1)
                            save_image(torch.clamp(env_t.permute(2, 0, 1), 0., 1.),
                                       os.path.join(out_dir, "pred_env_map.png"))
                        except Exception as e:
                            logging.warning(f"[val demo] env_map render failed: {e}")

                        gt_env = batch.get("gt_env_map")
                        if isinstance(gt_env, torch.Tensor):
                            try:
                                ge = gt_env[0, img_idx].float()
                                if ge.shape[-1] == 3 and ge.shape[0] != 3:
                                    ge = ge.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
                                ge = ge / (ge + 1.0)  # tone-map HDR env map
                                save_image(torch.clamp(ge, 0., 1.),
                                           os.path.join(out_dir, "gt_env_map.png"))
                            except Exception as e:
                                logging.warning(f"[val demo] gt_env_map save failed: {e}")

                    # --- d4rt per-pixel env demo (lighting_mode="per_pixel_env") ---
                    # No SG to render; instead visualize the predicted env tile at the
                    # center spatial pixel (8x16 hemisphere) vs the imenvlow GT tile.
                    pe = y_hat.get("pred_env_pixel")
                    if isinstance(pe, torch.Tensor):
                        try:
                            def _env_tile_img(tile):  # [env_h, env_w, 3] HDR -> [3,Hh,Ww] LDR
                                t = tile.float()
                                t = t / (t + 1.0)                  # Reinhard tone-map
                                t = t.permute(2, 0, 1).unsqueeze(0)
                                t = torch.nn.functional.interpolate(
                                    t, scale_factor=16, mode="nearest")  # upscale for visibility
                                return torch.clamp(t[0], 0., 1.)
                            gh, gw = pe.shape[2], pe.shape[3]
                            ptile = pe[0, img_idx, gh // 2, gw // 2]   # [env_h,env_w,3]
                            save_image(_env_tile_img(ptile),
                                       os.path.join(out_dir, "pred_env_pixel.png"))
                            gep = batch.get("gt_env_pixel")
                            if isinstance(gep, torch.Tensor) and gep[0, img_idx].abs().sum() > 0:
                                Hs, Ws = gep.shape[2], gep.shape[3]
                                gtile = gep[0, img_idx, Hs // 2, Ws // 2]  # [env_h,env_w,3]
                                save_image(_env_tile_img(gtile),
                                           os.path.join(out_dir, "gt_env_pixel.png"))
                        except Exception as e:
                            logging.warning(f"[val demo] env_pixel save failed: {e}")

                    render_keys = ["albedo", "normal", "roughness", "metallic",
                                   "sg_params", "pose_enc", "world_points"]
                    if all(y_hat.get(k) is not None for k in render_keys):
                        try:
                            from vggt.heads.brdf_renderer import BRDFRenderer, compute_camera_positions
                            sl = slice(img_idx, img_idx + 1)
                            cam_pos = compute_camera_positions(y_hat["pose_enc"][:1, sl].float())
                            rendered = BRDFRenderer(render_downsample=2)(
                                albedo=y_hat["albedo"][:1, sl].float(),
                                normal=y_hat["normal"][:1, sl].float(),
                                roughness=y_hat["roughness"][:1, sl].float(),
                                metallic=y_hat["metallic"][:1, sl].float(),
                                sg_params=y_hat["sg_params"][:1, sl].float(),
                                point_map=y_hat["world_points"][:1, sl].float(),
                                camera_pos=cam_pos,
                            )  # [1, 1, H, W, 3]; BRDF computes lighting from SG, so no shading multiply
                            save_image(torch.clamp(rendered[0, 0].permute(2, 0, 1), 0., 1.),
                                       os.path.join(out_dir, "pred_render.png"))

                            # Renderer "ceiling" check: render with GT materials + GT SG
                            # (+ predicted geometry) so we can see how close the renderer
                            # itself can get to the input given perfect materials/lighting.
                            gt_mat_keys = ["gt_albedo", "gt_normal", "gt_roughness", "gt_metallic"]
                            if (batch.get("gt_sg") is not None
                                    and all(batch.get(k) is not None for k in gt_mat_keys)):
                                def _match(t):  # resize GT map to the geometry resolution if needed
                                    pm = y_hat["world_points"][:1, sl]
                                    if t.shape[2:4] != pm.shape[2:4]:
                                        b_, s_, h_, w_, c_ = t.shape
                                        t = torch.nn.functional.interpolate(
                                            t.reshape(b_ * s_, h_, w_, c_).permute(0, 3, 1, 2),
                                            size=pm.shape[2:4], mode="bilinear", align_corners=False,
                                        ).permute(0, 2, 3, 1).reshape(b_, s_, pm.shape[2], pm.shape[3], c_)
                                    return t
                                rendered_gt = BRDFRenderer(render_downsample=2)(
                                    albedo=_match(batch["gt_albedo"][:1, sl].float()),
                                    normal=_match(batch["gt_normal"][:1, sl].float()),
                                    roughness=_match(batch["gt_roughness"][:1, sl].float()),
                                    metallic=_match(batch["gt_metallic"][:1, sl].float()),
                                    sg_params=batch["gt_sg"][:1, sl].float(),
                                    point_map=y_hat["world_points"][:1, sl].float(),
                                    camera_pos=cam_pos,
                                )
                                save_image(torch.clamp(rendered_gt[0, 0].permute(2, 0, 1), 0., 1.),
                                           os.path.join(out_dir, "gt_render.png"))
                        except Exception as e:
                            logging.warning(f"[val demo] BRDF render failed: {e}")

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)


        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            if self.rank == 0:
                seq_names = batch.get("seq_name", ["Unknown"])
                avail_gts = batch.get("available_gts", "None")
                if isinstance(avail_gts, list) and len(avail_gts) > 0 and isinstance(avail_gts[0], tuple):
                    gts_print_str = str([g[0] for g in avail_gts])
                else:
                    gts_print_str = str(avail_gts)
                logging.info(f"===> Batch {data_iter}: Datasets={seq_names}, GTs={gts_print_str}")

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            batch["phase_ratio"] = float(self.epoch) / max(1.0, float(self.max_epochs))

            accum_steps = self.accum_steps

            if accum_steps==1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict, y_hat = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics", 
            "cam_points", "world_points", "point_masks", 
        ]        
        string_keys = ["seq_name"]
        
        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, 
                                                torch.flip(original_tensor, dims=[1])], 
                                                dim=0)
        
        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2
        
        return batch

    def _process_batch(self, batch: Mapping):      
        if self.data_conf.train.common_config.get('repeat_batch', False):
            batch = self._apply_batch_repetition(batch)
        
        # Normalize camera extrinsics and points (only if geometry data is present)
        # Inverse rendering datasets may not have these keys.
        if all(k in batch for k in ["extrinsics", "cam_points", "world_points", "depths", "point_masks"]):
            normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = \
                normalize_camera_extrinsics_and_points_batch(
                    extrinsics=batch["extrinsics"],
                    cam_points=batch["cam_points"],
                    world_points=batch["world_points"],
                    depths=batch["depths"],
                    point_masks=batch["point_masks"],
                )
            batch["extrinsics"] = normalized_extrinsics
            batch["cam_points"] = normalized_cam_points
            batch["world_points"] = normalized_world_points
            batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        y_hat = model(images=batch["images"])
        
        # Ensure images are in batch for BRDF render loss
        if "images" not in batch:
            batch["images"] = batch.get("images")
        
        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict, y_hat

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['images'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )




def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data



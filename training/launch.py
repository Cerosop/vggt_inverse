# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def _apply_finetune_mode(cfg: DictConfig) -> None:
    """Translate `inverse_rendering.finetune_mode` into concrete cfg settings.

    Modes:
      - "lora": LoRA adapters on frame/global blocks; original blocks frozen.
      - "full": directly finetune the original frame/global blocks; no LoRA.

    This rewrites `enable_lora`, `optim.frozen_module_names`, and
    `optim.gradient_clip.configs` so the rest of the pipeline does not need
    to know about the mode.
    """
    inv = cfg.get("inverse_rendering", None)
    if inv is None:
        return

    mode = inv.get("finetune_mode", "lora")
    if mode not in ("lora", "full"):
        raise ValueError(
            f"inverse_rendering.finetune_mode must be 'lora' or 'full', got {mode!r}"
        )

    use_lora = (mode == "lora")
    OmegaConf.update(cfg, "inverse_rendering.enable_lora", use_lora, merge=False)

    # When training the light token, keep the per-frame pathway frozen and let only
    # the global pathway adapt:
    #   - full mode: unfreeze only the global blocks (frame blocks stay frozen).
    #   - lora mode: no LoRA on the frame blocks (handled in the Aggregator) and the
    #     original frame blocks stay frozen (already in the frozen list).
    light_token = bool(inv.get("enable_light_token", False))

    optim = cfg.get("optim", None)
    if optim is None:
        return

    # --- Adjust frozen_module_names ---
    # We explicitly add/remove the frame & global block patterns so the result is
    # correct regardless of whether the user left them in `frozen_module_names`.
    def _has_orig(patterns, substr):
        return any(substr in p and "lora" not in p for p in patterns)

    frozen = optim.get("frozen_module_names", None)
    if frozen is not None:
        new_frozen = list(frozen)
        if use_lora:
            # LoRA: the original frame & global blocks are always frozen (only the
            # adapters train) — ensure both are in the frozen list.
            if not _has_orig(new_frozen, "frame_blocks"):
                new_frozen.append("*aggregator.frame_blocks*")
            if not _has_orig(new_frozen, "global_blocks"):
                new_frozen.append("*aggregator.global_blocks*")
        elif light_token:
            # full + light: freeze the per-frame blocks, train ONLY the global blocks.
            new_frozen = [p for p in new_frozen if ("global_blocks" not in p)]
            if not _has_orig(new_frozen, "frame_blocks"):
                new_frozen.append("*aggregator.frame_blocks*")
        else:
            # full: train both frame & global blocks directly — unfreeze them.
            new_frozen = [
                p for p in new_frozen
                if ("frame_blocks" not in p) and ("global_blocks" not in p)
            ]
        OmegaConf.update(cfg, "optim.frozen_module_names", new_frozen, merge=False)

    # --- Adjust gradient_clip configs ---
    gc = optim.get("gradient_clip", None)
    if gc is None:
        return
    gc_configs = gc.get("configs", None)
    if gc_configs is None:
        return

    def _names(c):
        mn = c.get("module_name", [])
        return [mn] if isinstance(mn, str) else list(mn)

    new_gc_configs = []
    for c in gc_configs:
        names = _names(c)
        is_lora_entry = any("lora_frame_blocks" in n or "lora_global_blocks" in n for n in names)
        is_frame_lora_entry = any("lora_frame_blocks" in n for n in names)
        is_block_entry = any(
            ("frame_blocks" in n and "lora_" not in n) or
            ("global_blocks" in n and "lora_" not in n)
            for n in names
        )
        if use_lora:
            # Drop any direct frame/global block clip entries (those layers are frozen).
            if is_block_entry and not is_lora_entry:
                continue
            # Light-token training: frame LoRA is not built, so drop its clip entry.
            if light_token and is_frame_lora_entry:
                continue
        else:
            # Drop lora_* entries (those layers don't exist in full mode).
            if is_lora_entry:
                continue
        new_gc_configs.append(c)

    if not use_lora:
        existing_names_flat = [n for c in new_gc_configs for n in _names(c)]
        # Full mode trains the original blocks, so they need clip entries — except the
        # frame blocks stay frozen during light-token training, so only add global then.
        if not light_token and not any("frame_blocks" in n for n in existing_names_flat):
            new_gc_configs.append(OmegaConf.create({
                "module_name": ["aggregator.frame_blocks"],
                "max_norm": 1.0,
                "norm_type": 2,
            }))
        if not any("global_blocks" in n for n in existing_names_flat):
            new_gc_configs.append(OmegaConf.create({
                "module_name": ["aggregator.global_blocks"],
                "max_norm": 1.0,
                "norm_type": 2,
            }))

    OmegaConf.update(cfg, "optim.gradient_clip.configs", new_gc_configs, merge=False)


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    with initialize(version_base=None, config_path="config"):
        # cfg = compose(config_name=args.config)
        cfg = compose(config_name=args.config, overrides=unknown)

    _apply_finetune_mode(cfg)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()



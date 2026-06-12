#!/usr/bin/env python3
"""Backward-compatible entrypoint for inverse rendering training.

Supports legacy usage:
  torchrun ... train_inverse_rendering.py --config training/config/inverse_rendering.yaml <overrides>

Internally forwards to training/launch.py which expects:
  --config inverse_rendering <overrides>
"""

import argparse
import os
import runpy
import sys


def _normalize_config_name(config_arg: str) -> str:
    config_name = config_arg
    if config_name.endswith(".yaml"):
        config_name = os.path.basename(config_name)[:-5]
    return config_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy inverse rendering trainer entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/inverse_rendering.yaml",
        help="Config path or config name.",
    )
    args, unknown = parser.parse_known_args()

    config_name = _normalize_config_name(args.config)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(root_dir, "training")
    launch_path = os.path.join(training_dir, "launch.py")

    # `training/launch.py` imports `trainer` as a top-level module.
    if training_dir not in sys.path:
        sys.path.insert(0, training_dir)

    # Rebuild argv for training/launch.py
    sys.argv = [launch_path, "--config", config_name, *unknown]
    runpy.run_path(launch_path, run_name="__main__")


if __name__ == "__main__":
    main()

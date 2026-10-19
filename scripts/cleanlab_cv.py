#!/usr/bin/env python3
"""
CLI script to run K-Fold cross validation and dataset error auditing with Cleanlab.

Usage:
    python scripts/cleanlab_cv.py --config configs/cleanlab.yaml
    python scripts/cleanlab_cv.py --test --test-fraction 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plant_pheno.cleanlab import load_cleanlab_config, run_cleanlab_cv
from plant_pheno.infra import init_logger

# Add project root to sys.path so plant_pheno can be imported

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run K-Fold cross-validation and clean dataset using Cleanlab."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cleanlab.yaml"),
        help="Path to cleanlab configuration YAML (default: configs/cleanlab.yaml)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with a subsampled dataset fraction",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.05,
        help="Dataset fraction to use in test mode (default: 0.05)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override model backbone (e.g. 'efficientnet', 'bioclip')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs per fold",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Override number of cross-validation folds",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = PROJECT_ROOT / "log.log"
    init_logger(log_path)

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading configuration from {config_path}...")
    cfg_dict = load_cleanlab_config(config_path)

    # Apply overrides if provided
    if args.backbone:
        cfg_dict.setdefault("model_params", {})["backbone"] = args.backbone
    if args.epochs is not None:
        cfg_dict.setdefault("training_params", {})["epochs"] = args.epochs
    if args.n_folds is not None:
        cfg_dict.setdefault("cleanlab_params", {})["n_folds"] = args.n_folds

    # If any CLI overrides were passed, save temporary config or pass modified
    temp_cfg_path = config_path
    if args.backbone or args.epochs is not None or args.n_folds is not None:
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
            yaml.safe_dump(cfg_dict, tf)
            temp_cfg_path = Path(tf.name)

    try:
        results = run_cleanlab_cv(
            config_path=temp_cfg_path,
            test_mode=args.test,
            test_fraction=args.test_fraction,
        )
        print("\nCleanlab CV finished successfully!")
        print(f"Processed {len(results['obs_ids'])} observations.")
    finally:
        if temp_cfg_path != config_path and temp_cfg_path.exists():
            temp_cfg_path.unlink()


if __name__ == "__main__":
    main()

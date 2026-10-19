#!/usr/bin/env python3
"""
CLI script to run K-Fold cross validation and dataset error auditing with Cleanlab.

Usage:
    python scripts/cleanlab_cv.py --env local
    python scripts/cleanlab_cv.py --test --test-fraction 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so plant_pheno can be imported

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402

from plant_pheno.cleanlab import run_cleanlab_cv  # noqa: E402
from plant_pheno.config import CONFIG_DIR  # noqa: E402
from plant_pheno.infra import init_logger  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run K-Fold cross-validation and clean dataset using Cleanlab."
    )
    parser.add_argument(
        "--env",
        choices=["local", "colab"],
        default="local",
        help="Machine config group (default: local)",
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

    overrides = [
        "command=cleanlab",
        f"env={args.env}",
        f"test={str(args.test).lower()}",
        f"test_frac={args.test_fraction}",
    ]
    if args.backbone:
        overrides.append(f"model.backbone={args.backbone}")
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.n_folds is not None:
        overrides.append(f"cleanlab.n_folds={args.n_folds}")

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=overrides)

    results = run_cleanlab_cv(cfg)
    print("\nCleanlab CV finished successfully!")
    print(f"Processed {len(results['obs_ids'])} observations.")


if __name__ == "__main__":
    main()

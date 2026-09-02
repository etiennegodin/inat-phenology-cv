from __future__ import annotations

import pprint
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor, device, nn

if TYPE_CHECKING:
    from ..train.dataset import DatasetParams, PhenologyDataset

import logging
import os
import subprocess

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def get_mlflow_run_id() -> str | None:
    import mlflow

    return mlflow.active_run().info.run_id if mlflow.active_run() else None


def get_current_git_branch():
    try:
        # Runs the git command and decodes the byte output to a string
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        return branch
    except subprocess.CalledProcessError:
        return "Not a git repository or Git is not installed."


def format_dict(d: dict) -> str:
    return pprint.pformat(d, indent=4)


def resolve_uri() -> str:
    uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )
    logger.debug(uri)
    return uri


def resolve_env_config_path() -> Path:
    if os.environ.get("COLAB_GPU"):
        config_path = Path("configs/colab.yaml")
    else:
        config_path = Path("configs/local.yaml")
    logger.debug(f"Resolved config path '{config_path}'")
    return config_path


def unfreeze(block: nn.Module):
    count = 0
    for p in block.parameters(recurse=True):
        p.requires_grad = True
        count += p.numel()
    logger.debug(f"Unfreezed {count}")


def get_pos_weights(
    train: PhenologyDataset, params: DatasetParams, device: device
) -> Tensor:
    df = train.df
    total = np.array([df.shape[0], df.shape[0], df.shape[0]])
    label_pos = np.sum(df[params.label_col].tolist(), axis=0)
    label_neg = np.subtract(total, label_pos)
    return Tensor(np.divide(label_neg, label_pos)).to(device)


def get_pos_ratios(dataset):
    ratios = []
    if not hasattr(dataset, "df") or dataset.df is None:
        return []

    df = dataset.df
    labels_list = df["label"].tolist() if "label" in df.columns else []
    if not labels_list:
        return []
    labels_arr = np.array(labels_list)
    total_obs = len(labels_arr)
    pos_counts = np.sum(labels_arr, axis=0)

    for i, pos_c in enumerate(pos_counts):
        ratios.append(round(float(pos_c / total_obs), 4))
    return ratios


def clean_data(image_dir: str):
    for image_dir, dirs, files in os.walk(image_dir):
        for file in files:
            path = os.path.join(image_dir, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                logger.info(f"Deleting corrupted image: {path}")
                os.remove(path)

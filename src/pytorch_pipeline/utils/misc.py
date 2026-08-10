from __future__ import annotations

import pprint
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor, device, nn

if TYPE_CHECKING:
    from ..train.dataset import DatasetParams, PhenologyDataset

import logging
import os

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


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

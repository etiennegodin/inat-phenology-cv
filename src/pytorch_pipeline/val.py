from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.amp import autocast_mode, grad_scaler
from tqdm import tqdm

if TYPE_CHECKING:
    from torch import Tensor, device
    from torch.utils.data import DataLoader

    from .train.model import PhenologyModel

logger = logging.getLogger(__name__)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = None
if dtype == torch.float16:
    scaler = grad_scaler.GradScaler()


def execute(
    model: PhenologyModel,
    dataloader: DataLoader,
    device: device,
    as_numpy: bool = False,
) -> tuple[np.ndarray | list[float], np.ndarray | list[float]]:
    """Evaluate the model on validation set with rich multi-label metrics."""

    all_labels = []
    all_preds_raw = []
    all_obs_ids = []

    model.eval()

    pbar = tqdm(
        dataloader,
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for step, (images, labels, obs_ids) in enumerate(pbar):
            all_obs_ids.extend(obs_ids)
            labels: Tensor
            images = [t.to(device) for t in images]
            labels = labels.to(device)

            if device.type == "cuda":
                with autocast_mode.autocast(device_type=device.type, dtype=dtype):
                    predictions, class_weights = model(images)
                    torch.cuda.synchronize()
            else:
                predictions, class_weights = model(images)

            preds_raw = torch.sigmoid(predictions)

            all_labels.append(labels.detach().float().cpu())
            all_preds_raw.append(preds_raw.detach().float().cpu())

    if as_numpy:
        all_labels_np = torch.cat(all_labels).numpy()
        all_preds_raw_np = torch.cat(all_preds_raw).numpy()
        return all_labels_np, all_preds_raw_np
    return all_labels, all_preds_raw

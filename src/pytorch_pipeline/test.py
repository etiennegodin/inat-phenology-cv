from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.amp import autocast_mode

from .train.metrics import get_metrics

if TYPE_CHECKING:
    from torch import Tensor, device, nn
    from torch.utils.data import DataLoader


def execute(
    device: device,
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
):

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_preds_raw = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            labels: Tensor
            images = [t.to(device) for t in images]
            labels = labels.to(device)

            # Run foward prop and backprop
            if device.type == "cuda":
                with autocast_mode.autocast(device_type=device.type):
                    predictions, weights = model(images)
            else:
                predictions, weights = model(images)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            preds_raw = torch.sigmoid(predictions)
            preds = (preds_raw > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds)
            all_labels.append(labels)
            all_preds_raw.append(preds_raw)

    # Convert back to numpy arrays for metrics
    all_preds = torch.cat(all_preds).detach().cpu().numpy()
    all_labels = torch.cat(all_labels).detach().cpu().numpy()
    all_preds_raw = torch.cat(all_preds_raw).detach().cpu().numpy()

    eval_metrics = get_metrics(all_preds, all_labels, all_preds_raw)

    print(
        f"accuracy={float(correct / total):.3f} "
        f"roc={float(eval_metrics['val_roc_auc']):.3f}"
    )

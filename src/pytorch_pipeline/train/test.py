from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import device, nn
    from torch.utils.data import DataLoader

from .workflow import evaluate


def execute(
    device: device,
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
):

    epoch = 0

    eval_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        epoch=epoch,
    )

    print(
        f"Epoch {epoch}: test={eval_metrics['val_loss']:.3f} "
        f"accuracy={float(eval_metrics['val_accuracy']):.3f} "
        f"roc={float(eval_metrics['val_roc_auc']):.3f}"
    )

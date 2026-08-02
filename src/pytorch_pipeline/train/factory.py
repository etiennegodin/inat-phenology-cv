from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..utils.misc import freeze, unfreeze
from .model import PhenologyModel

if TYPE_CHECKING:
    from torch import nn, optim

    from ..utils.params import ModelParams, OptimizerParams, SchedulerParams


def get_device() -> torch.device:
    d = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {d}")
    return torch.device(d)


def build_pipeline_model(
    device: torch.device, model_params: ModelParams
) -> PhenologyModel:
    """Instantiate model and unfreezes backbone last params

    Returns:
        nn.Module: _description_
    """
    model = PhenologyModel(model_params)

    # Freeze backbone
    for block in model.backbone.get_trainable_blocks():
        freeze(block)

    # Unfreeze last block
    for block in model.backbone.get_trainable_blocks()[-model_params.last_blocks :]:
        unfreeze(block)

    model.to(device)
    return model


def build_scheduler(optimizer: optim.Optimizer, params: SchedulerParams, eta_min=1e-7):
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=params.warmup_epochs,
    )
    decay = CosineAnnealingLR(
        optimizer,
        T_max=params.total_epoch - params.warmup_epochs,
        eta_min=eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[params.warmup_epochs],
    )


def build_pipeline_optimizer(
    model: nn.Module, params: OptimizerParams
) -> optim.Optimizer:
    return optim.Adam(
        [
            {
                "params": [
                    p for p in model.backbone.encoder.parameters() if p.requires_grad
                ],
                "lr": params.backbone_lr,
            },
            {
                "params": model.attention.parameters(),
                "lr": params.attention_lr,
            },
            {
                "params": model.head.parameters(),
                "lr": params.head_lr,
            },
        ]
    )

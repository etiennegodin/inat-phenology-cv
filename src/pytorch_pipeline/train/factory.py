from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import torch
from torch import optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from ..utils.misc import unfreeze
from .model import PhenologyModel

if TYPE_CHECKING:
    from torch import nn, optim

    from ..utils.params import ModelParams, OptimizerParams, SchedulerParams

logger = logging.getLogger(__name__)


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

    blocks = model.backbone.get_trainable_blocks()

    # Unfreeze last block
    if model_params.last_blocks > 0:
        logger.debug("Unfreezing backbone parameters")
        for block in blocks[-model_params.last_blocks :]:
            unfreeze(block)

    # Show which blocks i trainable
    for i, block in enumerate(blocks):
        trainable = any(p.requires_grad for p in block.parameters())
        logger.debug(f"Block {i}: trainable={trainable}")

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

    attention_params = itertools.chain.from_iterable(
        branch.attention.parameters() for branch in model.branches
    )
    head_params = itertools.chain.from_iterable(
        branch.head.parameters() for branch in model.branches
    )

    # to_do if branches should converge at different rates
    # build 3 separate attention param groups (branch.attention.parameters() per branch
    # each with its own named LR) instead of pooling.
    return optim.Adam(
        [
            {
                "params": [
                    p for p in model.backbone.encoder.parameters() if p.requires_grad
                ],
                "lr": params.backbone_lr,
            },
            {
                "params": attention_params,
                "lr": params.attention_lr,
            },
            {
                "params": head_params,
                "lr": params.head_lr,
            },
        ]
    )

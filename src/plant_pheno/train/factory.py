from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

from torch import optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

if TYPE_CHECKING:
    from torch import nn, optim

    from ..config.params import OptimizerParams, SchedulerParams

logger = logging.getLogger(__name__)


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
                "name": "backbone",
                "params": [
                    p for p in model.backbone.encoder.parameters() if p.requires_grad
                ],
                "lr": params.backbone_lr,
            },
            {
                "name": "attention",
                "params": attention_params,
                "lr": params.attention_lr,
            },
            {
                "name": "classifier_head",
                "params": head_params,
                "lr": params.head_lr,
            },
        ]
    )

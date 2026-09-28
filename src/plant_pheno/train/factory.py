from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from ..config import Config
from ..core.backbone import BACKBONE_REGISTRY
from ..utils import get_pos_ratios, get_pos_weights
from .dataloader import build_pipeline_dataloaders
from .dataset import build_datasets

if TYPE_CHECKING:
    from ..config.params import OptimizerParams, SchedulerParams

logger = logging.getLogger(__name__)


@dataclass
class DataPipeline:
    datasets: tuple
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    criterion: nn.Module
    pos_ratios: list[float]


def build_data_pipeline(
    configs: Config, backbone_name: str, device, seed: int = 42
) -> DataPipeline:
    """Builds datasets, dataloaders, criterion, and pos_ratios once."""
    # Transforms come directly from the backbone
    # registry without needing the full model
    transforms = BACKBONE_REGISTRY[backbone_name]().get_transforms()
    datasets = build_datasets(configs, transforms, seed=seed)

    pos_weights = get_pos_weights(datasets[0], configs.dataset_params, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
    pos_ratios = get_pos_ratios(datasets[1])

    train_loader, val_loader, test_loader = build_pipeline_dataloaders(
        datasets, configs.dataloaders_params, seed=seed
    )

    return DataPipeline(
        datasets=datasets,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        pos_ratios=pos_ratios,
    )


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

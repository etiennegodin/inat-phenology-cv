from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import optim as optim
from torch.utils.data import DataLoader, Subset

from ..utils.registry import EFFICIENT_NET_LAST_BLOCK
from .dataloader import collate_fn
from .dataset import build_datasets
from .model import PhenologyModel

if TYPE_CHECKING:
    from torch import nn, optim

    from ..utils import Config
    from ..utils.params import ModelParams, OptimizerParams


def get_device() -> torch.device:
    d = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {d}")
    return torch.device(d)


def build_pipeline_model(device: torch.device, model_params: ModelParams) -> nn.Module:
    """Instantiate model and unfreezes backbone last params

    Returns:
        nn.Module: _description_
    """
    model = PhenologyModel(model_params)
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Unfreeze last block
    for name, p in model.backbone.named_parameters():
        if name.startswith(tuple(EFFICIENT_NET_LAST_BLOCK)):
            p.requires_grad = True

    model.to(device)
    return model


def build_pipeline_optimizer(
    model: nn.Module, params: OptimizerParams
) -> optim.Optimizer:
    return optim.Adam(
        [
            {
                "params": [p for p in model.backbone.parameters() if p.requires_grad],
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


def build_pipeline_dataloaders(
    config: Config, model: nn.Module
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = build_datasets(
        paths=config.paths,
        samples_params=config.samples_params,
        model_configs=model.backbone.default_cfg,
    )

    if config.test:
        n = 100
        train_set = Subset(train_set, range(min(n, len(train_set))))
        val_set = Subset(val_set, range(min(n, len(val_set))))
        test_set = Subset(test_set, range(min(n, len(test_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=config.dataloaders_params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.dataloaders_params.num_workers,
        pin_memory=config.dataloaders_params.pin_memory,
        persistent_workers=config.dataloaders_params.persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.dataloaders_params.batch_size,
        collate_fn=collate_fn,
        num_workers=config.dataloaders_params.num_workers,
        pin_memory=config.dataloaders_params.pin_memory,
        persistent_workers=config.dataloaders_params.persistent_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.dataloaders_params.batch_size,
        collate_fn=collate_fn,
        num_workers=config.dataloaders_params.num_workers,
        pin_memory=config.dataloaders_params.pin_memory,
        persistent_workers=config.dataloaders_params.persistent_workers,
    )
    return train_loader, val_loader, test_loader

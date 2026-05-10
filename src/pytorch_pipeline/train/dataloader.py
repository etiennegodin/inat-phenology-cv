from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Subset

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from ..utils import Config


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    images = []
    labels = []
    for images_tensor, label in batch:
        labels.append(label)
        images.append(images_tensor)
    return images, torch.stack(labels)


def build_pipeline_dataloaders(
    datasets: tuple[Dataset, Dataset, Dataset], config: Config
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_set = datasets[0]
    val_set = datasets[1]
    test_set = datasets[2]

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

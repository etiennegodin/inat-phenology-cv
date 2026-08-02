from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from .batch_sampler import MaxImagesBatchSampler

if TYPE_CHECKING:
    from ..utils import Config
    from .dataset import PhenologyDataset


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    images = []
    labels = []
    for images_tensor, label in batch:
        labels.append(label)
        images.append(images_tensor)
    return images, torch.stack(labels)


def build_pipeline_dataloaders(
    datasets: tuple[PhenologyDataset, PhenologyDataset, PhenologyDataset],
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_set = datasets[0]
    val_set = datasets[1]
    test_set = datasets[2]

    train_sampler = MaxImagesBatchSampler(
        train_set.bag_sizes,
        max_images=config.dataloaders_params.max_images,
    )
    val_sampler = MaxImagesBatchSampler(
        val_set.bag_sizes,
        max_images=config.dataloaders_params.max_images,
    )
    test_sampler = MaxImagesBatchSampler(
        test_set.bag_sizes,
        max_images=config.dataloaders_params.max_images,
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=config.dataloaders_params.num_workers,
        pin_memory=config.dataloaders_params.pin_memory,
        persistent_workers=config.dataloaders_params.persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=config.dataloaders_params.num_workers,
        pin_memory=config.dataloaders_params.pin_memory,
        persistent_workers=config.dataloaders_params.persistent_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        collate_fn=collate_fn,
        num_workers=config.dataloaders_params.num_workers,
        pin_memory=config.dataloaders_params.pin_memory,
        persistent_workers=config.dataloaders_params.persistent_workers,
    )
    return train_loader, val_loader, test_loader

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from ..infra.seed import seed_worker
from .batch_sampler import MaxImagesBatchSampler

if TYPE_CHECKING:
    from ..config import DataLoadersParams
    from .dataset import PhenologyDataset


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    images = []
    labels = []
    obs_ids = []
    for images_tensor, label, obs_id in batch:
        labels.append(label)
        images.append(images_tensor)
        obs_ids.append(obs_id)

    return images, torch.stack(labels), obs_ids


def build_single_dataloader(
    params: DataLoadersParams,
    dataset: PhenologyDataset,
    generator: torch.Generator,
    shuffle: bool,
    seed: int | None,
):

    if params.use_max_images:
        sampler = MaxImagesBatchSampler(
            dataset.bag_sizes,
            max_images=params.max_images,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            persistent_workers=params.persistent_workers,
            generator=generator,
            worker_init_fn=seed_worker,
        )
    return DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        persistent_workers=params.persistent_workers,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def build_pipeline_dataloaders(
    datasets: tuple[PhenologyDataset, PhenologyDataset, PhenologyDataset],
    params: DataLoadersParams,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    generator = torch.Generator().manual_seed(seed)
    loaders = []

    for i, set in enumerate(datasets):
        loaders.append(
            build_single_dataloader(
                dataset=set,
                shuffle=True if i == 0 else False,
                seed=seed if i == 0 else None,
                generator=generator,
                params=params,
            )
        )

    return tuple(loaders)

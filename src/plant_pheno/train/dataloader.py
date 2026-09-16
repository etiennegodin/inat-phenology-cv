from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from ..utils.seed import seed_worker
from .batch_sampler import MaxImagesBatchSampler

if TYPE_CHECKING:
    from ..utils.params import DataLoadersParams
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


def build_pipeline_dataloaders(
    datasets: tuple[PhenologyDataset, PhenologyDataset, PhenologyDataset],
    params: DataLoadersParams,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:

    generator = torch.Generator().manual_seed(seed)

    if params.use_max_images:
        samplers = []
        loaders = []
        for i, set in enumerate(datasets):
            samplers.append(
                MaxImagesBatchSampler(
                    set.bag_sizes,
                    max_images=params.max_images,
                    shuffle=True if i == 0 else False,
                    seed=seed if i == 0 else None,
                )
            )

        for sampler, dataset in zip(samplers, datasets):
            loaders.append(
                DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=collate_fn,
                    num_workers=params.num_workers,
                    pin_memory=params.pin_memory,
                    persistent_workers=params.persistent_workers,
                    generator=generator,
                    worker_init_fn=seed_worker,
                )
            )
        return tuple(loaders)

    else:
        loaders = []
        for i, set in enumerate(datasets):
            loaders.append(
                DataLoader(
                    set,
                    batch_size=params.batch_size,
                    shuffle=True if i == 0 else False,
                    collate_fn=collate_fn,
                    num_workers=params.num_workers,
                    pin_memory=params.pin_memory,
                    persistent_workers=params.persistent_workers,
                    generator=generator,
                    worker_init_fn=seed_worker,
                )
            )

        return tuple(loaders)

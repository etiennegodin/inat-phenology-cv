from __future__ import annotations

import logging

import torch
from torch.amp import autocast_mode, grad_scaler
from tqdm import tqdm

from .config import CachingParams, Config, ModelParams
from .core.backbone import build_backbone
from .core.device import get_device
from .train.dataloader import build_single_dataloader
from .train.dataset import build_full_dataset

logger = logging.getLogger(__name__)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = None
if dtype == torch.float16:
    scaler = grad_scaler.GradScaler()


def execute(
    configs: Config,
    caching_params: CachingParams,
    model_params: ModelParams,
    seed: int,
):

    # Initialise train modules
    device = get_device()

    # Build backbone
    backbone = build_backbone(
        model_params.backbone, model_params.dinov3.checkpoint_path
    )

    # Build full dataset without splits
    dataset = build_full_dataset(configs, backbone.get_transforms(), seed=seed)

    # Instantiate generator for dataloader
    generator = torch.Generator().manual_seed(seed)

    # Create datalaoder from full dataset
    dataloader = build_single_dataloader(
        params=configs.dataloaders_params,
        dataset=dataset,
        generator=generator,
        shuffle=False,
        seed=seed,
    )

    # Alter backbone from cache params

    pbar = tqdm(
        dataloader,
        desc="Caching  ",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for step, (images, _, _) in enumerate(pbar):
            images = [t.to(device) for t in images]

            # Concatenate all images in one Tensor
            stacked = torch.cat(images)

            if device.type == "cuda":
                with autocast_mode.autocast(device_type=device.type, dtype=dtype):
                    embeddings = backbone.encode(stacked)
                    torch.cuda.synchronize()
            else:
                embeddings = backbone.encode(stacked)

            save_path = f"{configs.paths_params.data_root}/cached_embeddings/{model_params.backbone}/{step}.pt"
            torch.save(embeddings, save_path)

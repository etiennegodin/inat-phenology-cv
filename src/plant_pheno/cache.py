from __future__ import annotations

import logging
from pathlib import Path

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

    # Resolve truncation depth.
    # to_layer=0 → full depth (None); to_layer=k → skip last k blocks.
    total_blocks = backbone.trainable_block_count
    if caching_params.to_layer <= 0:
        n_blocks: int | None = None
    else:
        n_blocks = total_blocks - caching_params.to_layer
        if n_blocks < 1:
            raise ValueError(
                f"to_layer={caching_params.to_layer} would leave 0 blocks for backbone "
                f"'{model_params.backbone}' which has {total_blocks} trainable blocks. "
                f"Valid range: 0 (full) to {total_blocks - 1}."
            )

    # Choose CLS vs patch encoding based on architecture
    use_patches = model_params.architecture == "cross_attention"
    if use_patches and not backbone.has_patch_support:
        raise ValueError(
            f"Architecture 'cross_attention' requires patch support, "
            f"but backbone '{model_params.backbone}' does not support it."
        )

    kind = "patches" if use_patches else "cls"
    layer_tag = str(n_blocks) if n_blocks is not None else "full"
    base_path = Path(
        f"{configs.paths_params.data_root}/cached_embeddings"
        f"/{model_params.backbone}/{kind}/to_layer_{layer_tag}"
    )
    base_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Caching '{kind}' embeddings for '{model_params.backbone}' "
        f"(n_blocks={n_blocks if n_blocks is not None else 'full'}) → {base_path}"
    )

    pbar = tqdm(
        dataloader,
        desc="Caching  ",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for step, (images, _, obs_ids) in enumerate(pbar):
            images = [t.to(device) for t in images]

            # Concatenate all images in one Tensor; bag boundaries are saved separately
            stacked = torch.cat(images)

            if device.type == "cuda":
                with autocast_mode.autocast(device_type=device.type, dtype=dtype):
                    if use_patches:
                        embeddings = backbone.encode_patches(stacked, n_blocks=n_blocks)
                    else:
                        embeddings = backbone.encode(stacked, n_blocks=n_blocks)
                    torch.cuda.synchronize()
            else:
                if use_patches:
                    embeddings = backbone.encode_patches(stacked, n_blocks=n_blocks)
                else:
                    embeddings = backbone.encode(stacked, n_blocks=n_blocks)

            payload = {
                "embeddings": embeddings.cpu(),
                "bag_sizes": [t.size(0) for t in images],
                "obs_ids": obs_ids,
                "kind": kind,
                "to_layer": layer_tag,
            }
            torch.save(payload, base_path / f"{step}.pt")

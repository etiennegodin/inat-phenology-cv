from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.amp import autocast_mode, grad_scaler
from tqdm import tqdm

from .config import CacheParams, Config, ModelParams, PathsParams
from .core.backbone import build_backbone, resolve_truncation_depth
from .core.device import get_device
from .data import LocalTorchSaver
from .train.dataloader import build_single_dataloader
from .train.dataset import build_full_dataset
from .utils import cuda_cleanup

logger = logging.getLogger(__name__)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = None
if dtype == torch.float16:
    scaler = grad_scaler.GradScaler()


@dataclass
class CachedEmbeddingSchema:
    model_backbone: str
    to_layer: int
    views: int
    layer_tag: str
    model_kind: str
    cache_dir: Path = field(init=False)

    @classmethod
    def from_params(cls, cache_params: CacheParams, model_params: ModelParams):
        to_layer = cache_params.cache_to_layer
        views = cache_params.cache_views
        return cls(
            model_backbone=model_params.backbone,
            to_layer=to_layer,
            views=views,
            layer_tag=str(to_layer) if to_layer > 0 else "full",
            model_kind="patches"
            if model_params.architecture == "cross_attention"
            else "cls",
        )

    def resolve_cache_dir(self, paths: PathsParams) -> Path:
        base_path = Path(
            f"{paths.data_root}/cached_embeddings"
            f"/{self.model_backbone}/{self.model_kind}/to_layer_{self.layer_tag}"
        )
        base_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = base_path
        return base_path

    def to_dict(self):
        return asdict(self)


def execute(
    configs: Config,
    model_params: ModelParams,
    seed: int,
):

    # Initialise train modules
    device = get_device()

    # Cached schema
    cache_params = configs.dataset_params.cache
    cache_schema = CachedEmbeddingSchema.from_params(
        cache_params, model_params=model_params
    )
    base_path = cache_schema.resolve_cache_dir(configs.paths_params)

    # Build backbone
    backbone = build_backbone(
        model_params.backbone, model_params.dinov3.checkpoint_path, device=device
    )

    # Resolve truncation depth.
    # to_layer=0 → full depth (None); to_layer=k → skip last k blocks.
    total_blocks = backbone.trainable_block_count
    n_blocks = resolve_truncation_depth(
        total_blocks=total_blocks,
        to_layer=cache_params.cache_to_layer,
        backbone_name=model_params.backbone,
    )

    # Choose CLS vs patch encoding based on architecture
    use_patches = model_params.architecture == "cross_attention"
    if use_patches and not backbone.has_patch_support:
        raise ValueError(
            f"Architecture 'cross_attention' requires patch support, "
            f"but backbone '{model_params.backbone}' does not support it."
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

    logger.info(
        f"Caching '{cache_schema.model_kind}' embeddings for "
        f"'{cache_schema.model_backbone}' "
        f"(n_blocks={n_blocks if n_blocks is not None else 'full'}) → {base_path}"
    )

    pbar = tqdm(
        dataloader,
        desc="Caching  ",
        leave=False,
        dynamic_ncols=True,
    )
    with LocalTorchSaver(max_pending=2) as saver:
        with torch.no_grad():
            for step, (images, _, obs_ids) in enumerate(pbar):
                save_file = base_path / f"{step}.pt"
                if save_file.exists():
                    continue

                with cuda_cleanup():
                    images = [t.to(device) for t in images]

                    # Concatenate all images in one Tensor;
                    # bag boundaries are saved separately
                    stacked = torch.cat(images)
                    t0 = time.perf_counter()

                    if device.type == "cuda":
                        with autocast_mode.autocast(
                            device_type=device.type, dtype=dtype
                        ):
                            if use_patches:
                                embeddings = backbone.encode_patches(
                                    stacked, n_blocks=n_blocks
                                )
                            else:
                                embeddings = backbone.encode(stacked, n_blocks=n_blocks)
                            torch.cuda.synchronize()
                    else:
                        if use_patches:
                            embeddings = backbone.encode_patches(
                                stacked, n_blocks=n_blocks
                            )
                        else:
                            embeddings = backbone.encode(stacked, n_blocks=n_blocks)
                    t1 = time.perf_counter()

                    payload = {
                        "embeddings": embeddings.cpu(),
                        "bag_sizes": [t.size(0) for t in images],
                        "obs_ids": obs_ids,
                        "kind": cache_schema.model_kind,
                        "backbone": cache_schema.model_backbone,
                    }

                    saver.save(
                        payload=payload,
                        path=save_file,
                    )
                    t2 = time.perf_counter()
                    logger.debug(f"compute {t1 - t0:.1f}s  save {t2 - t1:.1f}s")

                    # Explicitly delete large tensors to help garbage collection
                    del images, stacked, embeddings


def ensure_cached_embeddings(configs: Config, model_params: ModelParams, seed) -> Path:
    """
    Checks for local cached embeddings matching (backbone, architecture, to_layer).
    If missing (or dataset.force_recache=true),
    it calls cache.execute() and generates them.
    """

    cache_params = configs.dataset_params.cache

    cache_schema = CachedEmbeddingSchema.from_params(
        cache_params, model_params=model_params
    )
    cache_dir = cache_schema.resolve_cache_dir(configs.paths_params)

    if not cache_params.force_recache:
        if cache_dir.exists() and cache_dir.is_dir():
            if any(cache_dir.glob("*.pt")):
                logger.info(f"Found cached .pt files on disk under {cache_dir}")
                return cache_dir
            logger.info(f"Found no cached embeddings in {cache_dir}. Launching process")
    else:
        logger.info("Force recaching embeddings")

    # Caching process
    execute(
        configs=configs,
        model_params=model_params,
        seed=seed,
    )

    return cache_dir

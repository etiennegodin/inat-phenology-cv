from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NewType

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from tqdm import tqdm

from ..config import DatasetParams, HardwareProfile
from ..data import get_df_from_table
from ..utils import df_img_to_path

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose

    from ..config import Config, DatasetParams

logger = logging.getLogger(__name__)

CacheDataset = NewType("CacheDataset", bool)


class PhenologyDataset(Dataset, ABC):
    def __init__(
        self, df: pd.DataFrame, transform, dataset_params: DatasetParams
    ) -> None:
        super().__init__()
        self.transform = transform
        self.dataset_params = dataset_params
        self.df = df

        # Create list of photo count per observation
        self.bag_sizes = self.df["path"].str.len().to_list()

    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_profile(
        cls,
        cache: CacheDataset,
        df: pd.DataFrame,
        transform,
        dataset_params: DatasetParams,
        shared_cache: CacheDatasetdPhenologyDataset | None = None,
    ):
        if cache:
            return CacheDatasetdPhenologyDataset(
                df, transform, dataset_params, shared_cache=shared_cache
            )
        return UncachedPhenologyDataset(df, transform, dataset_params)

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        pass

    def get_obs_paths_map(self) -> dict[int, str] | None:
        obs_paths_map = None
        if "path" in self.df.columns:
            idx_col = (
                "observation_id"
                if "observation_id" in self.df.columns
                else self.df.columns[0]
            )
            obs_paths_map = dict(zip(self.df[idx_col], self.df["path"]))
        return obs_paths_map


class UncachedPhenologyDataset(PhenologyDataset):
    def __init__(
        self, df: pd.DataFrame, transform, dataset_params: DatasetParams
    ) -> None:
        super().__init__(df, transform, dataset_params)

    def __getitem__(self, index: int):
        obs_id = int(self.df.iat[index, 0])
        paths, target = self.df.iloc[index][["path", self.dataset_params.label_col]]
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
                images.append(image)
        return torch.stack(images), torch.tensor(target, dtype=torch.float32), obs_id


class CacheDatasetdPhenologyDataset(PhenologyDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        dataset_params: DatasetParams,
        shared_cache: CacheDatasetdPhenologyDataset | None = None,
    ) -> None:
        super().__init__(df, transform, dataset_params)
        if shared_cache is None:
            self.images, self.targets = self.preload_images_to_ram(df, dataset_params)
        else:
            self._reuse_shared_cache(shared_cache)

    def _reuse_shared_cache(self, shared_cache: CacheDatasetdPhenologyDataset) -> None:
        cache_indices = {
            int(obs_id): index
            for index, obs_id in enumerate(shared_cache.df[self.dataset_params.idx_col])
        }
        selected_indices = [
            cache_indices[int(obs_id)]
            for obs_id in self.df[self.dataset_params.idx_col]
        ]
        self.images = [shared_cache.images[index] for index in selected_indices]
        self.targets = [shared_cache.targets[index] for index in selected_indices]

    def __getitem__(self, index: int):
        obs_id = int(self.df.iat[index, 0])
        images = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            images = [self.transform(i) for i in images]
        return torch.stack(images), target, obs_id

    def preload_images_to_ram(self, df, params: DatasetParams):
        preloaded_images = []
        targets = []

        # Simple ToTensor conversion to get images into RAM as bytes/floats
        import torchvision.transforms.v2.functional as F

        print(f"Loading {len(df)} images into RAM... This may take a while...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            paths, target = (
                row["path"],
                row[params.label_col],
            )  # Adjust indices if your columns are named
            observation_tensors = []
            for path in paths:
                # Read image from local drive
                img = Image.open(path).convert("RGB")
                # Convert to tensor immediately to free PIL memory
                img_tensor = F.to_image(img)  # Keeps it as uint8 to save RAM
                observation_tensors.append(img_tensor)
            targets.append(torch.tensor(target, dtype=torch.float32))
            preloaded_images.append(observation_tensors)
        return preloaded_images, targets


def _prepare_build_dataset(
    configs: Config, seed: int = 42
) -> tuple[pd.DataFrame, CacheDataset]:
    """Reusable function to prepare dataframe and cache dataset
    for further datasets creation"""

    df = get_samples(configs.paths_params, configs.dataset_params)

    # Get mean image ratio on raw df
    mean_image_ratio = get_mean_img_ratio(df)

    # Decide if caching is possible with dataset size and current hardware
    cache = resolve_cache_decision(
        configs.hardware_profile, len(df), mean_image_ratio, configs.max_img_resolution
    )

    # Collapse df by observation
    df = (
        df.groupby(configs.dataset_params.idx_col)
        .agg({"path": list, configs.dataset_params.label_col: "first"})
        .reset_index(drop=False)
        .sort_values(by=configs.dataset_params.idx_col)
        .reset_index(drop=True)
    )

    # Reduce dataset size if testing
    if configs.test:
        df = reduce_dataset(
            df,
            fraction=configs.test_frac,
            params=configs.dataset_params,
            seed=seed,
        )

    return df, cache


def get_mean_img_ratio(df: pd.DataFrame):
    ratios = []
    for paths in df["path"]:
        image_paths = paths if isinstance(paths, (list, tuple)) else [paths]
        for path in image_paths:
            try:
                x, y = Image.open(path).size
            except UnidentifiedImageError:
                continue
            if x <= y:
                ratio = y / x
            else:
                ratio = x / y
            ratios.append(round(ratio, 3))
    return np.array(ratios).mean()


def resolve_cache_decision(
    hardware_profile: HardwareProfile,
    total_image_count: int,
    img_mean_ratio: float,
    max_resolution: int,
) -> CacheDataset:
    total_pixel = max_resolution * (max_resolution / img_mean_ratio)
    images_data_size = (total_pixel * 3 * total_image_count) / 1073741824
    ram_avail = max(
        hardware_profile.ram * 0.9 - 10.0, 1e-5
    )  # 90% of existing ram with 10 Gb buffer for running pipeline
    if images_data_size > ram_avail:
        return CacheDataset(False)
    return CacheDataset(True)


def split_dataset(
    df: pd.DataFrame, params: DatasetParams, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Collapse to one observation per row and sort deterministically
    obs_df = (
        df.drop_duplicates(subset=[params.idx_col])
        .sort_values(by=params.idx_col)
        .reset_index(drop=True)
    )
    train_idx = obs_df.sample(frac=params.train_frac, random_state=seed)
    val_idx = obs_df.drop(train_idx.index).sample(
        frac=params.val_frac / (1 - params.train_frac), random_state=seed
    )
    test_idx = obs_df.drop(train_idx.index).drop(val_idx.index)

    train_df = (
        df[df[params.idx_col].isin(train_idx[params.idx_col])]
        .sort_values(by=params.idx_col)
        .reset_index(drop=True)
    )
    val_df = (
        df[df[params.idx_col].isin(val_idx[params.idx_col])]
        .sort_values(by=params.idx_col)
        .reset_index(drop=True)
    )
    test_df = (
        df[df[params.idx_col].isin(test_idx[params.idx_col])]
        .sort_values(by=params.idx_col)
        .reset_index(drop=True)
    )
    return (train_df, val_df, test_df)


def get_samples(paths, params: DatasetParams) -> pd.DataFrame:
    df = get_df_from_table(paths.db_path, params.source_table)
    return df_img_to_path(
        df,
        img_dir=paths.image_dir,
        id_column=params.idx_col,
        photo_id_column=params.photo_idx_col,
    )


def reduce_dataset(
    df: pd.DataFrame, fraction: float, params: DatasetParams, seed: int = 42
) -> pd.DataFrame:
    # Sample obs id from fraction
    test_idx = df.sample(frac=fraction, random_state=seed)
    # Keep sampled observations
    df = df[df[params.idx_col].isin(test_idx[params.idx_col])].reset_index(drop=True)
    print(f"Test mode - keeping {fraction * 100:.0f}% of dataset")
    print(f"{test_idx.shape[0]} observations kept")
    return df


def build_full_dataset(
    configs: Config, transforms: tuple[Compose, Compose], seed: int = 42
):
    """Build full dataset without train,val,test splits"""

    df, cache = _prepare_build_dataset(configs, seed)
    train_transform, _ = transforms

    full_set = PhenologyDataset.from_profile(
        cache, df, train_transform, configs.dataset_params
    )
    logger.debug(f"Dataset type : {type(full_set)}")
    return full_set


def build_datasets(
    configs: Config, transforms: tuple[Compose, Compose], seed: int = 42
) -> tuple[PhenologyDataset, PhenologyDataset, PhenologyDataset]:

    df, cache = _prepare_build_dataset(configs, seed)

    # Create splits
    train_df, val_df, test_df = split_dataset(df, configs.dataset_params, seed=seed)

    # Get backbone specific transforms
    train_transform, val_transform = transforms

    train_set = PhenologyDataset.from_profile(
        cache, train_df, train_transform, configs.dataset_params
    )
    val_set = PhenologyDataset.from_profile(
        cache, val_df, val_transform, configs.dataset_params
    )
    test_set = PhenologyDataset.from_profile(
        cache, test_df, val_transform, configs.dataset_params
    )

    logger.debug(f"Dataset type : {type(train_set)}")

    return train_set, val_set, test_set


@dataclass
class CachedObs:
    obs_id: int
    embeddings: torch.Tensor
    label: torch.Tensor


class CachedEmbeddingDataset(Dataset):
    """Dataset that reads pre-computed backbone embeddings
    from disk and attaches labels."""

    def __init__(
        self,
        cache_dir: Path,
        obs_df: pd.DataFrame,
        dataset_params: DatasetParams,
    ) -> None:
        super().__init__()
        self.dataset_params = dataset_params
        self.df = obs_df
        self._obs: list[CachedObs] = self._load(cache_dir, obs_df, dataset_params)
        self.bag_sizes = [o.embeddings.size(0) for o in self._obs]

    def _load(
        self, cache_dir: Path, obs_df: pd.DataFrame, params: DatasetParams
    ) -> list[CachedObs]:
        if not cache_dir.exists():
            raise FileNotFoundError(
                f"Cached embeddings directory does not exist: '{cache_dir}'. "
                f"Please run 'python -m plant_pheno command=cache' first."
            )

        label_map = dict(zip(obs_df[params.idx_col], obs_df[params.label_col]))
        result = []
        pt_files = sorted(
            cache_dir.glob("*.pt"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        if not pt_files:
            raise FileNotFoundError(
                f"No cached embedding (*.pt) files found in '{cache_dir}'. "
                f"Please run 'python -m plant_pheno command=cache' first."
            )

        for pt_file in pt_files:
            payload = torch.load(pt_file, map_location="cpu")
            embeddings_all = payload["embeddings"].float()
            bag_sizes = payload["bag_sizes"]
            obs_ids = payload["obs_ids"]
            per_obs = torch.split(embeddings_all, bag_sizes)
            for emb, obs_id in zip(per_obs, obs_ids):
                if obs_id in label_map:
                    label = torch.tensor(label_map[obs_id], dtype=torch.float32)
                    result.append(CachedObs(obs_id=obs_id, embeddings=emb, label=label))
        return result

    def __len__(self) -> int:
        return len(self._obs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        obs = self._obs[index]
        return obs.embeddings, obs.label, obs.obs_id

    def get_obs_paths_map(self) -> dict[int, str] | None:
        if "path" in self.df.columns:
            idx_col = (
                self.dataset_params.idx_col
                if self.dataset_params.idx_col in self.df.columns
                else self.df.columns[0]
            )
            return dict(zip(self.df[idx_col], self.df["path"]))
        return None


def build_cached_datasets(
    configs: Config,
    cache_dir: Path,
    seed: int = 42,
) -> tuple[CachedEmbeddingDataset, CachedEmbeddingDataset, CachedEmbeddingDataset]:
    df, _ = _prepare_build_dataset(configs, seed)
    train_df, val_df, test_df = split_dataset(df, configs.dataset_params, seed=seed)

    train_set = CachedEmbeddingDataset(cache_dir, train_df, configs.dataset_params)
    val_set = CachedEmbeddingDataset(cache_dir, val_df, configs.dataset_params)
    test_set = CachedEmbeddingDataset(cache_dir, test_df, configs.dataset_params)

    logger.debug(
        f"Cached dataset counts: train={len(train_set)},"
        f"val={len(val_set)}, test={len(test_set)}"
    )
    return train_set, val_set, test_set

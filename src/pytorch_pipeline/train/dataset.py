from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NewType

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from tqdm import tqdm

from pytorch_pipeline.utils.params import DatasetParams

from ..utils import get_df_from_table
from ..utils.system import HardwareProfile

if TYPE_CHECKING:
    from ..utils.configs import Config
    from ..utils.params import DatasetParams
    from .model import PhenologyModel

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
    ):
        if cache:
            return CacheDatasetdPhenologyDataset(df, transform, dataset_params)
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
        self, df: pd.DataFrame, transform, dataset_params: DatasetParams
    ) -> None:
        super().__init__(df, transform, dataset_params)
        self.images, self.targets = self.preload_images_to_ram(df, dataset_params)

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


def get_mean_img_ratio(df: pd.DataFrame):
    ratios = []
    for path in df["path"]:
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

    # Collapse to one observation per row
    obs_df = df.drop_duplicates(subset=[params.idx_col])
    train_idx = obs_df.sample(frac=params.train_frac, random_state=seed)
    val_idx = obs_df.drop(train_idx.index).sample(
        frac=params.val_frac / (1 - params.train_frac), random_state=seed
    )
    test_idx = obs_df.drop(train_idx.index).drop(val_idx.index)

    train_df = df[df[params.idx_col].isin(train_idx[params.idx_col])]
    val_df = df[df[params.idx_col].isin(val_idx[params.idx_col])]
    test_df = df[df[params.idx_col].isin(test_idx[params.idx_col])]
    return (train_df, val_df, test_df)


def get_samples(paths, params: DatasetParams) -> pd.DataFrame:
    df = get_df_from_table(paths.db_path, "cv_photos2")
    df["path"] = paths.image_dir + "/" + df[params.photo_idx_col].astype(str) + ".jpg"
    return df


def reduce_dataset(df, params: DatasetParams, seed: int = 42) -> pd.DataFrame:
    # Sample obs id from fraction
    test_idx = df.sample(frac=params.testing_frac, random_state=seed)
    # Keep photos only from sampled observations
    df = df[df[params.idx_col].isin(test_idx[params.idx_col])]
    print(f"Test mode - keeping {params.testing_frac * 100}% of dataset")
    print(f"{test_idx.shape[0]} observations with {df.shape[0]} images")
    return df


def build_datasets(
    configs: Config, model: PhenologyModel, seed: int = 42
) -> tuple[PhenologyDataset, PhenologyDataset, PhenologyDataset]:

    df = get_samples(configs.paths_params, configs.dataset_params)

    # Reduce data set size if testing
    if configs.test:
        df = reduce_dataset(df, configs.dataset_params, seed=seed)

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
    )

    # Create splits
    train_df, val_df, test_df = split_dataset(df, configs.dataset_params, seed=seed)

    # Get backbone specific transforms
    train_transform, val_transform = model.backbone.get_transforms()

    train_set = PhenologyDataset.from_profile(
        cache, train_df, train_transform, configs.dataset_params
    )
    val_set = PhenologyDataset.from_profile(
        cache, val_df, val_transform, configs.dataset_params
    )
    test_set = PhenologyDataset.from_profile(
        cache, test_df, val_transform, configs.dataset_params
    )

    return train_set, val_set, test_set

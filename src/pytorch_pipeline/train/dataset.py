from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import get_df_from_table

if TYPE_CHECKING:
    from ..utils.config import Config
    from ..utils.params import DatasetParams


class PhenologyDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, transform, dataset_params: DatasetParams
    ) -> None:
        self.transform = transform
        self.dataset_params = dataset_params
        self.df = self._format_df(df)

        super().__init__()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        paths, target = self.df.iloc[index]
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
                images.append(image)
        return torch.stack(images), torch.tensor(target, dtype=torch.float32)

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(self.dataset_params.idx_col)
            .agg({"path": list, self.dataset_params.label_col: "first"})
            .reset_index(drop=True)
        )


def build_base_transforms(input_size: tuple, mean: tuple, std: tuple):
    return [
        transforms.Resize((input_size[1], input_size[2])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]


def build_transforms(
    base_transforms: list,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomPerspective(0.2),
            transforms.RandomAutocontrast(),
        ]
        + base_transforms
    )
    val_transform = transforms.Compose(base_transforms)
    return train_transform, val_transform


def split_dataset(
    df: pd.DataFrame, params: DatasetParams
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Collapse to one observation per row
    obs_df = df.drop_duplicates(subset=[params.idx_col])
    train_idx = obs_df.sample(frac=params.train_frac)
    val_idx = obs_df.drop(train_idx.index).sample(
        frac=params.val_frac / (1 - params.train_frac)
    )
    test_idx = obs_df.drop(train_idx.index).drop(val_idx.index)

    train_df = df[df[params.idx_col].isin(train_idx[params.idx_col])]
    val_df = df[df[params.idx_col].isin(val_idx[params.idx_col])]
    test_df = df[df[params.idx_col].isin(test_idx[params.idx_col])]
    return (train_df, val_df, test_df)


def get_samples(paths, params: DatasetParams) -> pd.DataFrame:
    df = get_df_from_table(paths.db_path, "cv_photos2")
    df["path"] = paths.image_dir + "/" + df[params.photo_idx_col].astype(str) + ".jpg"
    return df.sort_values(by=params.idx_col, ascending=True, axis=0)


def reduce_dataset(df, params: DatasetParams) -> pd.DataFrame:
    print(f"Test mode - keeping {params.testing_frac * 100}% of dataset")
    # Collapse to obs level
    test_df = df.drop_duplicates(subset=[params.idx_col])
    # Sample obs id from fraction
    test_idx = test_df.sample(frac=params.testing_frac)
    # Keep photos only from sampled observations
    return df[df[params.idx_col].isin(test_idx[params.idx_col])]


def build_datasets(
    config: Config, model_configs: dict
) -> tuple[PhenologyDataset, PhenologyDataset, PhenologyDataset]:
    df = get_samples(config.paths, config.dataset_params)

    # Reduce data set size if testing
    if config.test:
        df = reduce_dataset(df, config.dataset_params)

    train_df, val_df, test_df = split_dataset(df, config.dataset_params)

    base_transforms = build_base_transforms(
        **{k: model_configs[k] for k in ("input_size", "mean", "std")}
    )
    train_transform, val_transform = build_transforms(base_transforms)

    train_set = PhenologyDataset(train_df, train_transform, config.dataset_params)
    val_set = PhenologyDataset(val_df, val_transform, config.dataset_params)
    test_set = PhenologyDataset(test_df, val_transform, config.dataset_params)

    return train_set, val_set, test_set

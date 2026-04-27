import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import get_df_from_table

LABEL_MAPPING = {13: 1, 21: 0}


class PhenologyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, samples_params) -> None:
        self.transform = transform
        self.samples_params = samples_params
        self.df = self._format_df(df)
        super().__init__()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        path, target = self.df.iloc[index]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def _format_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["path", self.samples_params.label_col]].reset_index(drop=True)


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
    df: pd.DataFrame,
    idx_col: str = "observation_id",
    label_col: str = "controlled_value_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Collapse to one observation per row
    obs_df = df.drop_duplicates(subset=[idx_col])

    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        obs_df[idx_col],
        obs_df[label_col],
        test_size=0.40,
        random_state=42,
        stratify=obs_df[label_col],
    )

    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    train_df = df[df[idx_col].isin(train_idx)]
    val_df = df[df[idx_col].isin(val_idx)]
    test_df = df[df[idx_col].isin(test_idx)]

    return (train_df, val_df, test_df)


def get_samples(paths, samples_params) -> pd.DataFrame:
    df = get_df_from_table(paths.db_path, "cv_photos")
    df["path"] = (
        paths.root
        + "/"
        + df[samples_params.label_col].astype(str)
        + "/"
        + df[samples_params.photo_id_col].astype(str)
        + ".jpg"
    )
    df[samples_params.label_col] = df[samples_params.label_col].map(LABEL_MAPPING)
    return df


def build_datasets(
    paths, samples_params, model_configs: dict
) -> tuple[PhenologyDataset, PhenologyDataset, PhenologyDataset]:
    df = get_samples(paths, samples_params)

    train_df, val_df, test_df = split_dataset(
        df, idx_col=samples_params.observations_col, label_col=samples_params.label_col
    )
    print(train_df)
    base_transforms = build_base_transforms(
        **{k: model_configs[k] for k in ("input_size", "mean", "std")}
    )
    train_transform, val_transform = build_transforms(base_transforms)

    train_set = PhenologyDataset(train_df, train_transform, samples_params)
    val_set = PhenologyDataset(val_df, val_transform, samples_params)
    test_set = PhenologyDataset(test_df, val_transform, samples_params)

    return train_set, val_set, test_set

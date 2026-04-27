import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def get_split_idx(root: str) -> tuple[list, list, list]:
    full_dataset = ImageFolder(
        root,
    )
    idx = np.arange(0, len(full_dataset.samples))
    labels = np.array([item[1] for item in full_dataset.samples])
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        idx, labels, test_size=0.40, random_state=42
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels, test_size=0.75, random_state=42
    )
    return (train_idx, val_idx, test_idx)


def build_datasets(root: str):
    train_idx, val_idx, test_idx = get_split_idx(root=root)

    train_dataset = ImageFolder(root, transform=train_transform)
    val_dataset = ImageFolder(root, transform=val_transform)

    # Then Subset each with the right indices
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(val_dataset, test_idx)  # same transforms as val

    return train_subset, val_subset, test_subset


if __name__ == "__main__":
    build_datasets("/home/etienne/projects/inat-phenology-cv/data/photos")

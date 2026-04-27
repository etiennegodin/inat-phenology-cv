import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

base_transforms = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

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


def get_split_idx(root: str) -> tuple[list, list, list]:
    full_dataset = ImageFolder(
        root,
    )
    idx = np.arange(0, len(full_dataset.samples))
    labels = np.array([item[1] for item in full_dataset.samples])
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        idx, labels, test_size=0.40, random_state=42, stratify=labels
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        temp_idx, temp_labels, test_size=0.75, random_state=42, stratify=temp_labels
    )
    return (train_idx, val_idx, test_idx)


def build_datasets(root: str) -> tuple[Subset, Subset, Subset]:
    train_idx, val_idx, test_idx = get_split_idx(root=root)

    train_dataset = ImageFolder(root, transform=train_transform)
    val_dataset = ImageFolder(root, transform=val_transform)

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    test_subset = Subset(val_dataset, test_idx)  # same transforms as val
    return train_subset, val_subset, test_subset


if __name__ == "__main__":
    build_datasets("/home/etienne/projects/inat-phenology-cv/data/photos")

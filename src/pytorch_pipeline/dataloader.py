import torch


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    images = []
    labels = []
    for images_tensor, label in batch:
        labels.append(label)
        images.append(images_tensor)
    return images, torch.stack(labels)

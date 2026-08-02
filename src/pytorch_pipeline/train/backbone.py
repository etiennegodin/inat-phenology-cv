from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import open_clip
import timm
import torch
from torch import Tensor, nn
from torchvision import transforms

if TYPE_CHECKING:
    pass


class Backbone(ABC):
    encoder: Any
    output_dim: int

    def __init__(self) -> None:
        # Dummy pass to get backbone output dimensions
        with torch.no_grad():
            dummy_input_shape = self.get_dummy_input_shape()
            dummy_input = torch.zeros(1, *dummy_input_shape)
            self.output_dim = self.encode(dummy_input).shape[-1]
        super().__init__()

    @abstractmethod
    def encode(self, input: Tensor) -> Tensor:
        """Encodes incoming tensor through the backbone"""
        pass

    @abstractmethod
    def get_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        """Returns backbone specific image transformations"""
        pass

    @abstractmethod
    def get_trainable_blocks(self) -> nn.Sequential:
        """Returns block from shallow to deeper"""
        pass

    @abstractmethod
    def get_dummy_input_shape(self) -> tuple[int, int, int]:
        """Get shape for dummy input to calculate output_dim"""
        pass


class EfficientNetBackbone(Backbone):
    def __init__(self, *args, **kwargs) -> None:
        self.encoder = timm.create_model(
            "efficientnet_b0", num_classes=0, pretrained=True
        )
        super().__init__()

    def encode(self, input: Tensor) -> Tensor:
        return self.encoder(input)

    def get_transforms(self):
        # Recreate images transforms for this backbone
        configs = self.encoder.default_cfg
        img_size = (configs["input_size"][1], configs["input_size"][2])
        base_transforms = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=configs["mean"], std=configs["std"]),
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
        return train_transform, val_transform

    def get_trainable_blocks(self) -> nn.Sequential:
        return self.encoder.blocks

    def get_dummy_input_shape(self) -> tuple[int, int, int]:
        return self.encoder.default_cfg["input_size"]


class BioClipBackbone(Backbone):
    def __init__(self) -> None:
        self.model, self.train_transform, self.val_transform = (
            open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        )
        self.encoder = self.model.visual  # image tower only
        super().__init__()

    def get_transforms(self) -> tuple[transforms.Compose, transforms.Compose]:
        return self.train_transform, self.val_transform


BACKBONE_REGISTRY = {"efficientnet": EfficientNetBackbone}

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import open_clip
import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode, v2

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from ..train.flow_control import StageUnfreezeState  #

logger = logging.getLogger(__name__)


class Backbone(nn.Module, ABC):
    encoder: nn.Module
    output_dim: int
    patch_dim: int = 0
    has_patch_support: bool = False
    trainable_block_count: int

    def __init__(self) -> None:
        super().__init__()

    def log_trainable_blocks(self):
        # Show which blocks i trainable
        for i, block in enumerate(self.get_trainable_blocks()):
            trainable = any(p.requires_grad for p in block.parameters())
            logger.debug(f"Block {i}: trainable = {trainable}")

    def set_trainable_block_count(self):
        self.trainable_block_count = len(self.get_trainable_blocks())

    def unfreeze_stage(
        self,
        stage_state: StageUnfreezeState,
        optimizer: Optimizer,
        lr: float = 1.0,
    ):
        logger.debug(f"Unfreezing stage {stage_state.name}")
        stage_params = []
        all_blocks = self.get_trainable_blocks()
        blocks = [all_blocks[i] for i in stage_state.blocks]

        for block in blocks:
            for p in block.parameters(recurse=True):
                p.requires_grad = True
                stage_params.append(p)

        optimizer.add_param_group(
            {
                "name": stage_state.name,
                "params": stage_params,
                "lr": lr,
            },
        )

        # Log
        self.log_trainable_blocks()

    def freeze(self):
        """Freeze all backbone parameters"""
        for p in self.encoder.parameters(recurse=True):
            p.requires_grad = False

    def set_output_dim(self):
        """Single forward pass with a dummy input to get expected output shape"""
        # Dummy pass to get backbone output dimensions
        with torch.no_grad():
            self.encoder.eval()
            dummy_input_shape = self.get_dummy_input_shape()
            dummy_input = torch.zeros(1, *dummy_input_shape)
            self.output_dim = self.encode(dummy_input).shape[-1]

    def _validate_n_blocks(self, n_blocks: int) -> None:
        """Validate that n_blocks is in [1, total_blocks]."""
        total = len(self.get_trainable_blocks())
        if not (1 <= n_blocks <= total):
            raise ValueError(f"n_blocks={n_blocks} out of range [1, {total}]")

    @abstractmethod
    def encode(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        """Return CLS embeddings ready for attention pooling.

        Normalization, if any, is handled per-backbone to match its pretraining
        regime.  Pass ``n_blocks`` to run only the first *n_blocks* transformer
        blocks instead of the full depth; ``None`` (default) keeps the unchanged
        full-depth path used during training.
        """
        pass

    def encode_patches(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        """Return patch embeddings [B, num_patches, patch_dim] if supported.

        Pass ``n_blocks`` to truncate at that block depth (same semantics as
        ``encode``).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support patch encoding."
        )

    @abstractmethod
    def get_transforms(self) -> tuple[v2.Compose, v2.Compose]:
        """Returns backbone specific image transformations"""
        pass

    @abstractmethod
    def get_trainable_blocks(self) -> list[nn.Module]:
        """Returns block from shallow to deeper"""
        pass

    @abstractmethod
    def get_dummy_input_shape(self) -> tuple[int, int, int]:
        """Get shape for dummy input to calculate output_dim"""
        pass


class EfficientNetBackbone(Backbone):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder: nn.Module = timm.create_model(
            "efficientnet_b0", num_classes=0, pretrained=True
        )
        self.freeze()
        self.set_output_dim()
        self.set_trainable_block_count()

    def encode(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        if n_blocks is not None:
            raise ValueError(
                "EfficientNetBackbone does not support n_blocks; "
                "CNN stages are not ViT transformer blocks."
            )
        return self.encoder(input)

    def get_transforms(self):
        # Recreate images transforms for this backbone
        configs = self.encoder.default_cfg
        img_size = (configs["input_size"][1], configs["input_size"][2])
        base_transforms = [
            v2.Resize(img_size),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32),
            v2.Normalize(mean=configs["mean"], std=configs["std"]),
        ]

        train_transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(15),
                v2.RandomPerspective(0.2),
                v2.RandomAutocontrast(),
            ]
            + base_transforms
        )

        val_transform = v2.Compose(base_transforms)
        return train_transform, val_transform

    def get_trainable_blocks(self):
        return self.encoder.blocks

    def get_dummy_input_shape(self) -> tuple[int, int, int]:
        return self.encoder.default_cfg["input_size"]


class BioClipBackbone(Backbone):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model, self.train_transform, self.val_transform = (
            open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        )
        self.train_transform: v2.Compose
        self.val_transform: v2.Compose
        self.encoder = model.visual  # image tower only
        self.freeze()
        self.set_output_dim()
        self.patch_dim = 768
        self.has_patch_support = True
        self.set_trainable_block_count()

    def encode(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        # Normalising clip embeddings for scale sensitive attention pooling
        if n_blocks is None:
            embeddings = self.encoder(input)
            return F.normalize(embeddings, dim=1)  # embeddings normalised per row (img)
        self._validate_n_blocks(n_blocks)
        tokens = self._visual_tokens(input, n_blocks)
        cls = self.encoder.ln_post(tokens[:, 0, :])
        if self.encoder.proj is not None:
            cls = cls @ self.encoder.proj
        return F.normalize(cls, dim=1)

    def _visual_tokens(self, x: Tensor, n_blocks: int | None) -> Tensor:
        """Run BioCLIP-2 patch embed + positional embed + up to n_blocks resblocks.

        Returns tokens in NLD layout (batch, seq_len, hidden).  Both CLS and
        patch encode paths share this stem so the two methods stay in sync.
        """
        x = self.encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # N, L, D
        cls_tok = self.encoder.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tok, x], dim=1)
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD → LND
        resblocks = self.encoder.transformer.resblocks
        blocks_to_run = resblocks if n_blocks is None else resblocks[:n_blocks]
        for blk in blocks_to_run:
            x = blk(x)
        return x.permute(1, 0, 2)  # LND → NLD

    def encode_patches(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        if n_blocks is not None:
            self._validate_n_blocks(n_blocks)
        tokens = self._visual_tokens(input, n_blocks)
        patches = tokens[:, 1:, :]
        if hasattr(self.encoder, "ln_post"):
            patches = self.encoder.ln_post(patches)

        return patches

    def get_transforms(self, version="v2") -> tuple[v2.Compose, v2.Compose]:
        if version != "v2":
            return self.train_transform, self.val_transform

        v2_train_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        v2_val_transforms = v2.Compose(
            [
                v2.Resize(
                    size=224,
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                v2.CenterCrop(size=(224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        return v2_train_transform, v2_val_transforms

    def get_trainable_blocks(self) -> nn.ModuleList:
        return self.encoder.transformer.resblocks

    def get_dummy_input_shape(self) -> tuple[int, int, int]:
        img_size: tuple = tuple(self.encoder.image_size)
        return 3, *img_size


class BioClip2Backbone(Backbone):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model, self.train_transform, self.val_transform = (
            open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        )
        self.train_transform: v2.Compose
        self.val_transform: v2.Compose
        self.encoder = model.visual  # image tower only
        self.freeze()
        self.set_output_dim()
        self.patch_dim = 768
        self.has_patch_support = True
        self.set_trainable_block_count()

    def encode(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        # Normalising clip embeddings for scale sensitive attention pooling
        if n_blocks is None:
            embeddings = self.encoder(input)
            return F.normalize(embeddings, dim=1)  # embeddings normalised per row (img)
        self._validate_n_blocks(n_blocks)
        tokens = self._visual_tokens(input, n_blocks)
        cls = self.encoder.ln_post(tokens[:, 0, :])
        if self.encoder.proj is not None:
            cls = cls @ self.encoder.proj
        return F.normalize(cls, dim=1)

    def _visual_tokens(self, x: Tensor, n_blocks: int | None) -> Tensor:
        """Run BioCLIP-2 patch embed + positional embed + up to n_blocks resblocks.

        Returns tokens in NLD layout (batch, seq_len, hidden).  Both CLS and
        patch encode paths share this stem so the two methods stay in sync.
        """
        x = self.encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # N, L, D
        cls_tok = self.encoder.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_tok, x], dim=1)
        x = x + self.encoder.positional_embedding.to(x.dtype)
        x = self.encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD → LND
        resblocks = self.encoder.transformer.resblocks
        blocks_to_run = resblocks if n_blocks is None else resblocks[:n_blocks]
        for blk in blocks_to_run:
            x = blk(x)
        return x.permute(1, 0, 2)  # LND → NLD

    def encode_patches(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        if n_blocks is not None:
            self._validate_n_blocks(n_blocks)
        tokens = self._visual_tokens(input, n_blocks)
        patches = tokens[:, 1:, :]

        if hasattr(self.encoder, "ln_post"):
            patches = self.encoder.ln_post(patches)

        return patches

    def get_transforms(self, version="v2") -> tuple[v2.Compose, v2.Compose]:
        if version != "v2":
            return self.train_transform, self.val_transform

        v2_train_transform = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        v2_val_transforms = v2.Compose(
            [
                v2.Resize(
                    size=224,
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias=True,
                ),
                v2.CenterCrop(size=(224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        return v2_train_transform, v2_val_transforms

    def get_trainable_blocks(self) -> nn.ModuleList:
        return self.encoder.transformer.resblocks

    def get_dummy_input_shape(self) -> tuple[int, int, int]:
        img_size: tuple = tuple(self.encoder.image_size)
        return 3, *img_size


class DinoV3Backbone(Backbone):
    def __init__(self, checkpoint_path: str | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import sys

        dinov3_dir = Path(__file__).resolve().parents[3] / "external" / "dinov3"
        if str(dinov3_dir) not in sys.path:
            sys.path.insert(0, str(dinov3_dir))

        from dinov3.hub.backbones import dinov3_vitb16

        self.encoder = dinov3_vitb16(pretrained=False)
        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            logger.info(f"Loading DINOv3 weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.encoder.load_state_dict(state_dict)
        elif checkpoint_path is not None:
            logger.warning(
                f"DINOv3 checkpoint not found at {checkpoint_path}, "
                "initializing random weights."
            )
        else:
            logger.info("No DINOv3 checkpoint configured; initializing random weights.")
        self.freeze()
        self.output_dim = 768
        self.patch_dim = 768
        self.has_patch_support = True
        self.set_trainable_block_count()

    def encode(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        if n_blocks is None:
            out = self.encoder.forward_features(input)
            return F.normalize(out["x_norm_clstoken"], dim=1)
        self._validate_n_blocks(n_blocks)
        # n=[k] returns the hidden state after block k (0-indexed), i.e. blocks 0..k ran
        out = self.encoder.get_intermediate_layers(
            input, n=[n_blocks - 1], norm=True, return_class_token=True
        )
        # out is a list of (patch_tokens, cls_token) tuples, one per requested layer
        cls = out[0][1]
        return F.normalize(cls, dim=1)

    def encode_patches(self, input: Tensor, n_blocks: int | None = None) -> Tensor:
        if n_blocks is None:
            out = self.encoder.forward_features(input)
            return out["x_norm_patchtokens"]
        self._validate_n_blocks(n_blocks)
        out = self.encoder.get_intermediate_layers(
            input, n=[n_blocks - 1], norm=True, return_class_token=True
        )
        return out[0][0]

    def get_transforms(self) -> tuple[v2.Compose, v2.Compose]:
        base_transforms = [
            v2.Resize(
                (224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        train_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        val_transforms = v2.Compose(base_transforms)
        return train_transforms, val_transforms

    def get_trainable_blocks(self) -> nn.ModuleList:
        return self.encoder.blocks

    def get_dummy_input_shape(self) -> tuple[int, int, int]:
        return (3, 224, 224)


BACKBONE_REGISTRY = {
    "efficientnet": EfficientNetBackbone,
    "bioclip": BioClipBackbone,
    "bioclip2": BioClip2Backbone,
    "dinov3": DinoV3Backbone,
    "dinov3_vitb16": DinoV3Backbone,
}

LOCAL_CHECKPOINT_BACKBONES = frozenset({"dinov3", "dinov3_vitb16"})


def build_backbone(
    backbone_name: str,
    dinov3_checkpoint_path: str | None = None,
    device: torch.device | None = None,
) -> Backbone:
    backbone_cls = BACKBONE_REGISTRY[backbone_name]

    if backbone_name in LOCAL_CHECKPOINT_BACKBONES:
        bkb = backbone_cls(checkpoint_path=dinov3_checkpoint_path)
    else:
        bkb = backbone_cls()

    if device is not None:
        return bkb.to(device)
    return bkb

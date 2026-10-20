from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from ..config import CLASS_ORDER
from ..utils.misc import unfreeze
from .backbone import build_backbone

if TYPE_CHECKING:
    from torch import nn

    from ..config.params import ModelParams

logger = logging.getLogger(__name__)


class BaseAttentionPooling(nn.Module):
    def __init__(
        self, in_features: int, neurons: int, dropout_rate: float, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.V = nn.Linear(in_features, neurons)
        self.U = nn.Linear(in_features, neurons)
        self.w = nn.Linear(neurons, 1)
        self.dim = 0  # collapses on axis 0 (images)
        self.d = nn.Dropout(dropout_rate)

    def _compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.V(x))
        return self.w(v)

    def forward(self, x: torch.Tensor):
        scores = self._compute_scores(x)
        weights = F.softmax(scores, dim=self.dim)
        weights = self.d(weights)
        pooled = (weights * x).sum(dim=self.dim)
        return pooled, weights


class AttentionPooling(BaseAttentionPooling):
    """Simple (ungated) attention pooling: w^T tanh(V x)"""

    pass


class GatedAttentionPooling(BaseAttentionPooling):
    """Gated attention pooling: w^T (tanh(V x) * sigm(U x))"""

    def _compute_scores(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.V(x))
        u = torch.sigmoid(self.U(x))
        return self.w(v * u)


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        neurons: int,
        outputs: int,
        dropout_rate: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.V = nn.Linear(in_features, neurons)
        self.r = nn.ReLU()
        self.d = nn.Dropout(dropout_rate)
        self.w = nn.Linear(neurons, outputs)

    def forward(self, x):
        w1 = self.r(self.V(x))
        w2 = self.d(w1)
        return self.w(w2)


class AttentionBranch(nn.Module):
    def __init__(
        self, input_dim: int, params: ModelParams, named_class: str, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.named_class = named_class
        attn_cls = GatedAttentionPooling if params.gated else AttentionPooling
        self.attention = attn_cls(
            in_features=input_dim,
            neurons=params.attention_neurons,
            dropout_rate=params.attention_dropout_prob,
        )
        self.head = ClassifierHead(
            input_dim,
            params.head_neurons,
            params.head_outputs,
            params.head_dropout_prob,
        )

    def forward(self, observations):
        obs_pools = []
        obs_weights: list[torch.Tensor] = []
        for obs in observations:
            pool, weights = self.attention(obs)
            obs_pools.append(pool)
            obs_weights.append(weights)

        # Stack back in one Tensor
        pooled = torch.stack(obs_pools)

        predictions = self.head(pooled).squeeze(1)
        return predictions, obs_weights


class MLPProjectionNeck(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dropout_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        d_ff = dim_feedforward or (d_model * 4)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.cross_attn(
            query, key, value, key_padding_mask=key_padding_mask
        )
        x = self.norm1(query + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        out = self.norm2(x + self.dropout2(ffn_out))
        return out, attn_weights


class MilAttentionModel(nn.Module):
    def __init__(self, params: ModelParams, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params = params
        self.backbone = build_backbone(params.backbone, params.checkpoint_path)
        self.branches = nn.ModuleList(
            [AttentionBranch(self.backbone.output_dim, params, c) for c in CLASS_ORDER]
        )

    def forward(self, x):
        """Forward pass for MIL Attention Pooling.

        Args:
            x: list of image tensors per observation [Tensor(N_i, C, H, W)]

        Returns:
            tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
        """
        indices = []
        class_predictions = []
        class_attention_weights: dict[str, list[torch.Tensor]] = {}

        # Get indices for split
        for i in x:
            indices.append(i.size()[0])

        # Concatenate all images in one Tensor
        stacked = torch.cat(x)

        # Run backbone on stacked Tensor
        embeddings = self.backbone.encode(stacked)

        # Split embedding per observation
        observations = torch.split(embeddings, indices)

        # Iterate attention branches over each observation
        for b in self.branches:
            predictions, attention_weights = b(observations)
            class_predictions.append(predictions)
            class_attention_weights[b.named_class] = attention_weights

        # Stack back batch in shape (batch, n_classes)
        predictions = torch.stack(class_predictions, dim=1)

        return predictions, class_attention_weights

    def get_attention_params(self):
        return itertools.chain.from_iterable(
            branch.attention.parameters() for branch in self.branches
        )

    def get_head_params(self):
        return itertools.chain.from_iterable(
            branch.head.parameters() for branch in self.branches
        )


# Backward-compatible alias
PhenologyModel = MilAttentionModel


class CrossAttentionModel(nn.Module):
    def __init__(self, params: ModelParams, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params = params
        self.backbone = build_backbone(params.backbone, params.checkpoint_path)

        if not self.backbone.has_patch_support:
            raise ValueError(
                f"Backbone '{params.backbone}' "
                "does not support patch encoding required"
                " by cross_attention architecture."
            )

        self.projection_neck = MLPProjectionNeck(
            in_features=self.backbone.patch_dim,
            out_features=params.proj_dim,
            dropout_rate=params.cross_attn_dropout,
        )
        self.image_index_embed = nn.Embedding(
            params.max_images_per_obs, params.proj_dim
        )
        scale = params.proj_dim**-0.5
        self.class_queries = nn.Parameter(
            torch.randn(len(CLASS_ORDER), params.proj_dim) * scale
        )
        self.cross_attention = CrossAttentionBlock(
            d_model=params.proj_dim,
            nhead=params.cross_attn_heads,
            dropout=params.cross_attn_dropout,
        )
        self.heads = nn.ModuleList([nn.Linear(params.proj_dim, 1) for _ in CLASS_ORDER])

    def forward(self, x: list[torch.Tensor]):
        """Runs forward pass through patch backbone, projection neck,
        1D image index embedding, cross-attention with class queries, and linear heads.

        Args:
            x: list of image tensors per observation [Tensor(N_i, C, H, W)]

        Returns:
            tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
                predictions: Tensor of shape (batch, num_classes)
                class_attention_weights: dict mapping each class name
                to attention weights per observation
        """
        device = x[0].device
        obs_image_counts = [obs.size(0) for obs in x]
        batch_size = len(x)

        # Concatenate all images in one tensor
        stacked = torch.cat(x, dim=0)

        # Extract patch tokens: shape [total_images, num_patches, patch_dim]
        patch_tokens = self.backbone.encode_patches(stacked)
        num_patches = patch_tokens.size(1)

        # Project patch tokens: shape [total_images, num_patches, proj_dim]
        projected = self.projection_neck(patch_tokens)

        # Split back per observation
        obs_projected = torch.split(projected, obs_image_counts)

        # Add 1D image index embedding and flatten tokens per observation
        obs_tokens_list = []
        token_counts = []
        for obs_patches in obs_projected:
            num_imgs = obs_patches.size(0)
            clamped_idx = torch.clamp(
                torch.arange(num_imgs, device=device),
                max=self.params.max_images_per_obs - 1,
            )
            idx_embed = self.image_index_embed(clamped_idx).unsqueeze(1)
            obs_patches = obs_patches + idx_embed
            obs_tokens = obs_patches.reshape(
                num_imgs * num_patches, self.params.proj_dim
            )
            obs_tokens_list.append(obs_tokens)
            token_counts.append(obs_tokens.size(0))

        # Pad observations to max_tokens for parallel cross-attention
        max_tokens = max(token_counts)
        padded_keys = torch.zeros(
            batch_size,
            max_tokens,
            self.params.proj_dim,
            dtype=projected.dtype,
            device=device,
        )
        key_padding_mask = torch.ones(
            batch_size, max_tokens, dtype=torch.bool, device=device
        )

        for i, tokens in enumerate(obs_tokens_list):
            t_len = tokens.size(0)
            padded_keys[i, :t_len] = tokens
            key_padding_mask[i, :t_len] = False

        # Expand class queries to batch: [batch_size, num_classes, proj_dim]
        queries = self.class_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention pass
        query_out, attn_weights = self.cross_attention(
            query=queries,
            key=padded_keys,
            value=padded_keys,
            key_padding_mask=key_padding_mask,
        )

        # Classifier heads
        logits = [self.heads[c](query_out[:, c, :]) for c in range(len(CLASS_ORDER))]
        predictions = torch.cat(logits, dim=1)

        # Extract per-class attention weights unpadded
        class_attention_weights: dict[str, list[torch.Tensor]] = {}
        for c, class_name in enumerate(CLASS_ORDER):
            class_attention_weights[class_name] = [
                attn_weights[i, c, : token_counts[i]] for i in range(batch_size)
            ]

        return predictions, class_attention_weights

    def get_attention_params(self):
        return itertools.chain(
            self.projection_neck.parameters(),
            self.image_index_embed.parameters(),
            [self.class_queries],
            self.cross_attention.parameters(),
        )

    def get_head_params(self):
        return self.heads.parameters()


def build_pipeline_model(device: torch.device, model_params: ModelParams) -> nn.Module:
    """Instantiate model and unfreezes backbone last params.

    Dispatches to MilAttentionModel (formerly PhenologyModel) or
    CrossAttentionModel based on model_params.architecture.

    Returns:
        nn.Module: instantiated model
    """
    if model_params.architecture == "cross_attention":
        model: nn.Module = CrossAttentionModel(model_params)
    elif model_params.architecture == "mil_pooling":
        model = MilAttentionModel(model_params)
    else:
        raise ValueError(
            f"Unknown architecture '{model_params.architecture}'",
            " Must be 'mil_pooling' or 'cross_attention'.",
        )

    blocks = model.backbone.get_trainable_blocks()

    # Unfreeze last block
    if model_params.start_unfreezed > 0:
        logger.debug("Unfreezing backbone parameters")
        for block in blocks[-model_params.start_unfreezed :]:
            unfreeze(block)

    model.backbone.log_trainable_blocks()

    model.to(device)
    return model

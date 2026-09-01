from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.configs import CLASS_ORDER
from .backbone import BACKBONE_REGISTRY, Backbone

if TYPE_CHECKING:
    from torch import nn

    from ..utils.params import ModelParams


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


class PhenologyModel(nn.Module):
    def __init__(self, params: ModelParams, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone: Backbone = BACKBONE_REGISTRY[params.backbone]()
        self.branches = nn.ModuleList(
            [AttentionBranch(self.backbone.output_dim, params, c) for c in CLASS_ORDER]
        )

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            tuple[torch.Tensor, list[list[torch.Tensor]] ]:
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

        # Stack back batch in shape (batch,n_classes)
        predictions = torch.stack(class_predictions, dim=1)

        return predictions, class_attention_weights

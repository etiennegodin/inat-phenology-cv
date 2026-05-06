from typing import TYPE_CHECKING

import timm
import torch
import torch.nn.functional as F
from torch import nn

from ..utils.registry import EFFICIENT_NET_LAST_BLOCK

if TYPE_CHECKING:
    from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, in_features: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.V = nn.Linear(in_features, 128)
        self.w = nn.Linear(128, 1)

    def forward(self, chunk):
        # chunk: (n_images, in_features)
        scores = self.w(torch.tanh(self.V(chunk)))
        weights = F.softmax(scores, dim=0)
        pooled = (weights * chunk).sum(dim=0)
        return pooled, weights


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.V = nn.Linear(in_features, 256)
        self.r = nn.ReLU()
        self.d = nn.Dropout(0.5)
        self.w = nn.Linear(256, 1)

    def forward(self, x):
        w1 = self.r(self.V(x))
        w2 = self.d(w1)
        return self.w(w2)


class PhenologyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = get_backbone()
        self.attention = AttentionPooling(self.backbone.num_features)
        self.head = ClassifierHead(self.backbone.num_features)

    def forward(self, x):
        indices = []
        obs_pools = []
        obs_weights = []

        # Get indices for split
        for i in x:
            indices.append(i.size()[0])

        # Concatenate all images in one Tensor
        stacked = torch.cat(x)

        # Run backbone on stacked Tensor
        embeddings = self.backbone(stacked)

        # Split embedding per observation
        observations = torch.split(embeddings, indices)

        # Iterate attention pool over each observation
        for obs in observations:
            pool, weights = self.attention(obs)
            obs_pools.append(pool)
            obs_weights.append(weights)

        # Stack back in one Tensor
        pooled = torch.stack(obs_pools)

        predictions = self.head(pooled).squeeze(1)
        return predictions, obs_weights


def get_backbone():
    return timm.create_model("efficientnet_b0", num_classes=0, pretrained=True)


def build_head(num_features: int):
    return nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
    )


def build_model(head_inputs: int = 256, dropout_prob: float = 0.5) -> nn.Sequential:
    backbone = get_backbone()

    for p in backbone.parameters():
        p.requires_grad = False

    # Unfreeze last block
    for name, p in backbone.named_parameters():
        if name.startswith(tuple(EFFICIENT_NET_LAST_BLOCK)):
            p.requires_grad = True

    head = build_head(backbone.num_features)
    model = nn.Sequential(backbone, head)
    return model


if __name__ == "__main__":
    model = build_model()
    for name, p in model.named_parameters():
        print(name, p.requires_grad)

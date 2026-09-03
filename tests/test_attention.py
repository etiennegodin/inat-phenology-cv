import torch

from pytorch_pipeline.train.factory import build_pipeline_model
from pytorch_pipeline.train.model import (
    AttentionPooling,
    BaseAttentionPooling,
    GatedAttentionPooling,
)
from pytorch_pipeline.utils.params import ModelParams


def test_simple_attention_pooling():
    in_features = 64
    neurons = 32
    dropout_rate = 0.25

    attn = AttentionPooling(
        in_features=in_features, neurons=neurons, dropout_rate=dropout_rate
    )

    assert isinstance(attn, BaseAttentionPooling)
    assert attn.d.p == dropout_rate

    attn.eval()
    x = torch.randn(5, in_features)
    pooled, weights = attn(x)

    assert pooled.shape == (in_features,)
    assert weights.shape == (5, 1)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_gated_attention_pooling():
    in_features = 64
    neurons = 32
    dropout_rate = 0.15

    attn = GatedAttentionPooling(
        in_features=in_features, neurons=neurons, dropout_rate=dropout_rate
    )

    assert isinstance(attn, BaseAttentionPooling)
    assert hasattr(attn, "U")
    assert attn.d.p == dropout_rate

    attn.eval()
    x = torch.randn(5, in_features)
    pooled, weights = attn(x)

    assert pooled.shape == (in_features,)
    assert weights.shape == (5, 1)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_model_gated_attention_selection():
    device = torch.device("cpu")

    # Test gated=True
    params_gated = ModelParams(
        backbone="efficientnet",
        head_neurons=32,
        head_outputs=1,
        head_dropout_prob=0.1,
        attention_neurons=16,
        attention_dropout_prob=0.2,
        last_blocks=0,
        gated=True,
    )
    model_gated = build_pipeline_model(device, params_gated)
    for branch in model_gated.branches:
        assert isinstance(branch.attention, GatedAttentionPooling)
        assert branch.attention.d.p == 0.2

    # Test gated=False (simple attention)
    params_simple = ModelParams(
        backbone="efficientnet",
        head_neurons=32,
        head_outputs=1,
        head_dropout_prob=0.1,
        attention_neurons=16,
        attention_dropout_prob=0.35,
        last_blocks=0,
        gated=False,
    )
    model_simple = build_pipeline_model(device, params_simple)
    for branch in model_simple.branches:
        assert isinstance(branch.attention, AttentionPooling)
        assert not isinstance(branch.attention, GatedAttentionPooling)
        assert branch.attention.d.p == 0.35

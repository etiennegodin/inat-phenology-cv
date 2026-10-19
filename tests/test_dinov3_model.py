import torch

from plant_pheno.config import CLASS_ORDER, ModelParams, OptimizerParams
from plant_pheno.core import (
    BACKBONE_REGISTRY,
    CrossAttentionModel,
    DinoV3Backbone,
    MilAttentionModel,
    build_pipeline_model,
)
from plant_pheno.train.factory import build_pipeline_optimizer


def test_dinov3_backbone_shapes():
    backbone = DinoV3Backbone()
    assert backbone.has_patch_support
    assert backbone.output_dim == 768
    assert backbone.patch_dim == 768

    x = torch.randn(2, 3, 224, 224)
    cls_emb = backbone.encode(x)
    assert cls_emb.shape == (2, 768)

    patches = backbone.encode_patches(x)
    assert patches.shape == (2, 196, 768)

    train_t, val_t = backbone.get_transforms()
    assert train_t is not None
    assert val_t is not None


def test_bioclip_backbone_patch_encoding():
    backbone = BACKBONE_REGISTRY["bioclip"]()
    assert backbone.has_patch_support
    assert backbone.patch_dim == 768

    x = torch.randn(2, 3, 224, 224)
    patches = backbone.encode_patches(x)
    assert patches.shape == (2, 196, 768)


def test_cross_attention_model_forward():
    device = torch.device("cpu")
    params = ModelParams(
        backbone="dinov3",
        architecture="cross_attention",
        proj_dim=128,
        cross_attn_heads=4,
        cross_attn_dropout=0.1,
        max_images_per_obs=32,
        start_unfreezed=0,
    )
    model = build_pipeline_model(device, params)
    assert isinstance(model, CrossAttentionModel)

    # 3 observations with variable image counts: 2, 4, 1
    obs1 = torch.randn(2, 3, 224, 224)
    obs2 = torch.randn(4, 3, 224, 224)
    obs3 = torch.randn(1, 3, 224, 224)
    x = [obs1, obs2, obs3]

    model.eval()
    with torch.no_grad():
        preds, attn_weights = model(x)

    assert preds.shape == (3, len(CLASS_ORDER))
    assert set(attn_weights.keys()) == set(CLASS_ORDER)

    for c in CLASS_ORDER:
        weights_list = attn_weights[c]
        assert len(weights_list) == 3
        # Tokens per obs = num_images * 196
        assert weights_list[0].shape == (2 * 196,)
        assert weights_list[1].shape == (4 * 196,)
        assert weights_list[2].shape == (1 * 196,)


def test_cross_attention_model_with_bioclip():
    device = torch.device("cpu")
    params = ModelParams(
        backbone="bioclip",
        architecture="cross_attention",
        proj_dim=128,
        cross_attn_heads=4,
        start_unfreezed=0,
    )
    model = build_pipeline_model(device, params)
    assert isinstance(model, CrossAttentionModel)

    obs1 = torch.randn(1, 3, 224, 224)
    obs2 = torch.randn(2, 3, 224, 224)
    x = [obs1, obs2]

    model.eval()
    with torch.no_grad():
        preds, attn_weights = model(x)

    assert preds.shape == (2, len(CLASS_ORDER))


def test_mil_attention_model_with_dinov3():
    device = torch.device("cpu")
    params = ModelParams(
        backbone="dinov3",
        architecture="mil_pooling",
        start_unfreezed=0,
    )
    model = build_pipeline_model(device, params)
    assert isinstance(model, MilAttentionModel)

    obs1 = torch.randn(2, 3, 224, 224)
    obs2 = torch.randn(3, 3, 224, 224)
    x = [obs1, obs2]

    model.eval()
    with torch.no_grad():
        preds, attn_weights = model(x)

    assert preds.shape == (2, len(CLASS_ORDER))
    assert set(attn_weights.keys()) == set(CLASS_ORDER)


def test_optimizer_parameter_groups():
    device = torch.device("cpu")
    optim_params = OptimizerParams(base_lr=1e-3)

    # Test CrossAttentionModel param groups
    cross_params = ModelParams(
        backbone="dinov3",
        architecture="cross_attention",
        start_unfreezed=0,
    )
    cross_model = build_pipeline_model(device, cross_params)
    cross_optimizer = build_pipeline_optimizer(cross_model, optim_params)

    group_names = [g["name"] for g in cross_optimizer.param_groups]
    assert group_names == ["backbone", "attention", "classifier_head"]
    assert len(cross_optimizer.param_groups[1]["params"]) > 0  # attention group
    assert len(cross_optimizer.param_groups[2]["params"]) > 0  # classifier_head group

    # Test MilAttentionModel param groups
    mil_params = ModelParams(
        backbone="dinov3",
        architecture="mil_pooling",
        start_unfreezed=0,
    )
    mil_model = build_pipeline_model(device, mil_params)
    mil_optimizer = build_pipeline_optimizer(mil_model, optim_params)

    mil_group_names = [g["name"] for g in mil_optimizer.param_groups]
    assert mil_group_names == ["backbone", "attention", "classifier_head"]
    assert len(mil_optimizer.param_groups[1]["params"]) > 0
    assert len(mil_optimizer.param_groups[2]["params"]) > 0

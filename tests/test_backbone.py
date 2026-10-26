import pytest
import torch

from plant_pheno.core.backbone import EfficientNetBackbone
from plant_pheno.train.flow_control import StageUnfreezeState


def test_unfreeze_stage_registers_parameters():
    backbone = EfficientNetBackbone()
    # Freeze all initially
    backbone.freeze()

    # Simulate initial unfreezing of last block
    last_block = backbone.get_trainable_blocks()[-1]
    for p in last_block.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {
                "name": "backbone",
                "params": [p for p in backbone.parameters() if p.requires_grad],
                "lr": 1e-4,
            }
        ]
    )
    assert len(optimizer.param_groups) == 1

    # Unfreeze previous block (index 5)
    stage_state = StageUnfreezeState(
        name="unfreezed_stage_0", local_warmup_len=3, blocks=[5]
    )
    backbone.unfreeze_stage(stage_state, optimizer=optimizer)

    assert len(optimizer.param_groups) == 2
    new_group = optimizer.param_groups[1]
    assert new_group["name"] == "unfreezed_stage_0"
    assert len(new_group["params"]) > 0
    assert all(
        isinstance(p, torch.Tensor) and p.requires_grad for p in new_group["params"]
    )


# ---------------------------------------------------------------------------
# n_blocks / truncated-encoding tests
# ---------------------------------------------------------------------------


def test_efficientnet_n_blocks_raises():
    """EfficientNet must raise clearly when n_blocks is requested."""
    backbone = EfficientNetBackbone()
    x = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError, match="n_blocks"):
        backbone.encode(x, n_blocks=1)


def test_dinov3_truncated_cls_shape():
    """Truncated CLS embedding has the same shape as full-depth encode."""
    from plant_pheno.core.backbone import DinoV3Backbone

    backbone = DinoV3Backbone()
    x = torch.randn(2, 3, 224, 224)
    full = backbone.encode(x)
    k = backbone.trainable_block_count // 2
    truncated = backbone.encode(x, n_blocks=k)
    assert truncated.shape == full.shape  # [2, 768]


def test_dinov3_truncated_patches_shape():
    """Truncated patch embedding has the same spatial shape as full-depth."""
    from plant_pheno.core.backbone import DinoV3Backbone

    backbone = DinoV3Backbone()
    x = torch.randn(2, 3, 224, 224)
    full_patches = backbone.encode_patches(x)
    k = backbone.trainable_block_count // 2
    trunc_patches = backbone.encode_patches(x, n_blocks=k)
    assert trunc_patches.shape == full_patches.shape  # [2, 196, 768]


def test_dinov3_full_depth_equivalence():
    """n_blocks=total matches the default full-depth forward pass."""
    from plant_pheno.core.backbone import DinoV3Backbone

    backbone = DinoV3Backbone()
    x = torch.randn(1, 3, 224, 224)
    full = backbone.encode(x)
    total = backbone.trainable_block_count
    via_n = backbone.encode(x, n_blocks=total)
    assert torch.allclose(full, via_n, atol=1e-5)


def test_dinov3_invalid_n_blocks_raises():
    """n_blocks out of [1, total] raises ValueError."""
    from plant_pheno.core.backbone import DinoV3Backbone

    backbone = DinoV3Backbone()
    x = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError):
        backbone.encode(x, n_blocks=0)
    with pytest.raises(ValueError):
        backbone.encode(x, n_blocks=999)


def test_dinov3_variable_bag_cat_split_cls():
    """cat two bags → truncated encode → split restores per-obs shapes."""
    from plant_pheno.core.backbone import DinoV3Backbone

    backbone = DinoV3Backbone()
    bags = [torch.randn(2, 3, 224, 224), torch.randn(3, 3, 224, 224)]
    stacked = torch.cat(bags)
    k = backbone.trainable_block_count // 2
    embeddings = backbone.encode(stacked, n_blocks=k)
    bag_sizes = [b.size(0) for b in bags]
    split = torch.split(embeddings, bag_sizes)
    assert split[0].shape == (2, 768)
    assert split[1].shape == (3, 768)


def test_bioclip_truncated_cls_shape():
    """BioCLIP truncated CLS has same projected dimension as full encode."""
    from plant_pheno.core.backbone import BACKBONE_REGISTRY

    backbone = BACKBONE_REGISTRY["bioclip"]()
    x = torch.randn(2, 3, 224, 224)
    full = backbone.encode(x)
    truncated = backbone.encode(x, n_blocks=6)
    assert truncated.shape == full.shape


def test_bioclip_truncated_patches_shape():
    """BioCLIP truncated patches have same spatial shape [B, 196, 768]."""
    from plant_pheno.core.backbone import BACKBONE_REGISTRY

    backbone = BACKBONE_REGISTRY["bioclip"]()
    x = torch.randn(2, 3, 224, 224)
    patches = backbone.encode_patches(x, n_blocks=6)
    assert patches.shape == (2, 196, 768)

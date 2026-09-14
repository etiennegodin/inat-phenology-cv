import torch

from pytorch_pipeline.train.backbone import EfficientNetBackbone
from pytorch_pipeline.train.flow_control import StageUnfreezeState


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

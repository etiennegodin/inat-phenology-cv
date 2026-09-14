import pytest
import torch

from pytorch_pipeline.train.factory import (
    build_pipeline_model,
    build_pipeline_optimizer,
    build_scheduler,
)
from pytorch_pipeline.train.flow_control import (
    ClassesObjectiveState,
    StageUnfreezeState,
    TrainingState,
    create_stage_states,
)
from pytorch_pipeline.utils.params import (
    ModelParams,
    OptimizerParams,
    SchedulerParams,
    TrainingParams,
)


def test_warmup_lr_linear_interpolation():
    """Verify StageUnfreezeState.get_warmup_lr
    handles linear interpolation correctly."""
    stage = StageUnfreezeState(
        name="test_stage",
        local_warmup_len=3,
        blocks=[5],
        unlocked=True,
        unlocked_epoch=2,
    )

    # At unlock epoch (x = 0): should be start_factor (0.1)
    lr_0 = stage.get_warmup_lr(epoch=2, start_factor=0.1, end_factor=1.0)
    assert pytest.approx(lr_0, abs=1e-5) == 0.1

    # Middle of warmup (x = 1): should be midpoint (0.55)
    lr_1 = stage.get_warmup_lr(epoch=3, start_factor=0.1, end_factor=1.0)
    assert pytest.approx(lr_1, abs=1e-5) == 0.55

    # End of warmup (x = 2): should reach end_factor (1.0)
    lr_2 = stage.get_warmup_lr(epoch=4, start_factor=0.1, end_factor=1.0)
    assert pytest.approx(lr_2, abs=1e-5) == 1.0

    # Past warmup: capped at end_factor (1.0)
    lr_3 = stage.get_warmup_lr(epoch=5, start_factor=0.1, end_factor=1.0)
    assert pytest.approx(lr_3, abs=1e-5) == 1.0

    # Edge case: local_warmup_len <= 1 should return end_factor immediately
    instant_stage = StageUnfreezeState(
        name="instant_stage",
        local_warmup_len=1,
        blocks=[5],
        unlocked=True,
        unlocked_epoch=2,
    )
    assert instant_stage.get_warmup_lr(epoch=2) == 1.0


def test_unfreeze_pathway_multiepoch_simulation():
    """Simulate a multi-epoch training lifecycle to test the whole unfreezing pathway:
    - Staleness detection triggering first unfreeze
    - Parameter group registration & block grad enablement
    - Cooldown enforcement preventing premature next unfreeze
    - Warmup learning rate scaling per stage
    - Subsequent unfreeze after cooldown expires
    - Max stages condition halting further unfreezing
    """
    device = torch.device("cpu")
    training_params = TrainingParams(
        epochs=10,
        stopping_patience=10,
        unfreezing_patience=2,
        unfreezing_cooldown=3,
        starting_block=1,  # block 6 initially unfrozen
        max_stages=2,
        block_per_stage=1,
        start_epoch=None,
        best_objective=0.0,
        backbone_decay=0.9,
        unfreeze=True,
    )

    model_params = ModelParams(backbone="efficientnet", start_unfreezed=1)
    model = build_pipeline_model(device, model_params)
    optim_params = OptimizerParams(base_lr=1e-3)
    optimizer = build_pipeline_optimizer(model, optim_params)
    scheduler = build_scheduler(optimizer, SchedulerParams())

    training_state = TrainingState(
        classes_states=ClassesObjectiveState(class_count=3),
        training_params=training_params,
    )
    training_state.stages_states = create_stage_states(
        training_params=training_params,
        trainable_block_count=len(model.backbone.get_trainable_blocks())
        - training_params.starting_block,
    )

    assert len(training_state.stages_states) == 2
    assert training_state.stages_states[0].blocks == [5]
    assert training_state.stages_states[1].blocks == [4]
    assert (
        len(optimizer.param_groups) == 3
    )  # backbone (block 6), attention, classifier_head

    # Sequence of simulated metric values per epoch (flowers, fruiting, budding):
    # Epoch 0: All improve -> staleness [0, 0, 0]
    # Epoch 1: 2 classes stale (staleness = 1) -> below patience 2
    # Epoch 2: 2 classes reach patience 2 -> triggers first unfreeze (Stage 0)
    # Epoch 3: Stale again, but in cooldown -> Stage 1 does not unfreeze
    # Epoch 4: In cooldown -> Stage 1 does not unfreeze
    # Epoch 5: Cooldown done -> triggers second unfreeze (Stage 1)
    # Epoch 6: Stale again, but max_stages (2) reached -> no more unfreezing
    simulated_metrics = [
        [0.50, 0.50, 0.50],  # epoch 0
        [0.50, 0.10, 0.10],  # epoch 1
        [0.50, 0.10, 0.10],  # epoch 2 -> unlocks Stage 0 (block 5)
        [0.50, 0.10, 0.10],  # epoch 3 -> cooldown
        [0.50, 0.10, 0.10],  # epoch 4 -> cooldown
        [0.50, 0.10, 0.10],  # epoch 5 -> unlocks Stage 1 (block 4)
        [0.50, 0.10, 0.10],  # epoch 6 -> max stages hit
    ]

    for epoch in range(len(simulated_metrics)):
        metrics = simulated_metrics[epoch]
        training_state.patience_counter(metrics)

        # Mirror training workflow unfreeze block
        if training_state.unfreeze_condition():
            if training_state.max_stages_condition():
                pass
            elif not training_state.stages_states[0].unlocked:
                training_state.classes_states.reset()
                stage = training_state.stages_states[0]
                stage.unlocked = True
                stage.unlocked_epoch = epoch
                model.backbone.unfreeze_stage(stage, optimizer=optimizer)
            else:
                latest_stage = training_state.get_latest_stage_state()
                if not training_state.cooldown_condition(latest_stage, epoch=epoch):
                    training_state.classes_states.reset()
                    new_stage = training_state.stages_states[
                        training_state.unlocked_stage_count
                    ]
                    new_stage.unlocked_epoch = epoch
                    new_stage.unlocked = True
                    model.backbone.unfreeze_stage(new_stage, optimizer=optimizer)

        # Mirror learning rate assignment for unlocked stages
        for i, stage in enumerate(training_state.stages_states):
            if stage.unlocked:
                target_lr = scheduler.get_last_lr()[
                    0
                ] * training_params.get_depth_ratio(i)
                if training_state.cooldown_condition(stage, epoch=epoch):
                    lr = target_lr * stage.get_warmup_lr(epoch)
                else:
                    lr = target_lr
                for group in optimizer.param_groups:
                    if group.get("name") == stage.name:
                        group["lr"] = lr
                        break

        # Epoch-by-epoch assertions
        if epoch == 0:
            assert training_state.unlocked_stage_count == 0
            assert len(optimizer.param_groups) == 3

        elif epoch == 1:
            assert training_state.unlocked_stage_count == 0
            assert len(optimizer.param_groups) == 3

        elif epoch == 2:
            # Stage 0 unlocked
            assert training_state.unlocked_stage_count == 1
            assert training_state.stages_states[0].unlocked
            assert training_state.stages_states[0].unlocked_epoch == 2
            assert len(optimizer.param_groups) == 4
            assert optimizer.param_groups[-1]["name"] == "unfreezed_stage_0"
            # Block 5 parameters must have requires_grad == True
            block5 = model.backbone.get_trainable_blocks()[5]
            assert all(p.requires_grad for p in block5.parameters())
            # Warmup LR at epoch 2 is 10% of target_lr
            target_lr_0 = scheduler.get_last_lr()[0] * (0.9**0)
            stage0_group = next(
                g
                for g in optimizer.param_groups
                if g.get("name") == "unfreezed_stage_0"
            )
            assert pytest.approx(stage0_group["lr"], rel=1e-3) == 0.1 * target_lr_0

        elif epoch == 3:
            # Still in cooldown, stage 1 must not be unlocked
            assert training_state.unlocked_stage_count == 1
            assert not training_state.stages_states[1].unlocked
            stage0_group = next(
                g
                for g in optimizer.param_groups
                if g.get("name") == "unfreezed_stage_0"
            )
            target_lr_0 = scheduler.get_last_lr()[0] * (0.9**0)
            assert pytest.approx(stage0_group["lr"], rel=1e-3) == 0.55 * target_lr_0

        elif epoch == 4:
            # Cooldown's last epoch, stage 0 reaches full target_lr
            assert training_state.unlocked_stage_count == 1
            stage0_group = next(
                g
                for g in optimizer.param_groups
                if g.get("name") == "unfreezed_stage_0"
            )
            target_lr_0 = scheduler.get_last_lr()[0] * (0.9**0)
            assert pytest.approx(stage0_group["lr"], rel=1e-3) == 1.0 * target_lr_0

        elif epoch == 5:
            # Cooldown expired -> Stage 1 unlocked!
            assert training_state.unlocked_stage_count == 2
            assert training_state.stages_states[1].unlocked
            assert training_state.stages_states[1].unlocked_epoch == 5
            assert len(optimizer.param_groups) == 5
            assert optimizer.param_groups[-1]["name"] == "unfreezed_stage_1"
            # Block 4 parameters must now have requires_grad == True
            block4 = model.backbone.get_trainable_blocks()[4]
            assert all(p.requires_grad for p in block4.parameters())
            # Stage 1 warmup at start factor (10% of depth-scaled target_lr)
            target_lr_1 = scheduler.get_last_lr()[0] * (0.9**1)
            stage1_group = next(
                g
                for g in optimizer.param_groups
                if g.get("name") == "unfreezed_stage_1"
            )
            assert pytest.approx(stage1_group["lr"], rel=1e-3) == 0.1 * target_lr_1

        elif epoch == 6:
            # max_stages is 2, no further stages are unlocked
            assert training_state.max_stages_condition()
            assert training_state.unlocked_stage_count == 2
            assert len(optimizer.param_groups) == 5


def test_unfreeze_gradient_flow_and_weight_update():
    """Verify that unfreezing a stage enables gradients
    to flow and updates weights during optimizer.step()."""
    device = torch.device("cpu")
    model_params = ModelParams(backbone="efficientnet", start_unfreezed=1)
    model = build_pipeline_model(device, model_params)
    optimizer = build_pipeline_optimizer(model, OptimizerParams(base_lr=1e-3))

    block5 = model.backbone.get_trainable_blocks()[5]
    # Block 5 parameters are initially frozen
    assert all(not p.requires_grad for p in block5.parameters())

    # Unfreeze block 5
    stage = StageUnfreezeState(name="unfreezed_stage_0", local_warmup_len=3, blocks=[5])
    model.backbone.unfreeze_stage(stage, optimizer=optimizer, lr=1e-3)
    assert all(p.requires_grad for p in block5.parameters())

    # Pick a test parameter from block 5 to monitor
    test_param = next(block5.parameters())
    initial_weights = test_param.clone().detach()

    # Forward pass through PhenologyModel
    dummy_images = [torch.randn(2, 3, 224, 224)]
    predictions, _ = model(dummy_images)
    loss = predictions.sum()
    loss.backward()

    # Assert gradients exist and are non-zero
    assert test_param.grad is not None
    assert test_param.grad.abs().sum() > 0

    # Step optimizer and assert weights have updated
    optimizer.step()
    assert not torch.equal(test_param, initial_weights)

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir

from plant_pheno.config import (
    CONFIG_DIR,
    model_params_from_cfg,
    training_params_from_cfg,
)
from plant_pheno.config.choices import (
    available_choices,
    format_available_choices,
    validate_config_choices,
)


def compose_config(*overrides: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_default_config_uses_local_mil_pooling() -> None:
    cfg = compose_config()

    assert cfg.env_name == "local"
    assert cfg.paths.root == "/home/etienne/projects/inat-phenology-cv"
    assert cfg.model.architecture == "mil_pooling"
    assert cfg.command == "train"


def test_colab_environment_overrides_machine_settings() -> None:
    cfg = compose_config("env=colab")

    assert cfg.env_name == "colab"
    assert cfg.paths.root == "/content/inat-phenology-cv"
    assert cfg.dataloaders.batch_size == 256
    assert cfg.model.backbone == "bioclip"
    assert cfg.model.dinov3.checkpoint_path.endswith(
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )


def test_dinov3_checkpoint_setting_is_optional_and_explicit() -> None:
    cfg = compose_config(
        "model.backbone=dinov3",
        "model.dinov3.checkpoint_path=/content/weights.pth",
    )
    params = model_params_from_cfg(cfg.model)

    assert params.dinov3.checkpoint_path == "/content/weights.pth"


def test_cross_attention_group_sets_architecture_parameters() -> None:
    cfg = compose_config("architecture=cross_attention")

    assert cfg.model.architecture == "cross_attention"
    assert cfg.model.proj_dim == 256
    assert cfg.model.cross_attn_heads == 8


def test_cleanlab_command_keeps_selected_environment_paths() -> None:
    cfg = compose_config("command=cleanlab", "env=colab")

    assert cfg.command == "cleanlab"
    assert cfg.cleanlab.n_folds == 5
    assert cfg.paths.cleanlab_db_path == (
        "/content/drive/MyDrive/inat-phenology-cv/data/cleanlab.duckdb"
    )


def test_experiment_recipe_overrides_root_settings() -> None:
    cfg = compose_config("experiment=smoke_test")

    assert cfg.experiment_name == "test"
    assert cfg.test is True
    assert cfg.test_frac == 0.05
    assert cfg.train.epochs == 1


def test_progressive_unfreezing_is_disabled_by_default() -> None:
    cfg = compose_config()
    params = training_params_from_cfg(cfg)

    assert cfg.train.progressive_unfreezing.enabled is False
    assert params.unfreeze is False
    assert params.unfreezing_patience == 3
    assert params.unfreezing_cooldown == 3
    assert params.max_stages == 3
    assert params.block_per_stage == 1
    assert params.backbone_decay == 0.9


def test_unfreeze_experiment_enables_progressive_settings() -> None:
    cfg = compose_config("experiment=unfreeze")
    params = training_params_from_cfg(cfg)

    assert cfg.train.progressive_unfreezing.enabled is True
    assert params.unfreeze is True
    assert params.unfreezing_patience == 3
    assert params.unfreezing_cooldown == 3
    assert params.max_stages == 3
    assert params.block_per_stage == 1


def test_train_epoch_override_and_hydra_does_not_change_directory() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["train.epochs=3"],
            return_hydra_config=True,
        )

    assert cfg.train.epochs == 3
    assert cfg.hydra.job.chdir is False


def test_available_choices_include_backbones_and_hydra_groups() -> None:
    choices = available_choices()

    assert "bioclip" in choices["model.backbone"]
    assert "mil_pooling" in choices["architecture"]
    assert "cross_attention" in choices["architecture"]
    assert "local" in choices["env"]
    assert "options" in choices["command"]
    assert "model.backbone:" in format_available_choices()


def test_invalid_backbone_reports_available_values() -> None:
    cfg = compose_config()
    cfg.model.backbone = "not-a-backbone"

    with pytest.raises(ValueError, match="model.backbone='not-a-backbone'"):
        validate_config_choices(cfg)


def test_invalid_tuning_choice_reports_available_values() -> None:
    cfg = compose_config("command=tune")
    cfg.tune.sampler = "not-a-sampler"

    with pytest.raises(ValueError, match="tune.sampler='not-a-sampler'"):
        validate_config_choices(cfg)

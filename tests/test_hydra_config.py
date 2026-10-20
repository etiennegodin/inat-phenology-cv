from __future__ import annotations

from hydra import compose, initialize_config_dir

from plant_pheno.config import CONFIG_DIR


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


def test_train_epoch_override_and_hydra_does_not_change_directory() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["train.epochs=3"],
            return_hydra_config=True,
        )

    assert cfg.train.epochs == 3
    assert cfg.hydra.job.chdir is False

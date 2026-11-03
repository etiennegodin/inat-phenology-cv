from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from plant_pheno import cli
from plant_pheno.config import CONFIG_DIR


def test_tune_retraining_reuses_best_trial_seed(monkeypatch) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=["command=tune", "seed=41", "tune.n_trials=1"],
        )

    runtime = SimpleNamespace(
        dataset_params=SimpleNamespace(cache=SimpleNamespace(enable=False)),
        paths_params=SimpleNamespace(optuna_storage="sqlite:///test.db"),
    )
    run = SimpleNamespace(info=SimpleNamespace(run_id="test-run-id"))
    study = SimpleNamespace(
        best_params={"base_lr": 0.001},
        best_trial=SimpleNamespace(number=3),
        best_value=0.5,
        set_user_attr=lambda *_args: None,
        optimize=lambda *_args, **_kwargs: None,
    )
    training_params = []
    seeds = []
    study_kwargs = {}
    storage_urls = []
    storage = object()

    monkeypatch.setattr(cli, "cfg_to_runtime", lambda _cfg: runtime)
    monkeypatch.setattr(cli, "_require_colab_gpu", lambda _cfg: None)
    monkeypatch.setattr(cli, "get_device", lambda: "cpu")
    monkeypatch.setattr(cli, "resolve_uri", lambda: "file:///tmp/mlruns")
    monkeypatch.setattr(cli.mlflow, "set_tracking_uri", lambda *_args: None)
    monkeypatch.setattr(cli.mlflow, "set_experiment", lambda *_args: None)
    monkeypatch.setattr(cli.mlflow, "start_run", lambda: nullcontext(run))
    monkeypatch.setattr(cli.mlflow, "log_dict", lambda *_args: None)
    monkeypatch.setattr(cli.mlflow, "log_params", lambda *_args: None)
    monkeypatch.setattr(cli.mlflow, "log_metric", lambda *_args: None)
    monkeypatch.setattr(cli.mlflow, "set_tag", lambda *_args: None)
    monkeypatch.setattr(cli, "_log_resolved_config", lambda *_args: None)
    monkeypatch.setattr(cli, "_log_run_file", lambda: None)
    monkeypatch.setenv(
        "OPTUNA_STORAGE_URL",
        "postgresql+psycopg2://optuna_user:secret@100.64.0.1:5432/optuna",
    )
    monkeypatch.setattr(
        cli.optuna.storages,
        "RDBStorage",
        lambda *, url: storage_urls.append(url) or storage,
    )
    monkeypatch.setattr(
        cli, "seed_everything", lambda seed, **_kwargs: seeds.append(seed)
    )
    monkeypatch.setattr(cli, "build_data_pipeline", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "_instantiate", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        cli.optuna,
        "create_study",
        lambda **kwargs: study_kwargs.update(kwargs) or study,
    )
    monkeypatch.setattr(cli, "make_objective", lambda **_kwargs: object())
    monkeypatch.setattr(
        cli,
        "training_params_from_cfg",
        lambda _cfg, **kwargs: training_params.append(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setattr(cli.train, "run_training", lambda **_kwargs: None)

    cli.tune_cmd(cfg)

    expected_seed = cfg.seed + study.best_trial.number
    assert seeds == [cfg.seed, expected_seed]
    assert training_params[0]["seed"] == expected_seed
    assert storage_urls == [
        "postgresql+psycopg2://optuna_user:secret@100.64.0.1:5432/optuna"
    ]
    assert study_kwargs["storage"] is storage


def test_optuna_storage_uses_config_fallback_when_env_is_unset(monkeypatch) -> None:
    monkeypatch.delenv("OPTUNA_STORAGE_URL", raising=False)
    storage_urls = []
    storage = object()
    monkeypatch.setattr(
        cli.optuna.storages,
        "RDBStorage",
        lambda *, url: storage_urls.append(url) or storage,
    )

    assert cli._create_optuna_storage("sqlite:///data/optuna.sqlite3") is storage
    assert storage_urls == ["sqlite:///data/optuna.sqlite3"]


def test_optuna_storage_rejects_empty_url(monkeypatch) -> None:
    monkeypatch.setenv("OPTUNA_STORAGE_URL", " ")

    with pytest.raises(ValueError, match="Optuna storage URL is empty"):
        cli._create_optuna_storage("sqlite:///data/optuna.sqlite3")

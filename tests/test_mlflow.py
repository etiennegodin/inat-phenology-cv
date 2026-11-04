from __future__ import annotations

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from plant_pheno.config import CONFIG_DIR
from plant_pheno.infra import mlflow_helpers
from plant_pheno.infra.mlflow_helpers import (
    resolve_mlflow_experiment,
    resolve_resume_run,
)


def compose_config(*overrides: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides))


def test_resolve_mlflow_experiment():
    cfg = compose_config("command=train", "model.backbone=dinov3")
    experiment = resolve_mlflow_experiment(cfg)
    assert experiment.name == "dinov3_mil_pooling_train"


def test_resolve_mlflow_experiment_test():
    cfg = compose_config("command=train", "model.backbone=dinov3", "test=True")
    experiment = resolve_mlflow_experiment(cfg)
    assert experiment.name == "dinov3_mil_pooling_train_test"


def test_resolve_resume_run_requires_name():
    with pytest.raises(ValueError):
        resolve_resume_run("123")


def test_resolve_resume_run_returns_run_id(monkeypatch):
    def fake_search_runs(**kwargs):
        assert kwargs == {
            "experiment_ids": ["123"],
            "filter_string": "tags.mlflow.runName = 'baseline'",
            "output_format": "pandas",
        }
        return pd.DataFrame([{"run_id": "run-abc"}])

    monkeypatch.setattr(mlflow_helpers.mlflow, "search_runs", fake_search_runs)

    assert mlflow_helpers.resolve_resume_run("123", "baseline") == "run-abc"


def test_resolve_resume_run_can_search_all_experiments(monkeypatch):
    def fake_search_runs(**kwargs):
        assert kwargs == {
            "filter_string": "tags.mlflow.runName = 'baseline'",
            "output_format": "pandas",
            "search_all_experiments": True,
        }
        return pd.DataFrame([{"run_id": "run-abc"}])

    monkeypatch.setattr(mlflow_helpers.mlflow, "search_runs", fake_search_runs)

    assert mlflow_helpers.resolve_resume_run(None, "baseline") == "run-abc"


def test_resolve_resume_run_raises_when_no_run_matches(monkeypatch):
    monkeypatch.setattr(
        mlflow_helpers.mlflow,
        "search_runs",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(ValueError):
        mlflow_helpers.resolve_resume_run("123", "missing")

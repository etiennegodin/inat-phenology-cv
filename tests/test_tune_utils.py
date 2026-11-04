from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from plant_pheno.config import SEARCH_SPACES
from plant_pheno.tune.utils import load_tune_resume_config


def test_load_tune_resume_config_restores_parent_config_and_tune_params(
    monkeypatch,
) -> None:
    config = {
        "command": "tune",
        "seed": 41,
        "model": {"backbone": "bioclip"},
        "tune": {
            "n_trials": 5,
            "timeout": 100,
            "resume": False,
            "resume_name": None,
        },
    }
    tune_params = {
        "n_trials": 12,
        "timeout": 600,
        "sampler": "random",
        "pruner": "none",
        "subsample_frac": 0.5,
        "search_space": "coarse",
        "study_name": "original-study",
    }
    saved_search_space = {
        "base_lr": {"type": "float", "low": 2e-5, "high": 8e-4, "log": True}
    }
    loaded_uris = []
    monkeypatch.setattr(
        "plant_pheno.tune.utils.mlflow.MlflowClient",
        lambda: SimpleNamespace(
            list_artifacts=lambda _run_id: [
                SimpleNamespace(path="configs.json"),
                SimpleNamespace(path="tune_params.json"),
                SimpleNamespace(path="tune_search_space.json"),
            ]
        ),
    )

    def load_dict(uri):
        loaded_uris.append(uri)
        if uri.endswith("configs.json"):
            return deepcopy(config)
        if uri.endswith("tune_params.json"):
            return tune_params
        return saved_search_space

    monkeypatch.setattr("plant_pheno.tune.utils.mlflow.artifacts.load_dict", load_dict)

    restored, search_space = load_tune_resume_config(
        "run-123",
        "parent-study",
        SEARCH_SPACES,
    )

    assert loaded_uris == [
        "runs:/run-123/configs.json",
        "runs:/run-123/tune_params.json",
        "runs:/run-123/tune_search_space.json",
    ]
    assert search_space == saved_search_space
    assert restored.seed == 41
    assert restored.model.backbone == "bioclip"
    assert restored.tune.n_trials == 12
    assert restored.tune.sampler == "random"
    assert restored.tune.search_space == "coarse"
    assert restored.tune.study_name == "original-study"
    assert restored.tune.resume is True
    assert restored.tune.resume_name == "parent-study"


def test_load_tune_resume_config_reads_search_space_from_legacy_signature(
    monkeypatch,
) -> None:
    config = {
        "command": "tune",
        "tune": {"search_space": "coarse", "resume": False},
    }
    saved_search_space = {
        "base_lr": {"type": "float", "low": 5e-5, "high": 5e-4, "log": True}
    }
    signature = {"tuning_protocol": {"search_space": saved_search_space}}
    monkeypatch.setattr(
        "plant_pheno.tune.utils.mlflow.MlflowClient",
        lambda: SimpleNamespace(
            list_artifacts=lambda _run_id: [
                SimpleNamespace(path="configs.json"),
                SimpleNamespace(path="tune_resume_signature.json"),
            ]
        ),
    )
    monkeypatch.setattr(
        "plant_pheno.tune.utils.mlflow.artifacts.load_dict",
        lambda uri: signature if "signature" in uri else config,
    )

    _, search_space = load_tune_resume_config(
        "run-123",
        "parent-study",
        SEARCH_SPACES,
    )

    assert search_space == saved_search_space


def test_load_tune_resume_config_falls_back_to_named_space_for_old_runs(
    monkeypatch,
) -> None:
    config = {"command": "tune", "tune": {"search_space": "coarse"}}
    monkeypatch.setattr(
        "plant_pheno.tune.utils.mlflow.MlflowClient",
        lambda: SimpleNamespace(
            list_artifacts=lambda _run_id: [SimpleNamespace(path="configs.json")]
        ),
    )
    monkeypatch.setattr(
        "plant_pheno.tune.utils.mlflow.artifacts.load_dict",
        lambda _uri: config,
    )

    _, search_space = load_tune_resume_config(
        "run-123",
        "parent-study",
        SEARCH_SPACES,
    )

    assert search_space == SEARCH_SPACES["coarse"]

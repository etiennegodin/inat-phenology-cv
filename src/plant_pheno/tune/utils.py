from __future__ import annotations

import logging
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

TUNE_RESUME_SIGNATURE_ARTIFACT = "tune_resume_signature.json"


def resolve_study_name(cfg, run_id):
    if cfg.tune.study_name is not None:
        return cfg.tune.study_name

    study_name = f"{cfg.model.backbone}_{cfg.model.architecture}_{run_id[:8]}"
    study_name += f"{'_test' if cfg.test else ''}"
    return study_name


def load_tune_resume_config(
    run_id: str,
    resume_name: str,
    search_spaces: dict[str, dict[str, Any]],
) -> tuple[DictConfig, dict[str, Any]]:
    """Load the original resolved config and search space from a parent run."""
    client = mlflow.MlflowClient()
    artifacts = {artifact.path for artifact in client.list_artifacts(run_id)}
    if "configs.json" not in artifacts:
        raise ValueError(
            f"Cannot resume tuning run {run_id}: parent run is missing configs.json"
        )

    config_data = mlflow.artifacts.load_dict(f"runs:/{run_id}/configs.json")
    if config_data.get("command") != "tune":
        raise ValueError(
            f"Cannot resume run {run_id}: its logged command is "
            f"{config_data.get('command')!r}, not 'tune'"
        )

    if "tune_params.json" in artifacts:
        tune_params = mlflow.artifacts.load_dict(f"runs:/{run_id}/tune_params.json")
        tune_config = config_data.setdefault("tune", {})
        for key in ("n_trials", "timeout", "sampler", "pruner", "subsample_frac"):
            if key in tune_params:
                tune_config[key] = tune_params[key]
        if "search_space" in tune_params:
            tune_config["search_space"] = tune_params["search_space"]
        if tune_params.get("study_name") is not None:
            tune_config["study_name"] = tune_params["study_name"]

    search_space_name = config_data.get("tune", {}).get("search_space")
    if TUNE_RESUME_SIGNATURE_ARTIFACT in artifacts:
        signature = mlflow.artifacts.load_dict(
            f"runs:/{run_id}/{TUNE_RESUME_SIGNATURE_ARTIFACT}"
        )
        search_space = signature.get("tuning_protocol", {}).get("search_space")
        if not isinstance(search_space, dict):
            raise ValueError(
                f"Cannot resume tuning run {run_id}: its saved signature has no "
                "search-space definition"
            )
    elif "tune_search_space.json" in artifacts:
        search_space = mlflow.artifacts.load_dict(
            f"runs:/{run_id}/tune_search_space.json"
        )
        if not isinstance(search_space, dict):
            raise ValueError(
                f"Cannot resume tuning run {run_id}: tune_search_space.json "
                "does not contain a mapping"
            )
    else:
        try:
            search_space = search_spaces[search_space_name]
        except KeyError as error:
            raise ValueError(
                f"Cannot resume tuning run {run_id}: search space "
                f"{search_space_name!r} is unavailable"
            ) from error

    config_data.setdefault("tune", {})["resume"] = True
    config_data["tune"]["resume_name"] = resume_name
    return OmegaConf.create(config_data), search_space

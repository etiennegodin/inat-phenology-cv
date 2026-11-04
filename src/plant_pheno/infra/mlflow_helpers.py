from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mlflow.entities import Experiment


def resolve_resume_run(
    experiment_id: str | None,
    run_name: str | None = None,
):
    if run_name is None:
        raise ValueError(
            f"Attempted to resume run without name provided. run_name: {run_name}"
        )

    # Make sure experiment_id is a str
    if experiment_id is not None and not isinstance(experiment_id, str):
        experiment_id = str(experiment_id)

    # Search for the specific run name
    escaped_name = run_name.replace("'", "''")
    search_kwargs: dict[str, Any] = {
        "filter_string": f"tags.mlflow.runName = '{escaped_name}'",
        "output_format": "pandas",
    }
    if experiment_id is None:
        search_kwargs["search_all_experiments"] = True
    else:
        search_kwargs["experiment_ids"] = [str(experiment_id)]

    runs = mlflow.search_runs(**search_kwargs)

    if len(runs) == 1:
        return runs.iloc[0]["run_id"]
    if len(runs) > 1:
        raise ValueError(
            f"Multiple MLflow runs are named {run_name!r}; "
            "use a unique run name before resuming"
        )

    raise ValueError(
        f"Didn't find a run under {run_name} for experiment {experiment_id}"
    )


def resolve_mlflow_experiment(cfg: DictConfig) -> Experiment:
    """Resolve mlflow run id and experiment name from configs"""
    experiment_name = f"{cfg.model.backbone}_{cfg.model.architecture}"
    experiment_name += f"_{cfg.command}{'_test' if cfg.test else ''}"
    experiment = mlflow.set_experiment(experiment_name)
    return experiment


def save_log():
    import mlflow

    if mlflow.active_run():
        mlflow.log_artifact(str(Path.cwd() / "log.log"))


def get_mlflow_run_id() -> str | None:
    import mlflow

    return mlflow.active_run().info.run_id if mlflow.active_run() else None


def resolve_uri() -> str:
    uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )
    logger.debug(uri)
    return uri


def patch_mlflow_socks():
    try:
        import requests
    except ImportError:
        return

    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(var, None)

    mlflow_hosts = ("100.101.196.27",)
    socks_proxies = {
        "http": "socks5h://localhost:1055",
        "https": "socks5h://localhost:1055",
    }

    original_session_request = requests.Session.request

    def _scoped_request(self, method, url, *args, **kwargs):
        if any(host in url for host in mlflow_hosts):
            kwargs.setdefault("proxies", socks_proxies)
        return original_session_request(self, method, url, *args, **kwargs)

    requests.Session.request = _scoped_request

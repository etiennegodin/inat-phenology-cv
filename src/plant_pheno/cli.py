from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import mlflow
import optuna
from dotenv import load_dotenv
from mlflow import MlflowException
from omegaconf import DictConfig, OmegaConf
from torch import cuda, nn

from . import cache, train
from .config import (
    CONFIG_DIR,
    OPTUNA_PRUNERS,
    OPTUNA_SAMPLERS,
    SEARCH_SPACES,
    InferenceParams,
    IngestPhotosParams,
    cfg_to_runtime,
    model_params_from_cfg,
    optimizer_params_from_cfg,
    resolved_container,
    scheduler_params_from_cfg,
    training_params_from_cfg,
)
from .config.choices import format_available_choices, validate_config_choices
from .core.device import get_device, set_device
from .data import log_model_evaluation
from .inference import InatInferencePipeline
from .infra import init_logger, mlflow_socks_patch, resolve_uri, seed_everything  # noqa
from .status import status
from .train import (
    build_cached_data_pipeline,
    build_data_pipeline,
    build_datasets,
    build_pipeline_dataloaders,
)
from .tune import make_objective
from .utils import (
    _instantiate,
    get_pos_ratios,
    get_pos_weights,
)

if TYPE_CHECKING:
    from .core.model import PhenologyModel

mlflow.enable_system_metrics_logging()
mlflow.system_metrics.set_system_metrics_sampling_interval(10)
mlflow.system_metrics.set_system_metrics_samples_before_logging(3)

COMMANDS: dict[str, callable] = {}


def _command(name: str):
    def decorator(fn):
        COMMANDS[name] = fn
        return fn

    return decorator


def _require_colab_gpu(cfg: DictConfig) -> None:
    if cfg.env_name == "colab" and not cuda.is_available():
        raise RuntimeError(
            "Attempted to load env=colab on a non-gpu session "
            f"(env_name={cfg.env_name})"
        )


def _log_resolved_config(cfg: DictConfig, runtime) -> None:
    payload = resolved_container(cfg)
    mlflow.log_dict(payload, "configs.json")
    mlflow.log_params({"git_branch": runtime.git_branch})
    mlflow.log_params({"git_hash": runtime.git_hash})
    mlflow.log_params(runtime.dataset_params.to_dict())
    mlflow.log_params(runtime.dataloaders_params.to_dict())


def _log_run_file() -> None:
    log_path = Path.cwd() / "log.log"
    if log_path.exists():
        mlflow.log_artifact(str(log_path))


@_command("cache")
def cache_cmd(cfg: DictConfig):
    runtime = cfg_to_runtime(cfg)
    seed_everything(cfg.seed, set_cuda_deterministic=False)
    _require_colab_gpu(cfg)

    model_params = model_params_from_cfg(cfg.model)

    cache.execute(
        configs=runtime,
        model_params=model_params,
        seed=cfg.seed,
    )


@_command("train")
def train_cmd(cfg: DictConfig):
    runtime = cfg_to_runtime(cfg)
    seed_everything(cfg.seed, set_cuda_deterministic=False)
    _require_colab_gpu(cfg)

    device = get_device()
    print("Connecting to mlflow")
    mlflow.set_tracking_uri(resolve_uri())
    print("Initalizing experiment")

    model_params = model_params_from_cfg(cfg.model)
    optim_params = optimizer_params_from_cfg(cfg)
    scheduler_params = scheduler_params_from_cfg(cfg)
    training_params = training_params_from_cfg(cfg, best_objective=1e-5)

    print(runtime.dataset_params.cache.enable)
    print(type(runtime.dataset_params.cache.enable))
    if runtime.dataset_params.cache.enable:
        cache_dir = cache.ensure_cached_embeddings(runtime, model_params, cfg.seed)
        data = build_cached_data_pipeline(runtime, cache_dir, device, seed=cfg.seed)
    else:
        data = build_data_pipeline(runtime, cfg.model.backbone, device, seed=cfg.seed)

    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"\n{'=' * 60}")
        print(f"MLflow Run ID: {parent_run_id}")
        print(f"{'=' * 60}\n")

        _log_resolved_config(cfg, runtime)
        train.run_training(
            configs=runtime,
            model_params=model_params,
            optim_params=optim_params,
            scheduler_params=scheduler_params,
            training_params=training_params,
            data=data,
            device=device,
            use_cached=runtime.dataset_params.cache.enable,
        )
        _log_run_file()


@_command("tune")
def tune_cmd(cfg: DictConfig):
    runtime = cfg_to_runtime(cfg)
    _require_colab_gpu(cfg)
    device = get_device()
    mlflow.set_tracking_uri(resolve_uri())
    mlflow.set_experiment(cfg.experiment_name)
    seed_everything(cfg.seed, set_cuda_deterministic=False)

    if runtime.dataset_params.cache.enable:
        # Base model params from compose configs
        # Not fed in tuning workflow and actual hp reinstanced per objective
        default_model_params = model_params_from_cfg(cfg.model)
        cache_dir = cache.ensure_cached_embeddings(
            runtime, default_model_params, cfg.seed
        )
        data = build_cached_data_pipeline(runtime, cache_dir, device, seed=cfg.seed)
    else:
        data = build_data_pipeline(runtime, cfg.model.backbone, device, seed=cfg.seed)

    with mlflow.start_run() as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"\n{'=' * 60}")
        print(f"MLflow Run ID: {parent_run_id}")
        print(f"{'=' * 60}\n")

        study_name = cfg.tune.study_name or f"{cfg.experiment_name}_{parent_run_id[:8]}"
        tune_params = {
            "n_trials": cfg.tune.n_trials,
            "timeout": cfg.tune.timeout,
            "sampler": cfg.tune.sampler,
            "pruner": cfg.tune.pruner,
            "study_name": study_name,
            "subsample_frac": cfg.tune.subsample_frac,
            "search_space": cfg.tune.search_space,
            "resume": cfg.tune.resume,
        }
        mlflow.log_dict(tune_params, "tune_params.json")
        _log_resolved_config(cfg, runtime)
        mlflow.set_tag("optuna.study_name", study_name)

        study = optuna.create_study(
            study_name=study_name,
            storage=runtime.paths_params.optuna_storage,
            load_if_exists=bool(cfg.tune.resume),
            direction="maximize",
            sampler=_instantiate(OPTUNA_SAMPLERS, cfg.tune.sampler, {"seed": cfg.seed}),
            pruner=_instantiate(
                OPTUNA_PRUNERS,
                cfg.tune.pruner,
                {"n_startup_trials": 3, "n_warmup_steps": 2},
            ),
        )
        study.set_user_attr("mlflow_run_id", parent_run_id)
        study.set_user_attr("mlflow_experiment", cfg.experiment_name)

        objective = make_objective(
            configs=runtime,
            data=data,
            cfg=cfg,
            search_space=SEARCH_SPACES[cfg.tune.search_space],
            subsample_frac=cfg.tune.subsample_frac,
        )

        study.optimize(
            objective,
            n_trials=cfg.tune.n_trials,
            show_progress_bar=False,
            callbacks=[],
        )

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_dict(study.best_params, "best_params.json")
        mlflow.log_metric("best_pr_norm_excess_macro", study.best_value)
        print(
            f"\n[Optuna] Best trial: #{study.best_trial.number} "
            f" with value {study.best_value:.5f}"
        )
        print(f"[Optuna] Best params: {study.best_params}")
        print("Retraining with full model & best params")

        best = study.best_params
        model_params = model_params_from_cfg(cfg.model, **best)
        training_params = training_params_from_cfg(
            cfg,
            best_objective=1e-5,
            backbone_decay=best.get("backbone_decay"),
        )
        train.run_training(
            configs=runtime,
            model_params=model_params,
            optim_params=optimizer_params_from_cfg(
                cfg, base_lr=best.get("base_lr", cfg.train.base_lr)
            ),
            scheduler_params=scheduler_params_from_cfg(cfg),
            training_params=training_params,
            data=data,
            device=device,
            use_cached=runtime.dataset_params.cache.enable,
        )
        _log_run_file()


@_command("predict")
def inference_cmd(cfg: DictConfig):
    runtime = cfg_to_runtime(cfg)
    photo_params = IngestPhotosParams()
    inference_params = InferenceParams(
        cfg.predict.model_name,
        cfg.predict.model_version,
        db_path=runtime.paths_params.inference_db_path,
        photo_target_dir=runtime.paths_params.photo_target_dir,
        sql_dir=runtime.paths_params.sql_dir,
    )
    pipeline = InatInferencePipeline.from_params(inference_params)
    urls = OmegaConf.to_container(cfg.predict.urls) if cfg.predict.urls else None
    pipeline.execute(urls, photo_params)


@_command("val")
def val_cmd(cfg: DictConfig):
    runtime = cfg_to_runtime(cfg)
    print("Connecting to mlflow")
    mlflow.set_tracking_uri(resolve_uri())
    device = set_device(cfg.val.device)
    model_uri = f"models:/{cfg.val.model_name}/{cfg.val.model_version}"

    try:
        python_model: PhenologyModel = mlflow.pytorch.load_model(
            model_uri, map_location=device
        )
        python_model.to(device)
    except MlflowException:
        model = mlflow.pyfunc.load_model(model_uri)
        python_model: PhenologyModel = model._model_impl.python_model.model

    datasets = build_datasets(
        runtime, python_model.backbone.get_transforms(), seed=cfg.seed
    )
    _, val_loader, _ = build_pipeline_dataloaders(
        datasets, runtime.dataloaders_params, seed=cfg.seed
    )

    pos_weights = get_pos_weights(datasets[0], runtime.dataset_params, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")

    eval_metrics, _ = train.evaluate(
        model=python_model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        epoch=0,
        pos_ratios=get_pos_ratios(datasets[1]),
    )
    from pprint import pprint

    pprint(eval_metrics)

    if not cfg.val.no_db:
        eval_db_path = cfg.val.eval_db_path or runtime.paths_params.eval_db_path
        eval_id = log_model_evaluation(
            eval_db_path=eval_db_path,
            model_name=cfg.val.model_name,
            model_version=cfg.val.model_version,
            dataset_path=runtime.paths_params.db_path,
            dataset_table=runtime.dataset_params.source_table,
            seed=cfg.seed,
            is_test_mode=cfg.test,
            metrics=eval_metrics,
        )
        print(
            f"\n[DB] Model evaluation results committed to "
            f"'{eval_db_path}' (eval_id: {eval_id})"
        )


@_command("list")
def list_model_cmd(cfg: DictConfig):
    from mlflow import MlflowClient

    print("Connecting to mlflow")
    mlflow.set_tracking_uri(resolve_uri())
    client = MlflowClient()
    registered_models = client.search_registered_models()
    for rm in registered_models:
        print(f"--- Model: {rm.name} ---")
        print(f"Creation Timestamp: {rm.creation_timestamp}")
        print(f"Last Updated: {rm.last_updated_timestamp}")
        for version in rm.latest_versions:
            print(f"  -> Version: {version.version} | Stage: {version.current_stage}")


@_command("options")
def options_cmd(cfg: DictConfig):
    print(format_available_choices())


@_command("test")
def test_cmd(cfg: DictConfig):
    print("test command is not implemented")


@_command("status")
def status_cmd(cfg: DictConfig):
    status(cfg_to_runtime(cfg))


@_command("cleanlab")
def cleanlab_cmd(cfg: DictConfig):
    from .cleanlab import run_cleanlab_cv

    return run_cleanlab_cv(cfg)


def _inject_colab_env() -> None:
    if not os.environ.get("COLAB_GPU"):
        return
    if any(arg.startswith("env=") for arg in sys.argv[1:]):
        return
    sys.argv.append("env=colab")


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def hydra_entry(cfg: DictConfig):
    log_path = Path.cwd() / "log.log"
    logger = init_logger(log_path, logging.INFO)
    logger.info("Starting")
    logger.debug("Configs set up")

    validate_config_choices(cfg)
    command = cfg.command
    if command not in COMMANDS:
        raise ValueError(
            f"Unknown command={command!r}. Choose one of: {sorted(COMMANDS)}"
        )
    return COMMANDS[command](cfg)


def main():
    load_dotenv()
    _inject_colab_env()
    hydra_entry()


if __name__ == "__main__":
    main()

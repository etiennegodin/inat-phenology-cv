from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import yaml
from dotenv import load_dotenv
from torch import cuda, nn

from . import train, val
from .status import status
from .train import (
    build_datasets,
    build_pipeline_dataloaders,
    build_pipeline_model,
    build_pipeline_optimizer,
    build_scheduler,
    get_device,
)
from .train.backbone import BACKBONE_REGISTRY
from .train.metrics import log_experiment_metadata
from .utils import (
    Config,
    clean_data,
    get_current_git_branch,
    get_pos_ratios,
    get_pos_weights,
    init_logger,
    mlflow_socks_patch,  # noqa
    resolve_env_config_path,
    resolve_hardware_profile,
    resolve_uri,
    seed_everything,
    update_dataset,
)
from .utils.params import (
    DataLoadersParams,
    DatasetParams,
    ModelParams,
    OptimizerParams,
    PathsParams,
    SchedulerParams,
    TrainingParams,
)

mlflow.enable_system_metrics_logging()
mlflow.system_metrics.set_system_metrics_sampling_interval(10)
mlflow.system_metrics.set_system_metrics_samples_before_logging(3)


if TYPE_CHECKING:
    from .train.model import PhenologyModel


def train_cmd(args, configs: Config):
    # Set RNG seed across Python random, numpy, torch, cuda
    seed_everything(args.seed, set_cuda_deterministic=False)

    best_objective = 1e-5

    # Fast fail if in colab without GPU
    if not cuda.is_available() and configs.config_path == Path("configs/colab.yaml"):
        raise RuntimeError(
            f"Attempted to load {configs.config_path} on a non-gpu colab session"
        )

    # Set test
    configs.test = args.test

    print("Connecting to mlflow")
    mlflow.set_tracking_uri(resolve_uri())

    print("Initalizing experiment")
    # // Dataset params
    dataset_params = DatasetParams(testing_frac=args.test_frac)
    configs.dataset_params = dataset_params

    # // Optimiser params
    optim_params = OptimizerParams(
        base_lr=args.base_lr,
    )
    configs.optim_params = optim_params

    # // Scheduler params
    scheduler_params = SchedulerParams(
        warmup_epochs=args.warmup_epochs, total_epoch=args.epochs
    )
    configs.scheduler_params = scheduler_params

    # // Model params
    model_params = ModelParams(
        args.backbone,
        head_neurons=256,
        head_outputs=1,
        head_dropout_prob=0.5,
        attention_neurons=128,
        attention_dropout_prob=args.attention_dropout,
        last_blocks=args.unfreeze,
        gated=args.gated,
    )

    configs.model_params = model_params

    # Initialise train modules
    device = get_device()
    model = build_pipeline_model(device, model_params)
    optimizer = build_pipeline_optimizer(model, optim_params)
    scheduler = build_scheduler(optimizer, scheduler_params)
    datasets = build_datasets(configs, model, seed=args.seed)
    train_loader, val_loader, _ = build_pipeline_dataloaders(
        datasets, configs.dataloaders_params, seed=args.seed
    )

    pos_weights = get_pos_weights(datasets[0], configs.dataset_params, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")

    # Reinstate model and optimizer state if reload
    if args.reload:
        raise NotImplementedError("Training reloading not implemented")
        """
        model, optimizer, start_epoch, eval_metrics, previous_run_id = load_checkpoint(
            configs.paths_params.checkpoint_path, model=model, optimizer=optimizer
        )
        if eval_metrics is not None:
            best_objective = eval_metrics.pr_norm_excess_macro
        """

    else:
        start_epoch = None
        previous_run_id = None

    training_params = TrainingParams(
        epochs=args.epochs,
        patience=args.patience,
        start_epoch=start_epoch,
        best_objective=best_objective,
        seed=args.seed,
        log_step_interval=args.log_step_interval,
        pos_ratios=get_pos_ratios(datasets[1]),
    )

    # Set configs params
    configs.training_params = training_params

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_id=previous_run_id) as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"\n{'=' * 60}")
        print(f"MLflow Run ID: {parent_run_id}")
        print(f"{'=' * 60}\n")

        mlflow.log_dict(configs.to_dict(), "configs.json")
        mlflow.log_params(model_params.to_dict())
        mlflow.log_params(training_params.to_dict())
        mlflow.log_params(dataset_params.to_dict())
        mlflow.log_params(configs.dataloaders_params.to_dict())
        mlflow.log_params(optim_params.to_dict())
        mlflow.log_params(scheduler_params.to_dict())

        log_experiment_metadata(
            model=model,
            train_dataset=datasets[0],
            val_dataset=datasets[1],
        )

        train.execute(
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            checkpoint_path=configs.paths_params.checkpoint_path,
            training_params=configs.training_params,
        )

        """
        # Log and register best model
        pytorch.log_model(best_model, "cv_inat")
        model_uri = f"runs:/{parent_run_id}/cv_inat"
        mlflow.register_model(model_uri, "cv_inat")
        """

        # Log accumulated run logs to MLflow
        log_path = Path.cwd() / "log.log"
        if log_path.exists():
            mlflow.log_artifact(str(log_path))


def val_cmd(args, configs: Config):

    print("Connecting to mlflow")
    mlflow.set_tracking_uri(resolve_uri())
    # Set test
    configs.test = args.test
    device = get_device()
    dataset_params = DatasetParams(testing_frac=args.test_frac)
    configs.dataset_params = dataset_params

    # Construct the model URI
    model_uri = f"models:/{args.model_name}/{args.model_version}"

    # Load the native PyTorch model
    model = mlflow.pytorch.load_model(model_uri)
    model: PhenologyModel
    datasets = build_datasets(configs, model)
    _, val_loader, _ = build_pipeline_dataloaders(datasets, configs.dataloaders_params)

    x, y = val.execute(model=model, dataloader=val_loader, device=device)


def list_model_cmd(args, configs: Config):
    from mlflow import MlflowClient

    print("Connecting to mlflow")

    mlflow.set_tracking_uri(resolve_uri())

    # Initialize the client
    client = MlflowClient()

    # Search for all registered models
    registered_models = client.search_registered_models()
    # Fetch all registered models

    # Print out the names of the models
    for rm in registered_models:
        print(f"--- Model: {rm.name} ---")
        print(f"Creation Timestamp: {rm.creation_timestamp}")
        print(f"Last Updated: {rm.last_updated_timestamp}")

        # Iterate through individual versions of the model
        for version in rm.latest_versions:
            print(f"  -> Version: {version.version} | Stage: {version.current_stage}")


def test_cmd(args, configs: Config):
    """

    # Initialise Data loaders params
    if cuda.is_available():
        dataloader_params = DataLoadersParams(
            batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True
        )
    else:
        dataloader_params = DataLoadersParams(
            batch_size=16, num_workers=0, pin_memory=False, persistent_workers=False
        )
    configs.dataloaders_params = dataloader_params
    model_params = ModelParams(
        head_neurons=256,
        head_outputs=1,
        head_dropout_prob=0.5,
        attention_dim=0,
        attention_neurons=128,
    )
    device = get_device()
    model = build_pipeline_model(device, model_params)
    criterion = nn.BCEWithLogitsLoss()

    # Dummy optimiser
    optimizer = build_pipeline_optimizer(model, OptimizerParams())

    # Tes loaders
    loaders = build_pipeline_dataloaders(configs, model)
    model, optimizer, start_epoch, eval_metrics, previous_run_id = load_checkpoint(
        configs.paths_params.checkpoint_path, model=model, optimizer=optimizer
    )

    test.execute(device, model, loaders[2], criterion)
    """


def update_cmd(args, configs: Config):
    # Clean and update dataset
    clean_data(configs.paths_params.image_dir)
    update_dataset(configs.paths_params)


def status_cmd(args, configs: Config):
    status(configs)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torch_pipe")
    subparsers = parser.add_subparsers(
        title="commands", description="Available commands"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    add_train_args(train_parser)
    add_common_args(train_parser)
    train_parser.set_defaults(func=train_cmd)

    # Val command
    val_parser = subparsers.add_parser("val", help="Run inference on val set")
    add_val_args(val_parser)
    add_common_args(val_parser)
    val_parser.set_defaults(func=val_cmd)

    # List model command
    list_parser = subparsers.add_parser("list", help="List registered models")
    list_parser.set_defaults(func=list_model_cmd)

    # Test command
    test_parser = subparsers.add_parser("test", help="Test model")
    # add_train_args(train_parser)
    test_parser.set_defaults(func=test_cmd)

    # Update command
    update_parser = subparsers.add_parser("update", help="Update source dataset")
    update_parser.set_defaults(func=update_cmd)

    # Status command
    status_parser = subparsers.add_parser("status", help="Pipeline status")
    status_parser.set_defaults(func=status_cmd)
    return parser


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--test", "-t", action="store_true", default=False)
    parser.add_argument(
        "--test_frac",
        "-tf",
        help="Fraction of intial dataset to keep for testing",
        type=float,
        default=0.33,
    )


def add_val_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_version", "-mv", type=int, default=1)
    parser.add_argument("--model_name", "-mn", type=str, default="cv_inat")


def add_train_args(parser: argparse.ArgumentParser):

    backbone_models = list(BACKBONE_REGISTRY.keys())
    parser.add_argument(
        "--backbone", type=str, choices=backbone_models, default=backbone_models[0]
    )
    parser.add_argument("--epochs", "-n", type=int, default=10)
    parser.add_argument("--warmup_epochs", "-w", type=int, default=3)
    parser.add_argument("--patience", "-p", type=int, default=3)
    parser.add_argument("--base_lr", "-lr", type=float, default=0.0001)
    parser.add_argument("--reload", "-r", action="store_true", default=False)
    parser.add_argument("--unfreeze", type=int, default=1)
    parser.add_argument("--experiment_name", "-name", type=str, default="cv_inat_v0.4")

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for pipeline reproducibility",
    )

    parser.add_argument(
        "--log_step_interval",
        type=int,
        default=10,
        help="Interval of steps for logging batch metrics to MLflow",
    )
    parser.add_argument(
        "--gated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Use gated attention pooling mechanism (default: True).
        Use --no-gated for simple attention.""",
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="Dropout probability for attention weights (default: 0.1)",
    )


def main():
    load_dotenv()
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_path = Path.cwd() / "log.log"
    logger = init_logger(log_path, logging.INFO)
    logger.info("Starting")
    # Set up paths

    # Set up environment specific configs
    config_path = resolve_env_config_path()
    with open(config_path, "r") as file:
        env_configs = yaml.safe_load(file)
    paths_params = PathsParams(**env_configs["paths"])
    dataloader_params = DataLoadersParams(**env_configs["dataloader_params"])
    hardware_profile = resolve_hardware_profile()
    configs = Config(
        config_path,
        paths_params=paths_params,
        dataloaders_params=dataloader_params,
        hardware_profile=hardware_profile,
        git_branch=get_current_git_branch(),
    )

    # Execute command
    if hasattr(args, "func"):
        exit_code = args.func(args, configs)
        sys.exit(exit_code)
    """
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f'Unexpected error {e}')
        logger.error(e)
    """


if __name__ == "__main__":
    main()

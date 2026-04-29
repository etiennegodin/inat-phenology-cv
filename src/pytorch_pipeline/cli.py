import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from torch import nn

from .status import status
from .train import execute
from .train.factory import (
    build_pipeline_dataloaders,
    build_pipeline_model,
    build_pipeline_optimizer,
    get_device,
)
from .utils import Config, clean_data, init_logger, update_dataset
from .utils.params import (
    PathsParams,
    TrainingParams,
)


def train_cmd(args, configs: Config):

    training_params = TrainingParams(
        epochs=args.epochs,
        patience=args.patience,
        reload=args.reload,
        test=args.test,
        learning_rate=args.learning_rate,
    )

    configs.training_params = training_params
    mlflow.set_tracking_uri(f"sqlite:///{configs.paths.ml_flow_db}")

    device = get_device()
    model = build_pipeline_model(configs.model_params, device)
    optimizer = build_pipeline_optimizer(model, configs.training_params)
    train_loader, val_loader, _ = build_pipeline_dataloaders(configs, model[0])
    criterion = nn.BCEWithLogitsLoss()

    mlflow.set_experiment("cv_inat")
    with mlflow.start_run(run_name="run") as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"\n{'=' * 60}")
        print(f"MLflow Run ID: {parent_run_id}")
        print(f"{'=' * 60}\n")

        mlflow.log_dict(configs.to_dict(), "configs.json")

        execute(
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_path=configs.paths.checkpoint_path,
            training_params=configs.training_params,
        )


def update_cmd(args, configs: Config):
    # Clean and update dataset
    clean_data(configs.paths.image_dir)
    update_dataset(configs.paths)


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
    train_parser.set_defaults(func=train_cmd)

    # Update command
    update_parser = subparsers.add_parser("update", help="Update source dataset")
    update_parser.set_defaults(func=update_cmd)

    # Status command
    status_parser = subparsers.add_parser("status", help="Pipeline status")
    status_parser.set_defaults(func=status_cmd)
    return parser


def add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument("--epochs", "-n", type=int, default=10)
    parser.add_argument("--patience", "-p", type=int, default=3)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
    parser.add_argument("--reload", "-r", action="store_true")
    parser.add_argument("--test", "-t", action="store_true")


def main():
    load_dotenv()
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_path = Path.cwd() / "log.log"
    logger = init_logger(log_path, logging.INFO)
    # Set up paths
    paths = PathsParams(root=os.environ.get("INAT_DATA_ROOT", ""))

    # Set up pipeline configs
    configs = Config(
        paths=paths,
    )

    # Execute command
    try:
        if hasattr(args, "func"):
            exit_code = args.func(args, configs)
            sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    main()

import argparse
import logging
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import pytorch
from torch import cuda, nn

from .status import status
from .train import execute
from .train.factory import (
    build_pipeline_dataloaders,
    build_pipeline_model,
    build_pipeline_optimizer,
    get_device,
)
from .train.peristence import load_checkpoint
from .utils import Config, clean_data, init_logger, update_dataset
from .utils.params import (
    DataLoadersParams,
    OptimizerParams,
    PathsParams,
    TrainingParams,
)


def train_cmd(args, configs: Config):

    mlflow.set_tracking_uri(f"sqlite:///{configs.paths.ml_flow_db}")

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

    # Intialize otpim params
    optim_params = OptimizerParams(
        backbone_lr=args.backbone_lr,
        attention_lr=args.attention_lr,
        head_lr=args.head_lr,
    )
    configs.optim_params = optim_params

    # Initialise train modules
    device = get_device()
    model = build_pipeline_model()
    optimizer = build_pipeline_optimizer(model, optim_params)
    train_loader, val_loader, _ = build_pipeline_dataloaders(configs, model)
    criterion = nn.BCEWithLogitsLoss()

    # Reinstate model and optimizer state if reload
    if args.reload:
        model, optimizer, start_epoch, eval_metrics, previous_run_id = load_checkpoint(
            configs.paths.checkpoint_path, model=model, optimizer=optimizer
        )
        best_loss = eval_metrics["val_loss"]
    else:
        start_epoch = None
        best_loss = 1e10
        previous_run_id = None

    training_params = TrainingParams(
        epochs=args.epochs,
        patience=args.patience,
        start_epoch=start_epoch,
        best_loss=best_loss,
    )

    # Set config params
    configs.training_params = training_params

    mlflow.set_experiment("cv_inat")
    with mlflow.start_run(run_id=previous_run_id) as parent_run:
        parent_run_id = parent_run.info.run_id
        print(f"\n{'=' * 60}")
        print(f"MLflow Run ID: {parent_run_id}")
        print(f"{'=' * 60}\n")

        mlflow.log_dict(configs.to_dict(), "configs.json")

        best_model = execute(
            device=device,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_path=configs.paths.checkpoint_path,
            training_params=configs.training_params,
        )

        # Log and register best model
        pytorch.log_model(best_model, "cv_inat")
        model_uri = f"runs:/{parent_run_id}/cv_inat"
        mlflow.register_model(model_uri, "cv_inat")


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
    parser.add_argument("--head-lr", "-hlr", type=float, default=0.001)
    parser.add_argument("--attention-lr", "-alr", type=float, default=0.001)
    parser.add_argument("--backbone-lr", "-blr", type=float, default=0.00001)
    parser.add_argument("--reload", "-r", action="store_true", default=False)
    parser.add_argument("--test", "-t", action="store_true", default=False)


def main():
    load_dotenv()
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_path = Path.cwd() / "log.log"
    logger = init_logger(log_path, logging.INFO)
    logger.info("Starting")
    # Set up paths
    paths = PathsParams(root=os.environ.get("INAT_DATA_ROOT", ""))

    # Set up pipeline configs
    configs = Config(
        paths=paths,
        test=args.test,
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

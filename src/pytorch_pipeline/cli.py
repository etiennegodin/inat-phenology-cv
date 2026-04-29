import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from torch import nn

from .config import (
    Config,
    MlFlowParams,
    PathsParams,
    TrainingParams,
    build_pipeline_dataloaders,
    build_pipeline_model,
    build_pipeline_optimizer,
    get_device,
)
from .status import status
from .train import train
from .utils import clean_data, init_logger, update_dataset


def train_cmd(args, configs: Config):

    training_params = TrainingParams(
        epochs=args.epochs, patience=args.patience, reload=args.reload
    )
    configs.training_params = training_params
    ml_flow = MlFlowParams("1")
    configs.ml_flow_params = ml_flow
    configs.test = args.test

    device = get_device()
    model = build_pipeline_model(configs.model_params, device)
    optimizer = build_pipeline_optimizer(model, configs.optimizer_params)
    train_loader, val_loader, _ = build_pipeline_dataloaders(configs, model[0])
    criterion = nn.BCEWithLogitsLoss()

    train(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_path=configs.paths.checkpoint_path,
        training_params=configs.training_params,
        mlflow_params=configs.ml_flow_params,
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
    logger.info("Start")
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


if __name__ == "__main__":
    main()

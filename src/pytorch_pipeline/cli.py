import argparse
import os
import sys

from .config import Config, PathsParams, TrainingParams
from .status import status
from .train import train
from .utils import clean_data, update_dataset


def train_cmd(args, configs: Config):

    training_params = TrainingParams(epochs=args.epochs, patience=args.patience)
    configs.training_params = training_params

    train(
        model=configs.model,
        device=configs.device,
        train_loader=configs.train_loader,
        val_loader=configs.val_loader,
        optimizer=configs.optimizer,
        criterion=configs.criterion,
        reload=args.reload,
        checkpoint_path=configs.paths.checkpoint_path,
        **configs.modules_params_to_dict("training_params"),
    )


def update_cmd(args, configs: Config):
    # Clean and update dataset
    clean_data(configs.paths.root)
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


def main():

    parser = create_parser()
    args = parser.parse_args()

    # Set up paths
    paths = PathsParams(
        root="/home/etienne/projects/inat-phenology-cv/data/photos",
        checkpoint_path="/home/etienne/projects/inat-phenology-cv/checkpoints/checkpoint.pth",
        db_path="/home/etienne/projects/inat-phenology-cv/data/cv_raw.duckdb",
        source_db_path="/home/etienne/projects/inatML/data/inat_raw.duckdb",
    )

    os.makedirs("checkpoints", exist_ok=True)

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
        print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        # logger.exception("Unexpected error")
        print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)

import argparse
import os

from .config import Config, PathsParams, TrainingParams
from .train import train


def main():

    parser = argparse.ArgumentParser(prog="torch_train")
    parser.add_argument("--epochs", "-n", type=int, default=10)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)
    parser.add_argument("--reload", "-r", action="store_true")

    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    training_params = TrainingParams(epochs=args.epochs, patience=3)
    paths = PathsParams(
        root="/home/etienne/projects/inat-phenology-cv/data/photos",
        checkpoint_path="/home/etienne/projects/inat-phenology-cv/checkpoints/checkpoint.pth",
    )
    configs = Config(
        paths=paths,
        training_params=training_params,
    )

    train(
        model=configs.model,
        train_loader=configs.train_loader,
        val_loader=configs.val_loader,
        optimizer=configs.optimizer,
        criterion=configs.criterion,
        reload=args.reload,
        checkpoint_path=configs.paths.checkpoint_path,
        **configs.modules_params_to_dict("training_params"),
    )

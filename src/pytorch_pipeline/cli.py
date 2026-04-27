import argparse
import os

from .config import Config
from .train import train


def main():

    parser = argparse.ArgumentParser(prog="torch_train")
    parser.add_argument("--epochs", "-n", type=int, default=10)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001)

    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    configs = Config(
        root="/home/etienne/projects/inat-phenology-cv/data/photos",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )

    train(
        model=configs.model,
        train_loader=configs.train_loader,
        val_loader=configs.val_loader,
        optimizer=configs.optimizer,
        criterion=configs.criterion,
        epochs=configs.epochs,
    )

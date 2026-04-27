import os

from .config import Config
from .train import train


def main():
    os.makedirs("checkpoints", exist_ok=True)
    configs = Config(
        root="/home/etienne/projects/inat-phenology-cv/data/photos", epochs=10
    )
    train(
        model=configs.model,
        train_loader=configs.train_loader,
        val_loader=configs.val_loader,
        optimizer=configs.optimizer,
        criterion=configs.criterion,
        epochs=configs.epochs,
    )

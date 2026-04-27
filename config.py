from dataclasses import dataclass, field

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from .src.dataset import build_datasets
from .src.model import build_model


@dataclass
class configs:
    root: str
    model: nn.Module = field(default_factory=build_model)
    criterion: nn.Module = nn.BCEWithLogitsLoss()
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 0.001
    momentum: float = 0.9
    optimizer_class: type = optim.SGD
    optimizer: optim.Optimizer = field(init=False)
    train_loader: DataLoader = field(init=False)
    val_loader: DataLoader = field(init=False)
    test_loader: DataLoader = field(init=False)

    def __post_init__(self):
        train_set, val_set, test_set = build_datasets(self.root)
        self.train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_set, batch_size=self.batch_size)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size)
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, momentum=self.momentum
        )

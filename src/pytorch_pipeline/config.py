from dataclasses import asdict, dataclass, field, is_dataclass

import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from .dataset import build_datasets
from .model import build_model


@dataclass
class TrainingParams:
    epochs: int = 3
    patience: int = 3


@dataclass
class OptimizerParams:
    learning_rate: float = 0.001


@dataclass
class ModelParams:
    head_inputs: int = 256
    dropout_prob: float = 0.5


@dataclass
class DataLoadersParams:
    batch_size: int = 16


@dataclass
class PathsParams:
    root: str
    checkpoint_path: str


@dataclass
class Config:
    paths: PathsParams
    training_params: TrainingParams = field(default_factory=TrainingParams)
    optimizer_params: OptimizerParams = field(default_factory=OptimizerParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    dataloaders_params: DataLoadersParams = field(default_factory=DataLoadersParams)
    model: nn.Module = field(init=False)
    criterion: nn.Module = nn.BCEWithLogitsLoss()
    optimizer_class: type = optim.Adam
    optimizer: optim.Optimizer = field(init=False)
    train_loader: DataLoader = field(init=False)
    val_loader: DataLoader = field(init=False)
    test_loader: DataLoader = field(init=False)

    def __post_init__(self):
        self._build_dataloaders()
        self.model = build_model(
            self.model_params.head_inputs, self.model_params.dropout_prob
        )
        self.optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.optimizer_params.learning_rate
        )

    def _build_dataloaders(self):
        train_set, val_set, test_set = build_datasets(
            self.paths.root, model_configs=self.model.default_cfg
        )
        self.train_loader = DataLoader(
            train_set, batch_size=self.dataloaders_params.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_set, batch_size=self.dataloaders_params.batch_size
        )
        self.test_loader = DataLoader(
            test_set, batch_size=self.dataloaders_params.batch_size
        )

    def to_dict(self):
        return asdict(self)

    def modules_params_to_dict(self, module_name: str) -> dict:
        module_obj = getattr(self, module_name, None)
        if module_obj is not None and is_dataclass(module_obj):
            return asdict(module_obj)
        return {}  # Or raise an error/return None

from dataclasses import asdict, dataclass, field, is_dataclass

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataloader import collate_fn
from .dataset import build_datasets
from .model import build_model, get_backbone
from .utils.params import (
    DataLoadersParams,
    MlFlowParams,
    ModelParams,
    OptimizerParams,
    PathsParams,
    SamplesParams,
    TrainingParams,
)


@dataclass
class Config:
    paths: PathsParams
    ml_flow_params: MlFlowParams = field(default_factory=MlFlowParams)
    samples_params: SamplesParams = field(default_factory=SamplesParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    optimizer_params: OptimizerParams = field(default_factory=OptimizerParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    dataloaders_params: DataLoadersParams = field(default_factory=DataLoadersParams)
    model: nn.Sequential = field(init=False)
    backbone: nn.Module = field(init=False)
    criterion: nn.Module = nn.BCEWithLogitsLoss()
    optimizer_class: type = optim.Adam
    optimizer: optim.Optimizer = field(init=False)
    train_loader: DataLoader = field(init=False)
    val_loader: DataLoader = field(init=False)
    test_loader: DataLoader = field(init=False)
    device: torch.device = field(init=False)
    test: bool = False

    def __post_init__(self):
        self.backbone = get_backbone()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            self.model_params.head_inputs, self.model_params.dropout_prob
        ).to(self.device)
        self.optimizer = self.optimizer_class(
            [
                {
                    "params": [
                        p for p in self.model[0].parameters() if p.requires_grad
                    ],
                    "lr": 1e-4,
                },
                {
                    "params": self.model[1].parameters(),
                    "lr": self.optimizer_params.learning_rate,
                },
            ]
        )
        self._build_dataloaders()

    def _build_dataloaders(self):
        train_set, val_set, test_set = build_datasets(
            paths=self.paths,
            samples_params=self.samples_params,
            model_configs=self.backbone.default_cfg,
        )

        if self.test:
            train_set = Subset(train_set, range(500))

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.dataloaders_params.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.dataloaders_params.batch_size,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.dataloaders_params.batch_size,
            collate_fn=collate_fn,
        )

    def to_dict(self):
        return asdict(self)

    def modules_params_to_dict(self, module_name: str) -> dict:
        module_obj = getattr(self, module_name, None)
        if module_obj is not None and is_dataclass(module_obj):
            return asdict(module_obj)
        return {}  # Or raise an error/return None

from dataclasses import asdict, dataclass, field

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataloader import collate_fn
from .dataset import build_datasets
from .model import build_model
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
    test: bool = False

    def to_dict(self):
        return asdict(self)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pipeline_model(params: ModelParams, device: torch.device) -> nn.Sequential:
    model = build_model(params.head_inputs, params.dropout_prob).to(device)
    return model


def build_pipeline_optimizer(
    model: nn.Sequential, params: OptimizerParams
) -> optim.Optimizer:
    return optim.Adam(
        [
            {
                "params": [p for p in model[0].parameters() if p.requires_grad],
                "lr": 1e-4,
            },
            {
                "params": model[1].parameters(),
                "lr": params.learning_rate,
            },
        ]
    )


def build_pipeline_dataloaders(
    config: Config, backbone: nn.Module
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = build_datasets(
        paths=config.paths,
        samples_params=config.samples_params,
        model_configs=backbone.default_cfg,
    )

    if config.test:
        train_set = Subset(train_set, range(min(500, len(train_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=config.dataloaders_params.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.dataloaders_params.batch_size,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.dataloaders_params.batch_size,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader

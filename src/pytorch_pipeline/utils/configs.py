from dataclasses import asdict, dataclass, field
from pathlib import Path

from .params import (
    DataLoadersParams,
    DatasetParams,
    ModelParams,
    OptimizerParams,
    PathsParams,
    SchedulerParams,
    TrainingParams,
)


@dataclass
class Config:
    config_path: Path
    paths_params: PathsParams
    dataloaders_params: DataLoadersParams
    dataset_params: DatasetParams = field(default_factory=DatasetParams)
    model_params: ModelParams = field(init=False)
    training_params: TrainingParams = field(init=False)
    optim_params: OptimizerParams = field(init=False)
    scheduler_params: SchedulerParams = field(init=False)
    cuda: bool = False
    test: bool = False

    def to_dict(self):
        return asdict(self)

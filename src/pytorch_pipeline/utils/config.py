from dataclasses import asdict, dataclass, field

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
    paths: PathsParams
    dataset_params: DatasetParams = field(default_factory=DatasetParams)
    model_params: ModelParams = field(init=False)
    training_params: TrainingParams = field(init=False)
    dataloaders_params: DataLoadersParams = field(init=False)
    optim_params: OptimizerParams = field(init=False)
    scheduler_params: SchedulerParams = field(init=False)
    cuda: bool = False
    test: bool = False

    def to_dict(self):
        return asdict(self)

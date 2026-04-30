from dataclasses import asdict, dataclass, field

from .params import (
    DataLoadersParams,
    ModelParams,
    OptimizerParams,
    PathsParams,
    SamplesParams,
    TrainingParams,
)


@dataclass
class Config:
    paths: PathsParams
    samples_params: SamplesParams = field(default_factory=SamplesParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    training_params: TrainingParams = field(init=False)
    dataloaders_params: DataLoadersParams = field(init=False)
    optim_params: OptimizerParams = field(init=False)
    test: bool = False

    def to_dict(self):
        return asdict(self)

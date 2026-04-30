from dataclasses import asdict, dataclass, field

from .params import (
    DataLoadersParams,
    ModelParams,
    PathsParams,
    SamplesParams,
    TrainingParams,
)


@dataclass
class Config:
    paths: PathsParams
    training_params: TrainingParams = field(init=False)
    samples_params: SamplesParams = field(default_factory=SamplesParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    dataloaders_params: DataLoadersParams = field(init=False)

    def to_dict(self):
        return asdict(self)

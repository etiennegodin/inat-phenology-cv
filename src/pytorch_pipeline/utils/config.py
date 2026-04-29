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
    samples_params: SamplesParams = field(default_factory=SamplesParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    model_params: ModelParams = field(default_factory=ModelParams)
    dataloaders_params: DataLoadersParams = field(default_factory=DataLoadersParams)

    def to_dict(self):
        return asdict(self)

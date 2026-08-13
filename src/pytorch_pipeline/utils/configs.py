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
from .system import HardwareProfile

LABEL_MAPPING = {
    0: "Flowering",
    1: "Fruiting",
    2: "Flower Budding",
}

CLASS_ORDER = ["Flowering", "Fruiting", "Flower Budding"]


@dataclass
class Config:
    config_path: Path
    paths_params: PathsParams
    dataloaders_params: DataLoadersParams
    hardware_profile: HardwareProfile
    dataset_params: DatasetParams = field(default_factory=DatasetParams)
    model_params: ModelParams = field(init=False)
    training_params: TrainingParams = field(init=False)
    optim_params: OptimizerParams = field(init=False)
    scheduler_params: SchedulerParams = field(init=False)
    cuda: bool = False
    test: bool = False
    max_img_resolution: int = 500

    def to_dict(self):
        return asdict(self)

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
    2: "Flower_Budding",
}

CLASS_ORDER = ["Flowering", "Fruiting", "Flower_Budding"]


@dataclass
class ClassesPatienceCondition:
    class_count: int = 3
    best_metrics: list[float] = field(init=False)
    staleness: list[int] = field(init=False)

    def __post_init__(self):
        self.best_metrics = [0.0 for _ in range(self.class_count)]
        self.staleness = [0 for _ in range(self.class_count)]

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    config_path: Path
    paths_params: PathsParams
    dataloaders_params: DataLoadersParams
    hardware_profile: HardwareProfile
    git_branch: str
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

import os
from dataclasses import dataclass, field


@dataclass
class MlFlowParams:
    run_name: str = "temp"
    experiment_name: str = "cv_inat"


@dataclass
class TrainingParams:
    epochs: int = 3
    patience: int = 3
    reload: bool = False


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
    checkpoint_path: str = field(init=False)
    db_path: str = field(init=False)
    # source_db_path: str = field(init=False)
    image_dir: str = field(init=False)

    def __post_init__(self):

        self.image_dir = os.environ.get(
            "INAT_IMAGE_DIR", os.path.join(self.root, "images")
        )
        self.db_path = os.path.join(self.root, "cv_raw.duckdb")
        self.checkpoint_path = os.path.join(self.root, "checkpoints/checkpoint.pth")


@dataclass
class SamplesParams:
    obs_id: str = "observation_id"
    photo_id_col: str = "photo_id"
    label_id: str = "controlled_value_id"

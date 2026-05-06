import os
from dataclasses import dataclass, field


@dataclass
class TrainingParams:
    epochs: int
    patience: int
    start_epoch: int | None
    best_loss: float = 1e10


@dataclass
class OptimizerParams:
    backbone_lr: float
    attention_lr: float
    head_lr: float


@dataclass
class ModelParams:
    head_inputs: int = 256
    dropout_prob: float = 0.5


@dataclass
class DataLoadersParams:
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclass
class PathsParams:
    root: str
    checkpoint_path: str = field(init=False)
    db_path: str = field(init=False)
    ml_flow_db: str = field(init=False)
    # source_db_path: str = field(init=False)
    image_dir: str = field(init=False)

    def __post_init__(self):

        self.image_dir = os.environ.get(
            "INAT_IMAGE_DIR", os.path.join(self.root, "images")
        )
        self.db_path = os.path.join(self.root, "cv_raw.duckdb")
        self.checkpoint_path = os.path.join(self.root, "checkpoints/checkpoint.pth")
        self.ml_flow_db = os.path.join(self.root, "mlflow.db")


@dataclass
class SamplesParams:
    obs_id: str = "observation_id"
    photo_id_col: str = "photo_id"
    label_id: str = "controlled_value_id"

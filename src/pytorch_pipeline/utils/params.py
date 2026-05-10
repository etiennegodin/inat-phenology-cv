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
    backbone_lr: float = 1e-3
    attention_lr: float = 1e-3
    head_lr: float = 1e-3


@dataclass
class ModelParams:
    head_neurons: int
    head_outputs: int
    head_dropout_prob: float
    attention_dim: int
    attention_neurons: int


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
    source_db_path: str = "/home/etienne/projects/inatML/data/inat_raw.duckdb"
    image_dir: str = field(init=False)

    def __post_init__(self):

        self.image_dir = os.environ.get(
            "INAT_IMAGE_DIR", os.path.join(self.root, "images")
        )
        self.db_path = os.path.join(self.root, "cv_raw.duckdb")
        self.checkpoint_path = os.path.join(self.root, "checkpoints/checkpoint.pth")
        self.ml_flow_db = os.path.join(self.root, "mlflow.db")


@dataclass
class DatasetParams:
    size: int | None = None
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    idx_col: str = "observation_id"
    photo_idx_col: str = "photo_id"
    label_col: str = "label"

    def __post_init__(self):
        assert (self.train_frac + self.val_frac + self.test_frac) == 1

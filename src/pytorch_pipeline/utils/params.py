from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainingParams:
    epochs: int
    patience: int
    start_epoch: int | None
    best_objective: float
    seed: int = 42
    log_step_interval: int = 10
    pos_ratios: list[float] = field(default_factory=list[float])

    def to_dict(self):
        return asdict(self)


@dataclass
class SchedulerParams:
    warmup_epochs: int = 3
    total_epoch: int = 10

    def to_dict(self):
        return asdict(self)


@dataclass
class OptimizerParams:
    base_lr: float = 1e-3
    backbone_lr: float = field(init=False)
    attention_lr: float = field(init=False)
    head_lr: float = field(init=False)

    def __post_init__(self):
        self.backbone_lr = self.base_lr / 10
        self.attention_lr = self.base_lr
        self.head_lr = self.base_lr

    def to_dict(self):
        return asdict(self)


@dataclass
class ModelParams:
    backbone: str
    head_neurons: int
    head_outputs: int
    head_dropout_prob: float
    attention_neurons: int
    attention_dropout_prob: float
    last_blocks: int

    def to_dict(self):
        return asdict(self)


@dataclass
class DataLoadersParams:
    batch_size: int
    max_images: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    use_max_images: bool

    def to_dict(self):
        return asdict(self)


@dataclass
class PathsParams:
    root: str
    data_root: str
    image_dir: str
    db_path: str
    checkpoint_path: str
    source_db_path: str = "/home/etienne/projects/inatML/data/inat_raw.duckdb"

    def __post_init__(self):
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return asdict(self)


@dataclass
class DatasetParams:
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    idx_col: str = "observation_id"
    photo_idx_col: str = "photo_id"
    label_col: str = "label"
    testing_frac: float = 0.33

    def __post_init__(self):
        assert (self.train_frac + self.val_frac + self.test_frac) == 1

    def to_dict(self):
        return asdict(self)

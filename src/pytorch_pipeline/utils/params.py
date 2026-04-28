from dataclasses import dataclass


@dataclass
class TrainingParams:
    epochs: int = 3
    patience: int = 3


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
    checkpoint_path: str
    db_path: str
    source_db_path: str


@dataclass
class SamplesParams:
    observations_col: str = "observation_id"
    photo_id_col: str = "photo_id"
    label_col: str = "controlled_value_id"

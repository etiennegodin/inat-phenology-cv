from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class InferenceParams:
    model_name: str
    model_version: int
    db_path: str
    photo_target_dir: str
    sql_dir: str


@dataclass
class IngestPhotosParams:
    extensions: list[str] = field(default_factory=list[str])
    size: str = "medium"
    item_id: str = "photo_id"
    label: str = field(init=False)

    def __post_init__(self):
        if self.extensions == []:
            self.extensions = [".jpg", ".jpeg", ".png"]
        pass


@dataclass
class FineTuneParams:
    n_trials: int = 20
    timeout: int = 1000
    sampler: str = "tpe"
    pruner: str = "median"
    study_name: str | None = None
    subsample_frac: float = 1.0
    search_space: str = "coarse"
    resume: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class CleanlabParams:
    n_folds: int = 5
    start_fold: int = 1
    table_name: str = "oof_predictions"
    export_csv: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class ValParams:
    model_name: str = "cv_inat"
    model_version: int = 1
    device: str = "cuda"
    eval_db_path: str | None = None
    no_db: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class PredictParams:
    model_name: str = "cv_pheno_bioclip"
    model_version: int = 1
    device: str = "cuda"
    urls: list[str] | None = None
    ids: list[str] | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ProgressiveUnfreezingConfig:
    enabled: bool = False
    patience: int = 3
    cooldown: int = 3
    max_stages: int = 3
    blocks_per_stage: int = 1
    backbone_decay: float = 0.9


@dataclass
class TrainConfig:
    """Hydra-facing general training settings."""

    epochs: int = 10
    stopping_patience: int = 3
    best_objective: float = 1e-5
    log_step_interval: int = 10
    base_lr: float = 0.0001
    warmup_epochs: int = 3
    progressive_unfreezing: ProgressiveUnfreezingConfig = field(
        default_factory=ProgressiveUnfreezingConfig
    )

    def to_dict(self):
        return asdict(self)


@dataclass
class DinoV3Params:
    checkpoint_path: str | None = None


@dataclass
class ModelConfig:
    """Hydra-facing model section; converted to ModelParams at runtime."""

    backbone: str = "bioclip"
    architecture: str = "mil_pooling"
    start_unfreezed: int = 1
    dinov3: DinoV3Params = field(default_factory=DinoV3Params)
    head_neurons: int = 256
    head_outputs: int = 1
    head_dropout_prob: float = 0.5
    attention_neurons: int = 128
    attention_dropout_prob: float = 0.1
    gated: bool = True
    proj_dim: int = 256
    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.1
    max_images_per_obs: int = 64

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingParams:
    epochs: int
    stopping_patience: int
    unfreeze: bool
    unfreezing_patience: int
    unfreezing_cooldown: int
    starting_block: int
    max_stages: int
    block_per_stage: int
    best_objective: float
    accumulation_steps: int = 1
    backbone_decay: float = 0.9
    seed: int = 42
    log_step_interval: int = 10
    pos_ratios: list[float] = field(default_factory=list[float])

    def get_depth_ratio(self, block_depth: int):
        return self.backbone_decay ** (block_depth + 1)

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
    """Model hyperparameters.

    Fields are grouped by the architecture they apply to.  Passing
    architecture-irrelevant fields is silently allowed (they are simply
    ignored by the model), but a warning is emitted when non-default
    values are detected for the wrong architecture.

    Shared fields
    -------------
    backbone            : Backbone registry key (e.g. ``"bioclip"``,
                          ``"dinov3_vits14"``).
    architecture        : Aggregation head: ``"mil_pooling"`` (default) or
                          ``"cross_attention"``.
    start_unfreezed     : Number of backbone blocks unfrozen at initialisation.
    dinov3               : DINOv3-specific settings, including its optional local
                          pretrained checkpoint path.

    mil_pooling fields (MilAttentionModel)
    --------------------------------------
    head_neurons        : Hidden size of each classifier head MLP.
    head_outputs        : Output size of each classifier head (typically 1).
    head_dropout_prob   : Dropout inside the classifier head MLP.
    attention_neurons   : Hidden size of the attention pooling layer.
    attention_dropout_prob: Dropout applied to softmax attention weights.
    gated               : Use gated attention pooling (True) vs simple (False).

    cross_attention fields (CrossAttentionModel)
    --------------------------------------------
    proj_dim            : Projection dimension of the MLP neck + cross-attention.
    cross_attn_heads    : Number of heads in the cross-attention block.
    cross_attn_dropout  : Dropout used in both the MLP neck and attention block.
    max_images_per_obs  : Size of the 1-D image-index embedding table.
    """

    # -- Shared ----------------------------------------------------------------
    backbone: str
    architecture: str = "mil_pooling"
    start_unfreezed: int = 1
    dinov3: DinoV3Params = field(default_factory=DinoV3Params)

    # -- mil_pooling (MilAttentionModel) ---------------------------------------
    head_neurons: int = 256
    head_outputs: int = 1
    head_dropout_prob: float = 0.5
    attention_neurons: int = 128
    attention_dropout_prob: float = 0.1
    gated: bool = True

    # -- cross_attention (CrossAttentionModel) ---------------------------------
    proj_dim: int = 256
    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.1
    max_images_per_obs: int = 64

    # Defaults used for cross-validation below
    _MIL_DEFAULTS: dict = field(
        default_factory=lambda: {
            "head_neurons": 256,
            "head_outputs": 1,
            "head_dropout_prob": 0.5,
            "attention_neurons": 128,
            "attention_dropout_prob": 0.1,
            "gated": True,
        },
        repr=False,
        compare=False,
    )
    _CROSS_DEFAULTS: dict = field(
        default_factory=lambda: {
            "proj_dim": 256,
            "cross_attn_heads": 8,
            "cross_attn_dropout": 0.1,
            "max_images_per_obs": 64,
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        import logging as _logging

        _log = _logging.getLogger(__name__)

        if self.architecture == "mil_pooling":
            for field_name, default in self._CROSS_DEFAULTS.items():
                if getattr(self, field_name) != default:
                    _log.warning(
                        "ModelParams: field '%s' is cross_attention-specific but was "
                        "set to a non-default value (%r) while "
                        "architecture='mil_pooling'. "
                        "It will be ignored by MilAttentionModel.",
                        field_name,
                        getattr(self, field_name),
                    )
        elif self.architecture == "cross_attention":
            for field_name, default in self._MIL_DEFAULTS.items():
                if getattr(self, field_name) != default:
                    _log.warning(
                        "ModelParams: field '%s' is mil_pooling-specific but was "
                        "set to a non-default value (%r) "
                        "while architecture='cross_attention'. "
                        "It will be ignored by CrossAttentionModel.",
                        field_name,
                        getattr(self, field_name),
                    )

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
    gradient_accumulation_steps: int = 1

    def to_dict(self):
        return asdict(self)


@dataclass
class PathsParams:
    root: str
    data_root: str
    image_dir: str
    db_path: str
    checkpoint_path: str
    eval_db_path: str
    inference_db_path: str
    sql_dir: str
    photo_target_dir: str
    optuna_storage: str
    cleanlab_db_path: str
    corrections_dir: str

    def __post_init__(self):
        Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        return asdict(self)


@dataclass
class DatasetParams:
    source_table: str = "cv_photos3"
    idx_col: str = "observation_id"
    photo_idx_col: str = "photo_id"
    label_col: str = "label"
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    testing_frac: float = 1.0

    def __post_init__(self):
        assert (self.train_frac + self.val_frac + self.test_frac) == 1

    def to_dict(self):
        return asdict(self)

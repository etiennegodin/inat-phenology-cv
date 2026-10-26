from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .loader import Config
from .params import (
    CachingConfig,
    CachingParams,
    CleanlabParams,
    DataLoadersParams,
    DatasetParams,
    DinoV3Params,
    FineTuneParams,
    ModelConfig,
    ModelParams,
    OptimizerParams,
    PathsParams,
    PredictParams,
    SchedulerParams,
    TrainConfig,
    TrainingParams,
    ValParams,
)
from .system import resolve_hardware_profile

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


@dataclass
class AppConfig:
    """Root Hydra schema. Machine paths come from the ``env`` config group."""

    command: str = "train"
    env_name: str = MISSING
    seed: int = 42
    test: bool = False
    test_frac: float = 1.0
    experiment_name: str = "cv_inat_v0.4"
    paths: PathsParams = MISSING
    dataloaders: DataLoadersParams = MISSING
    dataset: DatasetParams = field(default_factory=DatasetParams)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cache: CachingConfig = field(default_factory=CachingConfig)
    tune: FineTuneParams = field(default_factory=FineTuneParams)
    val: ValParams = field(default_factory=ValParams)
    predict: PredictParams = field(default_factory=PredictParams)
    cleanlab: CleanlabParams = field(default_factory=CleanlabParams)


def register_config_store() -> ConfigStore:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=AppConfig)
    return cs


register_config_store()


def resolved_container(cfg: DictConfig | AppConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def cfg_to_runtime(cfg: DictConfig | AppConfig) -> Config:
    """Creates the project Config with paths, dataloaders,
    dataset settings, hardware profile, and git metadata"""
    from ..utils.misc import get_current_git_branch, get_git_hash

    data = resolved_container(cfg)
    dataset_kwargs = dict(data["dataset"])
    dataset_kwargs["testing_frac"] = data.get(
        "test_frac", dataset_kwargs.get("testing_frac", 1.0)
    )
    return Config(
        config_path=CONFIG_DIR / "config.yaml",
        paths_params=PathsParams(**data["paths"]),
        dataloaders_params=DataLoadersParams(**data["dataloaders"]),
        dataset_params=DatasetParams(**dataset_kwargs),
        hardware_profile=resolve_hardware_profile(),
        git_branch=get_current_git_branch(),
        git_hash=get_git_hash(),
        test=bool(data.get("test", False)),
    )


def _init_fields(cls) -> set[str]:
    return {f.name for f in fields(cls) if f.init and not f.name.startswith("_")}


def cache_params_from_cfg(
    cache_cfg: DictConfig | CachingConfig | dict[str, Any],
    **overrides: Any,
) -> CachingParams:
    """Creates CachingParams from DictConfig"""
    data = OmegaConf.to_container(cache_cfg, resolve=True)  # type: ignore[arg-type]
    if not isinstance(data, dict):
        raise TypeError("cache config must resolve to a mapping")
    data.update({k: v for k, v in overrides.items() if v is not None})
    allowed = _init_fields(CachingParams)
    return CachingParams(**{k: data[k] for k in allowed if k in data})


def model_params_from_cfg(
    model_cfg: DictConfig | ModelConfig | dict[str, Any],
    **overrides: Any,
) -> ModelParams:
    """Creates ModelParams from DictConfig"""
    data = OmegaConf.to_container(model_cfg, resolve=True)  # type: ignore[arg-type]
    if not isinstance(data, dict):
        raise TypeError("model config must resolve to a mapping")
    data.update({k: v for k, v in overrides.items() if v is not None})
    dinov3_data = data.get("dinov3")
    if isinstance(dinov3_data, dict):
        data["dinov3"] = DinoV3Params(**dinov3_data)
    allowed = _init_fields(ModelParams)
    return ModelParams(**{k: data[k] for k in allowed if k in data})


def training_params_from_cfg(
    cfg: DictConfig | AppConfig,
    **overrides: Any,
) -> TrainingParams:
    """Creates TrainingParams, adding shared values such as
    the seed and dataloader accumulation setting."""
    data = resolved_container(cfg)
    train = dict(data["train"])
    model = data["model"]
    progressive = train["progressive_unfreezing"]
    allowed = _init_fields(TrainingParams)
    kwargs: dict[str, Any] = {k: train[k] for k in allowed if k in train}
    kwargs["unfreeze"] = progressive["enabled"]
    kwargs["unfreezing_patience"] = progressive["patience"]
    kwargs["unfreezing_cooldown"] = progressive["cooldown"]
    kwargs["max_stages"] = progressive["max_stages"]
    kwargs["block_per_stage"] = progressive["blocks_per_stage"]
    kwargs["backbone_decay"] = progressive["backbone_decay"]
    kwargs.setdefault("seed", data["seed"])
    kwargs.setdefault(
        "starting_block", model.get("start_unfreezed", train.get("starting_block", 1))
    )
    kwargs.setdefault(
        "accumulation_steps",
        data["dataloaders"].get("gradient_accumulation_steps", 1),
    )
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return TrainingParams(**{k: kwargs[k] for k in allowed if k in kwargs})


def optimizer_params_from_cfg(
    cfg: DictConfig | AppConfig, **overrides: Any
) -> OptimizerParams:
    data = resolved_container(cfg)
    base_lr = overrides.get("base_lr", data["train"]["base_lr"])
    return OptimizerParams(base_lr=base_lr)


def scheduler_params_from_cfg(
    cfg: DictConfig | AppConfig, **overrides: Any
) -> SchedulerParams:
    data = resolved_container(cfg)
    warmup = overrides.get("warmup_epochs", data["train"]["warmup_epochs"])
    total = overrides.get("total_epoch", data["train"]["epochs"])
    return SchedulerParams(warmup_epochs=warmup, total_epoch=total)

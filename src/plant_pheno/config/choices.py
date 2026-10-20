from __future__ import annotations

from omegaconf import DictConfig

from .hydra_conf import CONFIG_DIR
from .registry import OPTUNA_PRUNERS, OPTUNA_SAMPLERS
from .search_spaces import SEARCH_SPACES


def _group_choices(group: str) -> list[str]:
    group_dir = CONFIG_DIR / group
    return sorted(path.stem for path in group_dir.glob("*.yaml"))


def available_choices() -> dict[str, list[str]]:
    from ..core.backbone import BACKBONE_REGISTRY

    return {
        "command": _group_choices("command"),
        "env": _group_choices("env"),
        "architecture": _group_choices("architecture"),
        "experiment": _group_choices("experiment"),
        "model.backbone": sorted(BACKBONE_REGISTRY),
        "tune.search_space": sorted(SEARCH_SPACES),
        "tune.sampler": sorted(OPTUNA_SAMPLERS),
        "tune.pruner": sorted(OPTUNA_PRUNERS),
    }


def validate_config_choices(cfg: DictConfig) -> None:
    choices = available_choices()
    selected = {
        "env": cfg.env_name,
        "architecture": cfg.model.architecture,
        "model.backbone": cfg.model.backbone,
    }
    if cfg.command == "tune":
        selected.update(
            {
                "tune.search_space": cfg.tune.search_space,
                "tune.sampler": cfg.tune.sampler,
                "tune.pruner": cfg.tune.pruner,
            }
        )

    invalid = {
        name: (value, choices[name])
        for name, value in selected.items()
        if value not in choices[name]
    }
    if invalid:
        details = "; ".join(
            f"{name}={value!r} (choose from: {', '.join(options)})"
            for name, (value, options) in invalid.items()
        )
        raise ValueError(f"Invalid configuration choice: {details}")


def format_available_choices() -> str:
    return "\n".join(
        f"{name}: {', '.join(values) if values else '(none configured)'}"
        for name, values in available_choices().items()
    )

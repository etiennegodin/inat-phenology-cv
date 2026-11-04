from .objective import make_objective

__all__ = ["make_objective"]


def resolve_study_name(cfg, run_id):
    if cfg.tune.study_name is not None:
        return cfg.tune.study_name

    return f"""{cfg.model.backbone}_{cfg.model.architecture}_{run_id[:8]}
{"_test" if cfg.test else ""}"""

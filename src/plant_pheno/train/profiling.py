from dataclasses import asdict as dc_asdict
from dataclasses import dataclass, field

import mlflow


def asdict_exclude(obj, exclude_fields=None):
    """Convert dataclass to dict, excluding specified fields."""
    if exclude_fields is None:
        exclude_fields = set()

    data = dc_asdict(obj)
    return {k: v for k, v in data.items() if k not in exclude_fields}


@dataclass
class Profile:
    def total(self):
        t = 0.0
        for v in self.to_dict().values():
            if isinstance(v, float):
                t += v
                continue
            elif isinstance(v, dict):
                t1 = 0.0
                for v1 in v.values():
                    t1 += v1
                t += t1
                continue
        return t

    def to_dict(self):
        return dc_asdict(self)


@dataclass
class MetricsProfiling(Profile):
    compute_metrics_time: float
    log_metrics_time: float
    log_attention_metrics_time: float
    error_analysis_time: float


@dataclass
class StepProfiling(Profile):
    data_time: float
    compute_time: float


@dataclass
class EpochProfiling(Profile):
    # Train
    train: StepProfiling
    epoch_scheduler_time: float
    eval: StepProfiling
    metrics: MetricsProfiling

    # Total
    epoch_duration: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self):
        self.epoch_duration = self.total()

    def to_dict(self):
        return asdict_exclude(self, exclude_fields={"epoch_duration"})

    def log_to_mlflow(self, epoch: int):
        if mlflow.active_run():
            for k, v in self.to_dict().items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"profiling/{k}", float(v), step=epoch)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        mlflow.log_metric(
                            f"profiling/{k}/{sub_k}", float(sub_v), step=epoch
                        )

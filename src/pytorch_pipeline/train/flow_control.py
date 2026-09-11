import logging
from dataclasses import asdict, dataclass, field

from ..utils.configs import CLASS_ORDER
from ..utils.params import TrainingParams

logger = logging.getLogger(__name__)


@dataclass
class ClassesObjectiveState:
    class_count: int = 3
    best_metrics: list[float] = field(init=False)
    staleness: list[int] = field(init=False)

    def __post_init__(self):
        self.best_metrics = [0.0 for _ in range(self.class_count)]
        self.staleness = [0 for _ in range(self.class_count)]

    def log_state_update(self, i: int) -> None:
        logger.info(
            f"Pr_norm_excess_{CLASS_ORDER[i]} improved to {self.best_metrics[i]:.5f}. "
        )

    def reset(self):
        self.staleness = [0 for _ in self.staleness]

    def to_dict(self):
        return asdict(self)


@dataclass
class StageUnfreezeState:
    local_warmup_len: int
    unlocked_epoch: int = field(init=False)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingState:
    classes_states: ClassesObjectiveState
    training_params: TrainingParams
    unlocked_stage_count: int = 0
    stages_states: list[StageUnfreezeState] = field(init=False)

    def __post_init__(self):
        self.stages_states = [
            StageUnfreezeState(
                local_warmup_len=self.training_params.unfreezing_cooldown
            )
            for _ in range(self.training_params.max_stages)
        ]

    def to_dict(self):
        return asdict(self)

    def _get_latest_stage_state(self) -> StageUnfreezeState:
        return self.stages_states[self.unlocked_stage_count]

    def patience_counter(
        self,
        classes_metric: list[float],
        min_delta: float = 0.003,
    ) -> None:
        """Worst class still improving patience.
        Compares each class newest metric to previous best and updates staleness.

        Args:
            classes_metric (list[float]): Current epoch metrics for each class
            classes_states (tuple[list[float],list[int]]):
            (best metric for this run, class staleness)

        Returns:
            tuple[list[float],list[int]]:
            (best metric for this run, updated class staleness)
        """

        # todo per class min_delta
        """
        min_delta implicitly assumes similar noise floors across three "
        classes that don't have similar sample sizes
        """
        for i, m in enumerate(classes_metric):
            if m > self.classes_states.best_metrics[i] + min_delta:
                self.classes_states.best_metrics[i] = m
                self.classes_states.staleness[i] = 0
                self.classes_states.log_state_update(i)
            else:
                self.classes_states.staleness[i] += 1

    def stop_condition(self) -> bool:
        """Evaluate staleness of each class and returns wether to stop

        Args:
            classes_patience (list[int]): stale_count for each class
            stop_patience (int): stop_patience value for this run

        Returns:
            bool: Stop run boolean
        """
        return (
            min(self.classes_states.staleness) >= self.training_params.stopping_patience
        )

    def unfreeze_condition(self, stale_class_ratio: float = 0.66) -> bool:
        return (
            sum(
                [
                    c >= self.training_params.unfreezing_patience
                    for c in self.classes_states.staleness
                ]
            )
            / len(self.classes_states.staleness)
            >= stale_class_ratio
        )

    def checkpoint_condition(self) -> bool:
        return min(self.classes_states.staleness) == 0

    def cooldown_condition(self, epoch: int):
        current_stage = self._get_latest_stage_state()
        return (
            epoch - current_stage.unlocked_epoch
            <= self.training_params.unfreezing_cooldown
        )

    def max_stages_condition(self):
        return self.unlocked_stage_count >= self.training_params.max_stages

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from ..utils.configs import CLASS_ORDER
from ..utils.params import TrainingParams

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_stage_states(training_params: TrainingParams, trainable_block_count: int):
    block_per_stage = training_params.block_per_stage
    stage_count = training_params.max_stages
    stages = []

    blocks = list(range(0, trainable_block_count))
    blocks.reverse()

    for i in range(0, len(blocks), block_per_stage):
        stages.append(
            StageUnfreezeState(
                name=f"unfreezed_stage_{i}",
                local_warmup_len=training_params.unfreezing_cooldown,
                blocks=blocks[i : i + block_per_stage],
            )
        )

    return stages[:stage_count]


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
    name: str
    local_warmup_len: int
    blocks: list[int]
    unlocked: bool = False
    unlocked_epoch: int = 0

    def get_warmup_lr(
        self, epoch: int, start_factor: float = 0.1, end_factor: float = 1.0
    ) -> float:
        """Simple linear interpolation for warmup lr factor

        Args:
            epoch (int): Epoch
            start_factor (float, optional): Warmup starting factor. Defaults to 0.1.
            end_factor (float, optional): Warmup ending factor. Defaults to 1.0.

        Returns:
            float: Warmup factor
        """
        x = epoch - self.unlocked_epoch
        if self.local_warmup_len <= 1:
            return end_factor
        progress = min(1.0, max(0.0, x / (self.local_warmup_len - 1)))
        return start_factor + progress * (end_factor - start_factor)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingState:
    classes_states: ClassesObjectiveState
    training_params: TrainingParams
    stages_states: list[StageUnfreezeState] = field(default_factory=list)

    @property
    def unlocked_stages(self) -> list[StageUnfreezeState]:
        return [s for s in self.stages_states if s.unlocked]

    @property
    def unlocked_stage_count(self) -> int:
        return len(self.unlocked_stages)

    def get_latest_stage_state(self) -> StageUnfreezeState | None:
        unlocked = self.unlocked_stages
        return unlocked[-1] if unlocked else None

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

    def cooldown_condition(self, stage_state: StageUnfreezeState, epoch: int):
        return (
            epoch - stage_state.unlocked_epoch
            < self.training_params.unfreezing_cooldown
        )

    def max_stages_condition(self) -> bool:
        return self.unlocked_stage_count >= min(
            self.training_params.max_stages, len(self.stages_states)
        )

    def to_dict(self):
        d = asdict(self)
        d["unlocked_stage_count"] = self.unlocked_stage_count
        return d

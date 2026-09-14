from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from ..utils.configs import CLASS_ORDER
from ..utils.params import TrainingParams

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from .model import PhenologyModel

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
        logger.debug(f"Reseting class states: {self.staleness}")

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

    def unfreeze(
        self, epoch: int, model: PhenologyModel, optimizer: Optimizer, last_lr: float
    ):
        latest_stage = self.get_latest_stage_state()

        if self.unfreeze_condition():
            # Check max layers condition
            if self.max_stages_condition():
                logger.debug("Hit max stage to unlock, skip unfreezing")
            else:
                # First unlock
                if not self.stages_states[0].unlocked:
                    self.classes_states.reset()
                    stage = self.stages_states[0]
                    stage.unlocked = True
                    stage.unlocked_epoch = epoch
                    model.backbone.unfreeze_stage(stage, optimizer=optimizer)

                # Check cooldown
                else:
                    if not self.cooldown_condition(latest_stage, epoch=epoch):
                        self.classes_states.reset()
                        new_stage = self.stages_states[self.unlocked_stage_count]
                        new_stage.unlocked_epoch = epoch
                        new_stage.unlocked = True
                        model.backbone.unfreeze_stage(new_stage, optimizer=optimizer)

        # Set lr for each unlocked stage
        for i, stage in enumerate(self.stages_states):
            if stage.unlocked:
                target_lr = last_lr * self.training_params.get_depth_ratio(i)
                if self.cooldown_condition(stage, epoch=epoch):
                    lr = target_lr * stage.get_warmup_lr(epoch)
                else:
                    lr = target_lr
                for group in optimizer.param_groups:
                    if group.get("name") == stage.name:
                        group["lr"] = lr
                        break

    def patience_counter(
        self,
        classes_metric: list[float],
        min_delta: float = 0.0015,
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

        logger.debug(self.classes_states)

    def stop_condition(self) -> bool:
        """Evaluate staleness of each class and returns wether to stop

        Args:
            classes_patience (list[int]): stale_count for each class
            stop_patience (int): stop_patience value for this run

        Returns:
            bool: Stop run boolean
        """
        c = min(self.classes_states.staleness) >= self.training_params.stopping_patience
        if c:
            logger.debug(
                "Stop condition reached, "
                f"patience = {self.training_params.stopping_patience} "
                f"state = {self.classes_states}"
            )
        return c

    def unfreeze_condition(self, stale_class_ratio: float = 0.66) -> bool:
        c = (
            sum(
                [
                    c >= self.training_params.unfreezing_patience
                    for c in self.classes_states.staleness
                ]
            )
            / len(self.classes_states.staleness)
            >= stale_class_ratio
        )
        if c:
            logger.debug(
                f"Unfreeze condition reached, "
                f"patience = {self.training_params.unfreezing_patience} "
                f"state = {self.classes_states}"
            )
        return c

    def checkpoint_condition(self) -> bool:
        return min(self.classes_states.staleness) == 0

    def cooldown_condition(self, stage_state: StageUnfreezeState, epoch: int):
        c = (
            epoch - stage_state.unlocked_epoch
            <= self.training_params.unfreezing_cooldown
        )
        if c:
            logger.debug(
                f"{stage_state.name} still in cooldown, "
                f"{epoch - stage_state.unlocked_epoch} / "
                f"{self.training_params.unfreezing_cooldown}"
            )
        return c

    def max_stages_condition(self) -> bool:
        c = self.unlocked_stage_count >= min(
            self.training_params.max_stages, len(self.stages_states)
        )
        if c:
            logger.debug(f"Max stages reached: {self.unlocked_stage_count}")
        return c

    def to_dict(self):
        d = asdict(self)
        d["unlocked_stage_count"] = self.unlocked_stage_count
        return d

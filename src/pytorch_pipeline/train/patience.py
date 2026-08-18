from ..utils.configs import ClassesObjectiveState


def patience_counter(
    classes_metric: list[float],
    classes_conditions: ClassesObjectiveState,
    min_delta: float = 0.003,
) -> ClassesObjectiveState:
    """Worst class still improving patience.
    Compares each class newest metric to previous best and updates staleness.

    Args:
        classes_metric (list[float]): Current epoch metrics for each class
        classes_conditions (tuple[list[float],list[int]]):
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
        if m > classes_conditions.best_metrics[i] + min_delta:
            classes_conditions.best_metrics[i] = m
            classes_conditions.staleness[i] = 0
            classes_conditions.log_state_update(i)
        else:
            classes_conditions.staleness[i] += 1

    return classes_conditions


def stop_condition(classes_conditions: ClassesObjectiveState, patience: int) -> bool:
    """Evaluate staleness of each class and returns wether to stop

    Args:
        classes_patience (list[int]): stale_count for each class
        patience (int): patience value for this run

    Returns:
        bool: Stop run boolean
    """
    return min(classes_conditions.staleness) >= patience

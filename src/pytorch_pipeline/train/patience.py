from ..utils.configs import ClassesPatienceCondition


def patience_counter(
    classes_metric: list[float], classes_conditions: ClassesPatienceCondition
) -> ClassesPatienceCondition:
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
    for i, m in enumerate(classes_metric):
        print(m)
        if m > classes_conditions.best_metrics[i]:
            classes_conditions.best_metrics[i] = m
            classes_conditions.staleness[i] = 0
        else:
            classes_conditions.staleness[i] += 1

    return classes_conditions


def stop_condition(classes_conditions: ClassesPatienceCondition, patience: int) -> bool:
    """Evaluate staleness of each class and returns wether to stop

    Args:
        classes_patience (list[int]): stale_count for each class
        patience (int): patience value for this run

    Returns:
        bool: Stop run boolean
    """
    return min(classes_conditions.staleness) >= patience

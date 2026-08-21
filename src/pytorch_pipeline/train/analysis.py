import mlflow
import numpy as np

from ..utils import LABEL_MAPPING
from .metrics import EpochMetrics


def error_analysis(
    all_obs_ids: list[int],
    all_preds_raw: np.ndarray,
    all_labels: np.ndarray,
    metrics: EpochMetrics,
):
    best_thresholds = metrics.get_per_class("best_thresh")
    num_classes = all_labels.shape[1]

    report: dict[str, dict[str, list[int]]] = {}
    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")
        y_true = all_labels[:, i]
        y_score = all_preds_raw[:, i]
        y_pred = (y_score >= best_thresholds[i]).astype(int)

        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        fp_obs_ids = np.asarray(all_obs_ids)[fp_mask].tolist()
        fn_obs_ids = np.asarray(all_obs_ids)[fn_mask].tolist()
        report[label_name] = {"fp": fp_obs_ids, "fn": fn_obs_ids}
    return report


def log_error_analysis(report: dict, epoch):
    if not mlflow.active_run():
        return
    mlflow.log_dict(report, f"error_analysis/epoch_{epoch:03d}/report.json")

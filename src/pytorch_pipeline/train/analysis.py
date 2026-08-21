from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
import numpy as np

from ..utils import LABEL_MAPPING

if TYPE_CHECKING:
    import torch

    from .metrics import EpochMetrics


def error_analysis(
    all_obs_ids: list[int],
    all_preds_raw: np.ndarray,
    all_labels: np.ndarray,
    observations_attention_weights: dict[str, list[torch.Tensor]],
    metrics: EpochMetrics,
):
    best_thresholds = metrics.get_per_class("best_thresh")
    num_classes = all_labels.shape[1]

    report: dict[str, dict] = {}
    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")
        y_true = all_labels[:, i]
        y_score = all_preds_raw[:, i]
        y_pred = (y_score >= best_thresholds[i]).astype(int)
        attention_weights = observations_attention_weights[label_name]

        assert len(all_obs_ids) == len(attention_weights), (
            f"{label_name}: "
            f"{len(all_obs_ids)} observation IDs but "
            f"{len(attention_weights)} attention weights"
        )

        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        fp = [
            {
                "obs_id": obs_id,
                "weights": weights.detach().cpu().squeeze(-1).tolist(),
            }
            for obs_id, weights, is_fp in zip(all_obs_ids, attention_weights, fp_mask)
            if is_fp
        ]

        fn = [
            {
                "obs_id": obs_id,
                "weights": weights.detach().cpu().squeeze(-1).tolist(),
            }
            for obs_id, weights, is_fn in zip(all_obs_ids, attention_weights, fn_mask)
            if is_fn
        ]

        report[label_name] = {"fp": fp, "fn": fn}
    return report


def log_error_analysis(report: dict, epoch):
    if not mlflow.active_run():
        return
    mlflow.log_dict(report, f"error_analysis/epoch_{epoch:03d}/report.json")

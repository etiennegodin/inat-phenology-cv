from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    all_obs_paths: list[list[str]] | dict[int, list[str]] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """
    Generate misclassification analysis report for False Positives and False Negatives.

    Includes observation ID, attention weights, raw prediction probability,
    ground truth target, optimal threshold, and optional photo file paths.
    """
    best_thresholds = metrics.get_per_class("best_thresh")
    num_classes = all_labels.shape[1]

    obs_paths_map: dict[int, list[str]] = {}
    if isinstance(all_obs_paths, dict):
        obs_paths_map = all_obs_paths
    elif isinstance(all_obs_paths, list):
        obs_paths_map = dict(zip(all_obs_ids, all_obs_paths))

    report: dict[str, dict[str, list[dict[str, Any]]]] = {}
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

        fp = []
        for obs_id, weights, score, target_val, is_fp in zip(
            all_obs_ids, attention_weights, y_score, y_true, fp_mask
        ):
            if is_fp:
                entry: dict[str, Any] = {
                    "obs_id": int(obs_id),
                    "weights": weights.detach().cpu().squeeze(-1).tolist(),
                    "prob": round(float(score), 4),
                    "target": int(target_val),
                    "threshold": round(float(best_thresholds[i]), 4),
                }
                if obs_id in obs_paths_map:
                    entry["paths"] = obs_paths_map[obs_id]
                fp.append(entry)

        fn = []
        for obs_id, weights, score, target_val, is_fn in zip(
            all_obs_ids, attention_weights, y_score, y_true, fn_mask
        ):
            if is_fn:
                entry = {
                    "obs_id": int(obs_id),
                    "weights": weights.detach().cpu().squeeze(-1).tolist(),
                    "prob": round(float(score), 4),
                    "target": int(target_val),
                    "threshold": round(float(best_thresholds[i]), 4),
                }
                if obs_id in obs_paths_map:
                    entry["paths"] = obs_paths_map[obs_id]
                fn.append(entry)

        report[label_name] = {"fp": fp, "fn": fn}
    return report


def log_error_analysis(report: dict, epoch: int):
    if not mlflow.active_run():
        return
    mlflow.log_dict(report, f"error_analysis/epoch_{epoch:03d}/report.json")

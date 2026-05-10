from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    roc_curve,
)

from ..utils.registry import LABEL_MAPPING


def log_metrics(all_preds, all_labels, all_preds_raw, epoch: int) -> dict[str, Any]:

    roc_aucs = np.arange(3, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(all_preds.shape[1]):
        # Metrics
        fpr, tpr, thresholds = roc_curve(all_labels[:, i], all_preds_raw[:, i])
        roc_auc = auc(fpr, tpr)
        roc_aucs[i] = roc_auc
        roc_display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, name=LABEL_MAPPING[i]
        )
        roc_display.plot(ax=ax)

    ax.set_title("Multi-label ROC Curves")
    plt.legend()

    eval_metrics = {
        "val_roc_auc_macro": roc_aucs.mean(axis=0),
        "val_roc_auc_label0": roc_aucs[0],
        "val_roc_auc_label1": roc_aucs[1],
        "val_roc_auc_label2": roc_aucs[2],
    }

    # Log metrics
    if mlflow.active_run():
        mlflow.log_metrics(eval_metrics, step=epoch)

    # Update with display object
    eval_metrics.update({"roc_fig": fig, "roc_ax": ax})

    return eval_metrics


def log_best_artifacts(eval_metrics: dict) -> None:
    # Val loss
    mlflow.log_metric("best_val_loss", eval_metrics["val_loss"])

    # Roc plot
    roc_fig = eval_metrics["roc_fig"]
    mlflow.log_figure(roc_fig, "roc.png")

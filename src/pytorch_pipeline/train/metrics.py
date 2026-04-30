from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)


def get_metrics(all_preds, all_labels, all_preds_raw) -> dict[str, Any]:
    # Metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds_raw)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
    )
    cm = confusion_matrix(all_labels, all_preds)
    cm_display = ConfusionMatrixDisplay(cm)
    cf = classification_report(all_labels, all_preds)

    eval_metrics = {
        "val_roc_auc": roc_auc,
        "cm": cm,
        "cm_display": cm_display,
        "cf": cf,
        "roc_display": roc_display,
    }

    return eval_metrics


def log_best_artifacts(eval_metrics: dict) -> None:
    # Val loss
    mlflow.log_metric("best_val_loss", eval_metrics["val_loss"])

    # Roc plot
    roc_display = eval_metrics["roc_display"]
    roc_display.plot()
    roc_fig = roc_display.figure_
    mlflow.log_figure(roc_fig, "roc.png")

    # Confusion matrix plot
    plt.figure()
    plt.title("Confusion matrix")
    plt.set_cmap("inferno")
    cm_display = eval_metrics["cm_display"]
    cm_display.plot()
    cm_fig = cm_display.figure_
    plt.close()
    mlflow.log_figure(cm_fig, "confusion_matrix.png")

    # CLassification_report
    with open("classification_report.txt", "w") as f:
        f.write(eval_metrics["cf"])
    mlflow.log_artifact("classification_report.txt")
    Path("classification_report.txt").unlink()

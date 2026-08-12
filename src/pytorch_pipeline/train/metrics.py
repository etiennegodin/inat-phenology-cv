from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from ..utils import format_dict
from ..utils.configs import LABEL_MAPPING

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from .model import PhenologyModel

logger = logging.getLogger(__name__)


def log_attention_metrics(
    epoch: int, obs_weights_dict: dict[str, list[Any]], prefix: str = "val"
):
    classes_df = []

    # Get attention values per class, accumulate binned for single plot
    for named_class, obs_weights_list in obs_weights_dict.items():
        entropies_records = compute_attention_values(obs_weights_list)
        if entropies_records:
            attn_metrics = {
                f"attention/{named_class}/entropy": float(
                    np.mean(entropies_records["entropy"])
                ),
                f"attention/{named_class}/entropy_std": float(
                    np.std(entropies_records["entropy"])
                ),
                f"attention/{named_class}/entropy_max": float(
                    np.max(entropies_records["entropy"])
                ),
                f"attention/{named_class}/entropy_min": float(
                    np.min(entropies_records["entropy"])
                ),
            }
            df_temp = pd.DataFrame(entropies_records)
            df_temp["class"] = named_class
            classes_df.append(df_temp)
            if mlflow.active_run():
                for k, v in attn_metrics.items():
                    mlflow.log_metric(f"{prefix}/{k}", v, step=epoch)

    if classes_df:
        # Single plot for attention binned
        df = pd.concat(classes_df, ignore_index=True)
        attn_ax = sns.lineplot(df, x="img_count", y="entropy", hue="class")
        plt.ylim(0, 1.1)
        plt.xlim(2, max(df["img_count"]))
        attn_ax.set_xlabel("n_images")
        attn_ax.set_ylabel("Mean Normalized Entropy")
        attn_ax.set_title("Mean Normalized Entropy by image count", fontsize=10)
        attn_fig = attn_ax.figure

        if mlflow.active_run():
            mlflow.log_figure(
                attn_fig, f"plots/epoch_{epoch:03d}/{prefix}_multi_class_attention.png"
            )
            plt.close(attn_fig)


def compute_attention_values(
    obs_weights_list: list[Any],
) -> dict[str, list[np.ndarray]]:
    """Compute statistics on ADBIL attention weights across observations.

    obs_weights_list is a list of weight Tensors per batch/observation.
    """
    if not obs_weights_list:
        return {}

    # Log entropy per n_images
    entropies_records = {"img_count": [], "entropy": []}

    for batch in obs_weights_list:
        weights_iter = batch if isinstance(batch, (list, tuple)) else [batch]
        for w in weights_iter:
            if w is None:
                continue
            if hasattr(w, "detach"):
                w_arr = w.detach().cpu().numpy().squeeze()
            else:
                w_arr = np.squeeze(np.asarray(w))

            if w_arr.ndim == 0:
                w_arr = np.array([w_arr])

            # Get img count for this observation
            img_count = len(w_arr)
            if img_count < 2:
                continue

            # Entropy ceiling for relative entropies
            max_entropy = math.log(img_count)
            entropy = -np.sum(w_arr * np.log(w_arr + 1e-12))
            normalised_entropy = entropy / max_entropy
            entropies_records["img_count"].append(img_count)
            entropies_records["entropy"].append(normalised_entropy)

    if not entropies_records["entropy"]:
        return {}

    return entropies_records


def find_optimal_threshold(
    labels: np.ndarray, preds_raw: np.ndarray
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Find threshold that maximizes F1 score for a single binary label."""
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = []
    precs = []
    recalls = []

    for t in thresholds:
        bin_preds = (preds_raw >= t).astype(int)
        p = precision_score(labels, bin_preds, zero_division=0)
        r = recall_score(labels, bin_preds, zero_division=0)
        f1 = f1_score(labels, bin_preds, zero_division=0)
        f1s.append(f1)
        precs.append(p)
        recalls.append(r)

    f1s = np.array(f1s)
    best_idx = np.argmax(f1s)
    best_thresh = float(thresholds[best_idx])
    max_f1 = float(f1s[best_idx])
    prec_at_best = float(precs[best_idx])
    recall_at_best = float(recalls[best_idx])

    return best_thresh, max_f1, prec_at_best, recall_at_best, thresholds, f1s


def calculate_multilabel_metrics(
    all_preds_raw: np.ndarray,
    all_labels: np.ndarray,
    prefix: str = "val",
) -> dict[str, Any]:
    """Calculate comprehensive multi-label evaluation metrics grouped with slashes."""
    num_classes = all_labels.shape[1]
    all_preds_bin_05 = (all_preds_raw >= 0.5).astype(int)

    metrics: dict[str, Any] = {}
    per_class_reports = []
    roc_aucs = []
    pr_aucs = []
    f1s_05 = []
    best_f1s = []
    best_thresholds = []

    thresh_curves = {}

    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")
        label_name_clean = label_name.lower().replace(" ", "_")

        y_true = all_labels[:, i]
        y_score = all_preds_raw[:, i]
        y_pred_05 = all_preds_bin_05[:, i]

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc_val = auc(fpr, tpr)
            pr_auc_val = average_precision_score(y_true, y_score)
        else:
            roc_auc_val = 0.5
            pr_auc_val = float(np.mean(y_true))

        roc_aucs.append(roc_auc_val)
        pr_aucs.append(pr_auc_val)

        p_05 = precision_score(y_true, y_pred_05, zero_division=0)
        r_05 = recall_score(y_true, y_pred_05, zero_division=0)
        f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
        f1s_05.append(f1_05)

        best_t, best_f1, best_p, best_r, thresh_arr, f1_arr = find_optimal_threshold(
            y_true, y_score
        )
        best_f1s.append(best_f1)
        best_thresholds.append(best_t)
        thresh_curves[label_name] = (thresh_arr, f1_arr)

        # Per-class grouped metrics with slash notation
        metrics[f"{prefix}/roc_auc/{label_name_clean}"] = float(roc_auc_val)
        metrics[f"{prefix}/pr_auc/{label_name_clean}"] = float(pr_auc_val)
        metrics[f"{prefix}/f1_0.5/{label_name_clean}"] = float(f1_05)
        metrics[f"{prefix}/precision_0.5/{label_name_clean}"] = float(p_05)
        metrics[f"{prefix}/recall_0.5/{label_name_clean}"] = float(r_05)
        metrics[f"{prefix}/best_thresh/{label_name_clean}"] = float(best_t)
        metrics[f"{prefix}/best_f1/{label_name_clean}"] = float(best_f1)

        per_class_reports.append(
            {
                "Class": label_name,
                "Support_Pos": int(np.sum(y_true)),
                "Support_Neg": int(len(y_true) - np.sum(y_true)),
                "ROC_AUC": round(float(roc_auc_val), 4),
                "PR_AUC": round(float(pr_auc_val), 4),
                "F1_0.5": round(float(f1_05), 4),
                "Prec_0.5": round(float(p_05), 4),
                "Recall_0.5": round(float(r_05), 4),
                "Best_Thresh": round(float(best_t), 4),
                "Best_F1": round(float(best_f1), 4),
                "Prec_Best": round(float(best_p), 4),
                "Recall_Best": round(float(best_r), 4),
            }
        )

    # Macro & Global Aggregate Metrics with slash notation
    metrics[f"{prefix}/roc_auc_macro"] = float(np.mean(roc_aucs))
    metrics[f"{prefix}/pr_auc_macro"] = float(np.mean(pr_aucs))
    metrics[f"{prefix}/f1_macro_0.5"] = float(np.mean(f1s_05))
    metrics[f"{prefix}/f1_macro_best"] = float(np.mean(best_f1s))

    metrics[f"{prefix}/f1_micro_0.5"] = float(
        f1_score(all_labels, all_preds_bin_05, average="micro", zero_division=0)
    )
    metrics[f"{prefix}/f1_weighted_0.5"] = float(
        f1_score(all_labels, all_preds_bin_05, average="weighted", zero_division=0)
    )

    exact_matches = np.all(all_preds_bin_05 == all_labels, axis=1)
    metrics[f"{prefix}/exact_match_ratio"] = float(np.mean(exact_matches))
    metrics[f"{prefix}/hamming_loss"] = float(
        hamming_loss(all_labels, all_preds_bin_05)
    )

    metrics["_per_class_reports"] = per_class_reports
    metrics["_thresh_curves"] = thresh_curves
    metrics["_best_thresholds"] = best_thresholds

    return metrics


def generate_metric_plots(
    all_preds_raw: np.ndarray,
    all_labels: np.ndarray,
    metrics: dict[str, Any],
) -> dict[str, plt.Figure]:
    """Generate diagnostic plots for multi-label classification."""
    num_classes = all_labels.shape[1]
    figures = {}

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # 1. Multi-label ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")
        y_true = all_labels[:, i]
        y_score = all_preds_raw[:, i]
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc_val = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc_val:.3f})", lw=2)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Multi-Label ROC Curves", fontsize=14, fontweight="bold")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    figures["roc_curves"] = fig_roc

    # 2. Multi-label Precision-Recall Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")
        y_true = all_labels[:, i]
        y_score = all_preds_raw[:, i]
        if len(np.unique(y_true)) > 1:
            p, r, _ = precision_recall_curve(y_true, y_score)
            ap_val = average_precision_score(y_true, y_score)
            ax_pr.plot(r, p, label=f"{label_name} (AP = {ap_val:.3f})", lw=2)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(
        "Multi-Label Precision-Recall Curves", fontsize=14, fontweight="bold"
    )
    ax_pr.legend(loc="lower left")
    fig_pr.tight_layout()
    figures["pr_curves"] = fig_pr

    # 3. Threshold vs F1-Score Plot
    fig_tf1, ax_tf1 = plt.subplots(figsize=(8, 6))
    thresh_curves = metrics.get("_thresh_curves", {})
    best_thresholds = metrics.get("_best_thresholds", [])

    for i, (label_name, (threshs, f1s)) in enumerate(thresh_curves.items()):
        ax_tf1.plot(threshs, f1s, label=f"{label_name}", lw=2)
        if i < len(best_thresholds):
            bt = best_thresholds[i]
            ax_tf1.axvline(
                x=bt,
                linestyle="--",
                alpha=0.5,
                label=f"Opt {label_name} ({bt:.2f})",
            )

    ax_tf1.set_xlabel("Decision Threshold")
    ax_tf1.set_ylabel("F1 Score")
    ax_tf1.set_title("Per-Class Threshold vs F1-Score", fontsize=14, fontweight="bold")
    ax_tf1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig_tf1.tight_layout()
    figures["threshold_f1"] = fig_tf1

    # 4. Confusion Matrices Grid
    all_preds_bin_05 = (all_preds_raw >= 0.5).astype(int)
    cols = min(num_classes, 3)
    rows = (num_classes + cols - 1) // cols
    fig_cm, axes_cm = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_flat = np.atleast_1d(axes_cm).flatten()

    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")
        cm = confusion_matrix(all_labels[:, i], all_preds_bin_05[:, i])
        ax = axes_flat[i]
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"CM: {label_name}", fontsize=11, fontweight="bold")
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Absent", "Present"])
        ax.set_yticklabels(["Absent", "Present"])
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(
                    c,
                    r,
                    format(cm[r, c], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontweight="bold",
                )

    for j in range(num_classes, len(axes_flat)):
        axes_flat[j].axis("off")

    fig_cm.tight_layout()
    figures["confusion_matrices"] = fig_cm

    return figures


def log_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    all_preds_raw: np.ndarray,
    epoch: int,
    prefix: str = "val",
) -> dict[str, Any]:
    """Calculate and log evaluation metrics with slash grouping and epoch plots."""
    eval_metrics = calculate_multilabel_metrics(
        all_preds_raw=all_preds_raw,
        all_labels=all_labels,
        prefix=prefix,
    )

    scalar_metrics = {
        k: v
        for k, v in eval_metrics.items()
        if isinstance(v, (int, float, np.number)) and not k.startswith("_")
    }

    if mlflow.active_run():
        mlflow.log_metrics(scalar_metrics, step=epoch)

        plots = generate_metric_plots(
            all_preds_raw=all_preds_raw,
            all_labels=all_labels,
            metrics=eval_metrics,
        )

        # Log plot figures grouped in per-epoch subfolder: plots/epoch_001/
        for name, fig in plots.items():
            mlflow.log_figure(fig, f"plots/epoch_{epoch:03d}/{name}.png")
            plt.close(fig)

        # Log classification report in epoch subfolder: plots/epoch_001/
        if "_per_class_reports" in eval_metrics:
            report_df = pd.DataFrame(eval_metrics["_per_class_reports"])
            csv_str = report_df.to_csv(index=False)
            mlflow.log_text(
                csv_str, f"plots/epoch_{epoch:03d}/classification_report.csv"
            )

    return eval_metrics


def log_best_artifacts(eval_metrics: dict[str, Any]) -> None:
    """Log final best checkpoint metrics grouped with slashes."""
    if not mlflow.active_run():
        return

    best_scalars = {
        "best/val_loss": eval_metrics.get(
            "val/loss", eval_metrics.get("val_loss", 0.0)
        ),
        "best/val_roc_auc_macro": eval_metrics.get(
            "val/roc_auc_macro", eval_metrics.get("val_roc_auc_macro", 0.0)
        ),
        "best/val_pr_auc_macro": eval_metrics.get(
            "val/pr_auc_macro", eval_metrics.get("val_pr_auc_macro", 0.0)
        ),
        "best/val_f1_macro_best": eval_metrics.get(
            "val/f1_macro_best", eval_metrics.get("val_f1_macro_best", 0.0)
        ),
    }
    mlflow.log_metrics(best_scalars)

    if "_per_class_reports" in eval_metrics:
        report_df = pd.DataFrame(eval_metrics["_per_class_reports"])
        mlflow.log_text(report_df.to_csv(index=False), "best_classification_report.csv")
        mlflow.log_text(
            report_df.to_markdown(index=False), "best_classification_report.md"
        )


def log_experiment_metadata(
    model: PhenologyModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> None:
    """Log dataset statistics and model architecture overview to MLflow."""
    if not mlflow.active_run():
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Backbone params
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    backbone_trainable_params = sum(
        p.numel() for p in model.backbone.parameters() if p.requires_grad
    )
    backbone_frozen_params = backbone_params - backbone_trainable_params

    # Attention params
    attention_params = sum(p.numel() for p in model.branches[0].attention.parameters())
    attention_trainable_params = sum(
        p.numel() for p in model.branches[0].attention.parameters() if p.requires_grad
    )
    attention_frozen_params = attention_params - attention_trainable_params

    # Classifier params
    classifier_params = sum(p.numel() for p in model.branches[0].head.parameters())
    classifier_trainable_params = sum(
        p.numel() for p in model.branches[0].head.parameters() if p.requires_grad
    )
    classifier_frozen_params = classifier_params - classifier_trainable_params

    model_summary = {
        "percentages": {
            "total": {
                "backbone_pct": round(backbone_params / total_params * 100, 2),
                "attention_pct": round(attention_params / total_params * 100, 2),
                "classifier_pct": round(classifier_params / total_params * 100, 2),
            },
            "trainable": {
                "backbone_pct": round(
                    backbone_trainable_params / trainable_params * 100, 2
                ),
                "attention_pct": round(
                    attention_trainable_params / trainable_params * 100, 2
                ),
                "classifier_pct": round(
                    classifier_trainable_params / trainable_params * 100, 2
                ),
            },
        },
        "total": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "trainable_percent": round(
                (trainable_params / max(total_params, 1)) * 100, 2
            ),
        },
        "backbone_architecture": {
            "params": backbone_params,
            "trainable_params": backbone_trainable_params,
            "frozen_params": backbone_frozen_params,
            "trainable_percent": round(
                (backbone_trainable_params / max(backbone_params, 1)) * 100, 2
            ),
        },
        "attention_architecture": {
            "params": attention_params,
            "trainable_params": attention_trainable_params,
            "frozen_params": attention_frozen_params,
            "trainable_percent": round(
                (attention_trainable_params / max(attention_params, 1)) * 100, 2
            ),
        },
        "classifier_architecture": {
            "params": classifier_params,
            "trainable_params": classifier_trainable_params,
            "frozen_params": classifier_frozen_params,
            "trainable_percent": round(
                (classifier_trainable_params / max(classifier_params, 1)) * 100, 2
            ),
        },
    }
    mlflow.log_params(model_summary)
    mlflow.log_dict(model_summary, "model_summary.json")

    def get_dataset_stats(dataset, name):
        if not hasattr(dataset, "df") or dataset.df is None:
            return {}

        df = dataset.df
        labels_list = df["label"].tolist() if "label" in df.columns else []
        if not labels_list:
            return {}

        labels_arr = np.array(labels_list)
        total_obs = len(labels_arr)
        pos_counts = np.sum(labels_arr, axis=0)

        stats = {"total_observations": total_obs}
        for i, pos_c in enumerate(pos_counts):
            label_name = LABEL_MAPPING.get(i, f"label_{i}")
            stats[f"{label_name}_pos_count"] = int(pos_c)
            stats[f"{label_name}_pos_ratio"] = round(float(pos_c / total_obs), 4)

        if hasattr(dataset, "bag_sizes"):
            stats["avg_images_per_obs"] = round(float(np.mean(dataset.bag_sizes)), 2)

        return stats

    train_stats = get_dataset_stats(train_dataset, "train")
    val_stats = get_dataset_stats(val_dataset, "val")

    dataset_summary = {
        "train": train_stats,
        "val": val_stats,
    }
    mlflow.log_dict(dataset_summary, "dataset_summary.json")
    logger.debug(
        f"Experiment metadata logged to MLflow:\n {format_dict(model_summary)}"
    )

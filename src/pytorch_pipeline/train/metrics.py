from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
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
from ..utils.configs import CLASS_ORDER, LABEL_MAPPING

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from .model import PhenologyModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EpochMetrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class EpochMetrics:
    """Typed container for one evaluation epoch's scalar metrics.

    Scalar-only: no matplotlib figures, numpy curve arrays, or report dicts.
    Safe to serialise with dataclasses.asdict() and checkpoint alongside model
    weights. Import this class anywhere (tests, inference) without pulling in
    mlflow or sklearn.
    """

    # -- Aggregate scalars (checkpoint / patience decisions) -----------------
    roc_auc_macro: float
    pr_auc_macro: float
    pr_norm_excess_macro: float
    f1_macro_best: float
    f1_macro_05: float
    f1_micro_05: float
    f1_weighted_05: float
    exact_match_ratio: float
    hamming_loss: float
    val_loss: float

    # -- Per-class scalars ----------------------------------------------------
    roc_auc: dict[str, float]  # {"Flowering": 0.87, ...}
    pr_auc: dict[str, float]
    pr_norm_excess: dict[str, float]
    best_thresh: dict[str, float]
    best_f1: dict[str, float]
    best_prec: dict[str, float]
    best_recall: dict[str, float]
    f1_05: dict[str, float]
    precision_05: dict[str, float]
    recall_05: dict[str, float]
    support_pos: dict[str, int]
    support_neg: dict[str, int]

    prefix: str = "val"

    # -- Convenience methods --------------------------------------------------

    def pr_norm_excess_per_class(self) -> list[float]:
        """Ordered list matching CLASS_ORDER -- feeds patience_counter directly."""
        return [self.pr_norm_excess[c] for c in CLASS_ORDER]

    def scalar_dict(self) -> dict[str, float]:
        """Flat '{prefix}/key' dict suitable for mlflow.log_metrics().

        All keys are derived from field names -- no hardcoded strings in callers.
        """
        p = self.prefix
        d: dict[str, float] = {
            f"{p}/roc_auc_macro": self.roc_auc_macro,
            f"{p}/pr_auc_macro": self.pr_auc_macro,
            f"{p}/pr_norm_excess_macro": self.pr_norm_excess_macro,
            f"{p}/f1_macro_best": self.f1_macro_best,
            f"{p}/f1_macro_0.5": self.f1_macro_05,
            f"{p}/f1_micro_0.5": self.f1_micro_05,
            f"{p}/f1_weighted_0.5": self.f1_weighted_05,
            f"{p}/exact_match_ratio": self.exact_match_ratio,
            f"{p}/hamming_loss": self.hamming_loss,
            f"{p}/loss": self.val_loss,
        }
        per_class_fields: list[tuple[str, dict]] = [
            ("roc_auc", self.roc_auc),
            ("pr_auc", self.pr_auc),
            ("pr_norm_excess", self.pr_norm_excess),
            ("best_thresh", self.best_thresh),
            ("best_f1", self.best_f1),
            ("f1_0.5", self.f1_05),
            ("precision_0.5", self.precision_05),
            ("recall_0.5", self.recall_05),
        ]
        for group, mapping in per_class_fields:
            for cls_name, v in mapping.items():
                cls_clean = cls_name.lower().replace(" ", "_")
                d[f"{p}/{group}/{cls_clean}"] = float(v)
        return d

    def to_dict(self) -> dict[str, Any]:
        """Full serialisable dict for checkpointing."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpochMetrics":
        """Reconstruct from a checkpointed dict (e.g. loaded via torch.load)."""
        return cls(**d)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


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


def compute_metrics(
    all_preds_raw: np.ndarray,
    all_labels: np.ndarray,
    pos_ratios: list[float],
    val_loss: float = 0.0,
    prefix: str = "val",
) -> EpochMetrics:
    """Compute comprehensive multi-label evaluation metrics.

    Returns a typed EpochMetrics dataclass. Has no side-effects (no logging).
    Call log_epoch_metrics() separately to push results to MLflow.
    """
    num_classes = all_labels.shape[1]
    all_preds_bin_05 = (all_preds_raw >= 0.5).astype(int)

    roc_auc: dict[str, float] = {}
    pr_auc: dict[str, float] = {}
    pr_norm_excess: dict[str, float] = {}
    best_thresh: dict[str, float] = {}
    best_f1: dict[str, float] = {}
    best_prec: dict[str, float] = {}
    best_recall: dict[str, float] = {}
    f1_05: dict[str, float] = {}
    precision_05: dict[str, float] = {}
    recall_05: dict[str, float] = {}
    support_pos: dict[str, int] = {}
    support_neg: dict[str, int] = {}

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    pr_norm_excesses: list[float] = []
    f1s_05: list[float] = []
    best_f1s: list[float] = []

    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")

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

        pr_norm_excess_val = (pr_auc_val - pos_ratios[i]) / (1 - pos_ratios[i])

        roc_aucs.append(roc_auc_val)
        pr_aucs.append(pr_auc_val)
        pr_norm_excesses.append(pr_norm_excess_val)

        p_05 = precision_score(y_true, y_pred_05, zero_division=0)
        r_05 = recall_score(y_true, y_pred_05, zero_division=0)
        f1_05_val = f1_score(y_true, y_pred_05, zero_division=0)
        f1s_05.append(f1_05_val)

        best_t, best_f1_val, best_p, best_r, _, _ = find_optimal_threshold(
            y_true, y_score
        )
        best_f1s.append(best_f1_val)

        roc_auc[label_name] = float(roc_auc_val)
        pr_auc[label_name] = float(pr_auc_val)
        pr_norm_excess[label_name] = float(pr_norm_excess_val)
        best_thresh[label_name] = float(best_t)
        best_f1[label_name] = float(best_f1_val)
        best_prec[label_name] = float(best_p)
        best_recall[label_name] = float(best_r)
        f1_05[label_name] = float(f1_05_val)
        precision_05[label_name] = float(p_05)
        recall_05[label_name] = float(r_05)
        support_pos[label_name] = int(np.sum(y_true))
        support_neg[label_name] = int(len(y_true) - np.sum(y_true))

    return EpochMetrics(
        roc_auc_macro=float(np.mean(roc_aucs)),
        pr_auc_macro=float(np.mean(pr_aucs)),
        pr_norm_excess_macro=float(np.mean(pr_norm_excesses)),
        f1_macro_best=float(np.mean(best_f1s)),
        f1_macro_05=float(np.mean(f1s_05)),
        f1_micro_05=float(
            f1_score(all_labels, all_preds_bin_05, average="micro", zero_division=0)
        ),
        f1_weighted_05=float(
            f1_score(all_labels, all_preds_bin_05, average="weighted", zero_division=0)
        ),
        exact_match_ratio=float(
            np.mean(np.all(all_preds_bin_05 == all_labels, axis=1))
        ),
        hamming_loss=float(hamming_loss(all_labels, all_preds_bin_05)),
        val_loss=val_loss,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        pr_norm_excess=pr_norm_excess,
        best_thresh=best_thresh,
        best_f1=best_f1,
        best_prec=best_prec,
        best_recall=best_recall,
        f1_05=f1_05,
        precision_05=precision_05,
        recall_05=recall_05,
        support_pos=support_pos,
        support_neg=support_neg,
        prefix=prefix,
    )


# ---------------------------------------------------------------------------
# Logging (side-effects only -- no computation)
# ---------------------------------------------------------------------------


def _build_per_class_report(metrics: EpochMetrics) -> list[dict[str, Any]]:
    """Reconstruct the per-class report rows from EpochMetrics scalar dicts."""
    rows = []
    for label_name in CLASS_ORDER:
        if label_name not in metrics.roc_auc:
            continue
        rows.append(
            {
                "Class": label_name,
                "Support_Pos": metrics.support_pos.get(label_name, 0),
                "Support_Neg": metrics.support_neg.get(label_name, 0),
                "ROC_AUC": round(metrics.roc_auc.get(label_name, 0.0), 4),
                "PR_AUC": round(metrics.pr_auc.get(label_name, 0.0), 4),
                "F1_0.5": round(metrics.f1_05.get(label_name, 0.0), 4),
                "Prec_0.5": round(metrics.precision_05.get(label_name, 0.0), 4),
                "Recall_0.5": round(metrics.recall_05.get(label_name, 0.0), 4),
                "Best_Thresh": round(metrics.best_thresh.get(label_name, 0.0), 4),
                "Best_F1": round(metrics.best_f1.get(label_name, 0.0), 4),
                "Prec_Best": round(metrics.best_prec.get(label_name, 0.0), 4),
                "Recall_Best": round(metrics.best_recall.get(label_name, 0.0), 4),
            }
        )
    return rows


def generate_metric_plots(
    all_preds_raw: np.ndarray,
    all_labels: np.ndarray,
    best_thresholds: dict[str, float],
) -> dict[str, plt.Figure]:
    """Generate diagnostic plots for multi-label classification.

    Args:
        all_preds_raw: Raw sigmoid predictions, shape (N, C).
        all_labels: Ground-truth binary labels, shape (N, C).
        best_thresholds: Per-class optimal thresholds from EpochMetrics.best_thresh.

    Returns:
        Dict mapping plot name to matplotlib Figure.
    """
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
    # Recompute per-class threshold curves (fast numpy ops, cheap vs. inference)
    fig_tf1, ax_tf1 = plt.subplots(figsize=(8, 6))
    for i in range(num_classes):
        label_name = LABEL_MAPPING.get(i, f"label_{i}")
        y_true = all_labels[:, i]
        y_score = all_preds_raw[:, i]
        _, _, _, _, threshs, f1s = find_optimal_threshold(y_true, y_score)
        ax_tf1.plot(threshs, f1s, label=f"{label_name}", lw=2)
        bt = best_thresholds.get(label_name)
        if bt is not None:
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


def log_epoch_metrics(
    metrics: EpochMetrics,
    all_preds_raw: np.ndarray,
    all_labels: np.ndarray,
    epoch: int,
) -> None:
    """Log an EpochMetrics snapshot to MLflow. Pure side-effect, no return value.

    Generates diagnostic plots and CSV report on-the-fly from raw arrays;
    these are logged and immediately discarded (not stored on EpochMetrics).

    Args:
        metrics: Typed scalar metrics for this epoch.
        all_preds_raw: Raw sigmoid predictions used to regenerate diagnostic plots.
        all_labels: Ground-truth binary labels.
        epoch: Current epoch number (used as MLflow step).
    """
    if not mlflow.active_run():
        return

    mlflow.log_metrics(metrics.scalar_dict(), step=epoch)

    plots = generate_metric_plots(
        all_preds_raw=all_preds_raw,
        all_labels=all_labels,
        best_thresholds=metrics.best_thresh,
    )
    for name, fig in plots.items():
        mlflow.log_figure(fig, f"plots/epoch_{epoch:03d}/{name}.png")
        plt.close(fig)

    rows = _build_per_class_report(metrics)
    if rows:
        report_df = pd.DataFrame(rows)
        mlflow.log_text(
            report_df.to_csv(index=False),
            f"plots/epoch_{epoch:03d}/classification_report.csv",
        )


def log_best_artifacts(metrics: EpochMetrics) -> None:
    """Log final best-checkpoint summary metrics and report to MLflow."""
    if not mlflow.active_run():
        return

    best_scalars = {
        "best/val_loss": metrics.val_loss,
        "best/val_roc_auc_macro": metrics.roc_auc_macro,
        "best/val_pr_auc_macro": metrics.pr_auc_macro,
        "best/val_f1_macro_best": metrics.f1_macro_best,
    }
    mlflow.log_metrics(best_scalars)

    rows = _build_per_class_report(metrics)
    if rows:
        report_df = pd.DataFrame(rows)
        mlflow.log_text(report_df.to_csv(index=False), "best_classification_report.csv")
        mlflow.log_text(
            report_df.to_markdown(index=False), "best_classification_report.md"
        )


# ---------------------------------------------------------------------------
# Attention metrics (unchanged)
# ---------------------------------------------------------------------------


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
    """_summary_

    Args:
        obs_weights_list (list[Any]): _description_

    Returns:
        dict[str, list[np.ndarray]]: {"img_count": [], "entropy": []}

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


# ---------------------------------------------------------------------------
# Experiment metadata (unchanged)
# ---------------------------------------------------------------------------


def log_experiment_metadata(
    model: "PhenologyModel",
    train_dataset: "Dataset",
    val_dataset: "Dataset",
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

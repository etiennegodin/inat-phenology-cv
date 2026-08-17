import numpy as np
import pytest

from pytorch_pipeline.train.metrics import (
    calculate_multilabel_metrics,
    compute_attention_values,
    find_optimal_threshold,
    generate_metric_plots,
)


def test_compute_attention_metrics():
    w1 = np.array([0.8, 0.2])
    w2 = np.array([0.33, 0.33, 0.34])
    w3 = np.array([1.0])

    res = compute_attention_values([w1, w2, w3])
    assert "img_count" in res
    assert "entropy" in res
    assert res["img_count"] == [2, 3]
    assert res["entropy"] == [
        np.float64(0.7219280948844771),
        np.float64(0.9999092749813393),
    ]


def test_find_optimal_threshold():
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    preds_raw = np.array([0.9, 0.85, 0.7, 0.2, 0.1, 0.4, 0.65, 0.3])

    best_thresh, max_f1, prec, recall, threshs, f1s = find_optimal_threshold(
        labels, preds_raw
    )
    assert 0.4 <= best_thresh <= 0.7
    assert max_f1 == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_calculate_multilabel_metrics():
    np.random.seed(42)
    N = 50
    C = 3

    all_labels = np.random.randint(0, 2, size=(N, C))
    all_preds_raw = np.clip(
        all_labels * 0.7 + np.random.uniform(0, 0.3, size=(N, C)), 0.01, 0.99
    )

    metrics = calculate_multilabel_metrics(all_preds_raw, all_labels, prefix="val")

    # Check aggregate metrics with slash notation
    assert "val/roc_auc_macro" in metrics
    assert "val/pr_auc_macro" in metrics
    assert "val/f1_macro_0.5" in metrics
    assert "val/f1_macro_best" in metrics
    assert "val/f1_micro_0.5" in metrics
    assert "val/exact_match_ratio" in metrics
    assert "val/hamming_loss" in metrics

    # Check per-class metrics with slash notation
    assert "val/roc_auc/flowering" in metrics
    assert "val/pr_auc/fruiting" in metrics
    assert "val/best_thresh/flower_budding" in metrics

    assert 0.0 <= metrics["val/roc_auc_macro"] <= 1.0
    assert 0.0 <= metrics["val/pr_auc_macro"] <= 1.0


def test_generate_metric_plots():
    N = 30
    C = 3
    all_labels = np.random.randint(0, 2, size=(N, C))
    all_preds_raw = np.random.uniform(0.1, 0.9, size=(N, C))

    metrics = calculate_multilabel_metrics(all_preds_raw, all_labels, prefix="val")
    figures = generate_metric_plots(all_preds_raw, all_labels, metrics)

    assert "roc_curves" in figures
    assert "pr_curves" in figures
    assert "threshold_f1" in figures
    assert "confusion_matrices" in figures

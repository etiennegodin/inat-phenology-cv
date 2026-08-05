import numpy as np
import pytest

from pytorch_pipeline.train.metrics import (
    calculate_multilabel_metrics,
    compute_attention_metrics,
    find_optimal_threshold,
    generate_metric_plots,
)


def test_compute_attention_metrics():
    # Simulated attention weights for 3 observations in a batch
    w1 = np.array([0.8, 0.2])  # Selective attention
    w2 = np.array([0.33, 0.33, 0.34])  # Uniform attention
    w3 = np.array([1.0])  # Single image

    res = compute_attention_metrics([w1, w2, w3])
    assert "attn_entropy" in res
    assert "attn_max_weight" in res
    assert "attn_min_weight" in res
    assert "attn_bag_size_mean" in res
    assert res["attn_bag_size_mean"] == pytest.approx(2.0)
    assert res["attn_max_weight"] > 0.6


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

    # Generate synthetic multi-label targets and predictions
    all_labels = np.random.randint(0, 2, size=(N, C))
    all_preds_raw = np.clip(
        all_labels * 0.7 + np.random.uniform(0, 0.3, size=(N, C)), 0.01, 0.99
    )

    metrics = calculate_multilabel_metrics(all_preds_raw, all_labels, prefix="val")

    # Check aggregate metrics
    assert "val_roc_auc_macro" in metrics
    assert "val_pr_auc_macro" in metrics
    assert "val_f1_macro_0.5" in metrics
    assert "val_f1_macro_best" in metrics
    assert "val_f1_micro_0.5" in metrics
    assert "val_exact_match_ratio" in metrics
    assert "val_hamming_loss" in metrics

    # Check per-class metrics
    assert "val_roc_auc_flowering" in metrics
    assert "val_pr_auc_fruiting" in metrics
    assert "val_best_thresh_flower_budding" in metrics

    assert 0.0 <= metrics["val_roc_auc_macro"] <= 1.0
    assert 0.0 <= metrics["val_pr_auc_macro"] <= 1.0


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

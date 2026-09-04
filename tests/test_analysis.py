from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import torch

from pytorch_pipeline.review import (
    plot_misclassified_observation,
    resolve_report_paths,
    review_misclassifications,
)
from pytorch_pipeline.train.analysis import error_analysis
from pytorch_pipeline.train.dataset import UncachedPhenologyDataset
from pytorch_pipeline.utils.params import DatasetParams


class DummyMetrics:
    def get_per_class(self, metric: str) -> list[float]:
        if metric == "best_thresh":
            return [0.5, 0.4]
        return [0.5, 0.5]


def test_error_analysis_with_paths():
    all_obs_ids = [101, 102, 103]
    all_preds_raw = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.8]])
    all_labels = np.array([[0, 1], [1, 0], [1, 1]])

    obs_attention_weights = {
        "Flowering": [
            torch.tensor([[0.7], [0.3]]),
            torch.tensor([[0.5], [0.5]]),
            torch.tensor([[0.2], [0.8]]),
        ],
        "Fruiting": [
            torch.tensor([[0.4], [0.6]]),
            torch.tensor([[0.9], [0.1]]),
            torch.tensor([[0.3], [0.7]]),
        ],
    }
    metrics = DummyMetrics()
    all_obs_paths = {
        101: ["/path/img1_1.jpg", "/path/img1_2.jpg"],
        102: ["/path/img2_1.jpg", "/path/img2_2.jpg"],
        103: ["/path/img3_1.jpg", "/path/img3_2.jpg"],
    }

    report = error_analysis(
        all_obs_ids=all_obs_ids,
        all_preds_raw=all_preds_raw,
        all_labels=all_labels,
        observations_attention_weights=obs_attention_weights,
        metrics=metrics,
        all_obs_paths=all_obs_paths,
    )

    assert "Flowering" in report
    assert "Fruiting" in report
    # Obs 101: pred Flowering=0.8 >= 0.5 (1), label=0 -> False Positive
    fp_flowers = report["Flowering"]["fp"]
    assert len(fp_flowers) == 1
    assert fp_flowers[0]["obs_id"] == 101
    assert fp_flowers[0]["prob"] == 0.8
    assert fp_flowers[0]["target"] == 0
    assert fp_flowers[0]["threshold"] == 0.5
    assert fp_flowers[0]["paths"] == ["/path/img1_1.jpg", "/path/img1_2.jpg"]


def test_dataset_get_obs_paths_map():
    df = pd.DataFrame(
        {
            "observation_id": [101, 102],
            "path": [["/p/1.jpg", "/p/2.jpg"], ["/p/3.jpg"]],
            "label": [[1, 0, 0], [0, 1, 0]],
        }
    )
    dataset = UncachedPhenologyDataset(
        df, transform=None, dataset_params=DatasetParams()
    )
    obs_paths_map = dataset.get_obs_paths_map()
    assert obs_paths_map == {
        101: ["/p/1.jpg", "/p/2.jpg"],
        102: ["/p/3.jpg"],
    }


def test_resolve_report_paths_from_dataframe():
    report = {
        "flowers": {
            "fp": [{"obs_id": 201, "weights": [0.6, 0.4]}],
            "fn": [{"obs_id": 202, "weights": [1.0]}],
        }
    }
    df = pd.DataFrame(
        {
            "observation_id": [201, 202],
            "path": [["/images/201_a.jpg", "/images/201_b.jpg"], ["/images/202_a.jpg"]],
        }
    )

    resolved = resolve_report_paths(report, dataset_df=df)
    assert resolved["flowers"]["fp"][0]["paths"] == [
        "/images/201_a.jpg",
        "/images/201_b.jpg",
    ]
    assert resolved["flowers"]["fn"][0]["paths"] == ["/images/202_a.jpg"]


def test_resolve_report_paths_from_duckdb(tmp_path):
    db_file = tmp_path / "test.duckdb"
    with duckdb.connect(str(db_file)) as con:
        con.execute(
            """
            CREATE TABLE my_table (
                observation_id BIGINT,
                photo_id BIGINT
            );
            INSERT INTO my_table VALUES (301, 1001), (301, 1002);
            """
        )

    report = {
        "flowers": {
            "fp": [{"obs_id": 301, "weights": [0.5, 0.5]}],
            "fn": [],
        }
    }

    resolved = resolve_report_paths(
        report, db_path=db_file, table_name="my_table", image_dir="/data/images"
    )
    assert resolved["flowers"]["fp"][0]["paths"] == [
        "/data/images/1001.jpg",
        "/data/images/1002.jpg",
    ]


def test_plot_misclassified_observation(tmp_path):
    obs_entry = {
        "obs_id": 401,
        "weights": [0.7, 0.3],
        "prob": 0.85,
        "target": 0,
        "threshold": 0.5,
        "paths": ["/nonexistent/img1.jpg", "/nonexistent/img2.jpg"],
    }

    save_path = tmp_path / "plot.png"
    fig, axes = plot_misclassified_observation(
        obs_entry, label_name="flowers", error_type="fp", save_path=save_path
    )
    assert fig is not None
    assert save_path.exists()


def test_review_misclassifications():
    report = {
        "flowers": {
            "fp": [
                {
                    "obs_id": 501,
                    "weights": [0.8, 0.2],
                    "prob": 0.9,
                    "target": 0,
                    "threshold": 0.5,
                }
            ],
            "fn": [],
        }
    }
    result = review_misclassifications(report)
    assert result == report


def test_resolve_report_paths_rebase_colab_paths():
    report = {
        "Flowering": {
            "fp": [
                {
                    "obs_id": 601,
                    "weights": [0.8],
                    "paths": [
                        "/content/data/images/10001.jpg",
                        "/content/data/images/10002.jpg",
                    ],
                }
            ],
            "fn": [],
        }
    }
    resolved = resolve_report_paths(
        report, image_dir="/home/etienne/projects/inat-phenology-cv/data/images"
    )
    assert resolved["Flowering"]["fp"][0]["paths"] == [
        "/home/etienne/projects/inat-phenology-cv/data/images/10001.jpg",
        "/home/etienne/projects/inat-phenology-cv/data/images/10002.jpg",
    ]


def test_review_label_issues_returns_entries():
    """review_label_issues resolves paths and returns entry dicts."""
    from pytorch_pipeline.review import review_label_issues

    obs_ids = [601, 602]
    df = pd.DataFrame(
        {
            "observation_id": [601, 602],
            "path": [["/img/601a.jpg", "/img/601b.jpg"], ["/img/602a.jpg"]],
        }
    )
    result = review_label_issues(obs_ids, label_name="Flowering", dataset_df=df)

    assert len(result) == 2
    assert result[0]["obs_id"] == 601
    assert result[0]["paths"] == ["/img/601a.jpg", "/img/601b.jpg"]
    assert result[1]["obs_id"] == 602
    assert result[1]["paths"] == ["/img/602a.jpg"]


def test_review_label_issues_empty():
    """review_label_issues handles an empty obs_ids list gracefully."""
    from pytorch_pipeline.review import review_label_issues

    result = review_label_issues([])
    assert result == []


def test_review_label_issues_no_paths():
    """review_label_issues returns entries with empty paths when no source given."""
    from pytorch_pipeline.review import review_label_issues

    result = review_label_issues([999])
    assert len(result) == 1
    assert result[0]["obs_id"] == 999
    assert result[0].get("paths", []) == []


def test_review_label_issues_preserves_order():
    """review_label_issues preserves the ranked order of obs_ids."""
    from pytorch_pipeline.review import review_label_issues

    obs_ids = [700, 701, 702]
    df = pd.DataFrame(
        {
            "observation_id": [700, 701, 702],
            "path": [["/img/700.jpg"], ["/img/701.jpg"], ["/img/702.jpg"]],
        }
    )
    result = review_label_issues(obs_ids, dataset_df=df)

    assert [r["obs_id"] for r in result] == [700, 701, 702]


def test_review_label_issues_with_prob_and_target():
    """review_label_issues enriches entries with prob and
    target when inference outputs provided."""
    from pytorch_pipeline.review import review_label_issues

    all_obs_ids = [800, 801, 802]
    raw_labels = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    raw_preds = np.array([[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.7, 0.6, 0.05]])

    flagged_obs_ids = [802, 800]  # cleanlab ranked order
    df = pd.DataFrame(
        {
            "observation_id": [800, 801, 802],
            "path": [["/img/800.jpg"], ["/img/801.jpg"], ["/img/802.jpg"]],
        }
    )

    result = review_label_issues(
        obs_ids=flagged_obs_ids,
        label_name="Flowering",
        class_idx=0,
        all_obs_ids=all_obs_ids,
        raw_labels=raw_labels,
        raw_preds=raw_preds,
        dataset_df=df,
    )

    assert len(result) == 2
    # First entry: obs_id=802, class_idx=0 -> label=1, pred=0.7
    assert result[0]["obs_id"] == 802
    assert result[0]["target"] == 1
    assert abs(result[0]["prob"] - 0.7) < 1e-6
    # Second entry: obs_id=800, class_idx=0 -> label=1, pred=0.9
    assert result[1]["obs_id"] == 800
    assert result[1]["target"] == 1
    assert abs(result[1]["prob"] - 0.9) < 1e-6

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from plant_pheno.cleanlab import (
    export_label_issues_csv,
    run_cleanlab_analysis,
    save_oof_predictions_to_duckdb,
    stratified_multilabel_kfold,
)
from plant_pheno.train.dataset import get_mean_img_ratio


@pytest.fixture
def mock_multilabel_df() -> pd.DataFrame:
    np.random.seed(42)
    n_obs = 100
    rows = []
    # 3 classes: budding, flowering, fruiting
    possible_labels = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]
    for obs_id in range(1, n_obs + 1):
        label = possible_labels[obs_id % len(possible_labels)]
        n_photos = (obs_id % 3) + 1
        for p in range(n_photos):
            rows.append(
                {
                    "observation_id": obs_id,
                    "photo_id": f"photo_{obs_id}_{p}",
                    "label": label,
                    "path": f"/images/{obs_id}_{p}.jpg",
                }
            )
    return pd.DataFrame(rows)


def test_get_mean_img_ratio_with_grouped_paths(tmp_path: Path):
    portrait_path = tmp_path / "portrait.png"
    square_path = tmp_path / "square.png"
    Image.new("RGB", (10, 20)).save(portrait_path)
    Image.new("RGB", (10, 10)).save(square_path)

    df = pd.DataFrame({"path": [[portrait_path, square_path]]})

    assert get_mean_img_ratio(df) == 1.5


def test_stratified_multilabel_kfold(mock_multilabel_df: pd.DataFrame):
    n_splits = 5
    folds = stratified_multilabel_kfold(
        mock_multilabel_df,
        idx_col="observation_id",
        label_col="label",
        n_splits=n_splits,
        seed=42,
    )
    assert len(folds) == n_splits

    all_val_obs = []
    unique_obs_count = mock_multilabel_df["observation_id"].nunique()

    for train_df, val_df in folds:
        train_obs = set(train_df["observation_id"])
        val_obs = set(val_df["observation_id"])

        # No overlap between train and val within a fold
        assert len(train_obs.intersection(val_obs)) == 0
        # Observation count sums to total
        assert len(train_obs) + len(val_obs) == unique_obs_count
        all_val_obs.extend(list(val_obs))

    # All observations were validated exactly once across the K folds
    assert len(all_val_obs) == unique_obs_count
    assert len(set(all_val_obs)) == unique_obs_count


def test_save_oof_predictions_to_duckdb():
    obs_ids = [101, 102, 103]
    labels = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    preds = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.9, 0.6, 0.05]])

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_cleanlab.duckdb"
        save_oof_predictions_to_duckdb(
            db_path=db_path,
            obs_ids=obs_ids,
            labels=labels,
            preds=preds,
            table_name="oof_predictions",
        )

        assert db_path.exists()
        with duckdb.connect(str(db_path)) as con:
            res = con.execute(
                "SELECT obs_id, labels, preds FROM oof_predictions ORDER BY obs_id"
            ).fetchall()
            assert len(res) == 3
            assert res[0][0] == 101
            assert np.allclose(res[0][1], [1.0, 0.0, 0.0])
            assert np.allclose(res[0][2], [0.8, 0.1, 0.1])

            # Check view
            res_view = con.execute("SELECT COUNT(*) FROM test_set").fetchone()
            assert res_view[0] == 3


def test_run_cleanlab_analysis():
    # Construct synthetic data with an intentional label issue
    obs_ids = [101, 102, 103, 104, 105, 106, 107, 108]
    # Obs 101 has label 0 for class 0, but prediction is 0.99 (likely mislabeled)
    labels = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    preds = np.array(
        [
            [0.99, 0.95, 0.05],
            [0.90, 0.05, 0.05],
            [0.85, 0.88, 0.10],
            [0.10, 0.10, 0.92],
            [0.89, 0.08, 0.85],
            [0.05, 0.91, 0.87],
            [0.92, 0.89, 0.95],
            [0.02, 0.03, 0.04],
        ]
    )

    results = run_cleanlab_analysis(
        obs_ids=obs_ids,
        labels=labels,
        preds=preds,
        class_names=["budding", "flowering", "fruiting"],
    )

    assert "per_class_issues" in results
    assert "per_class_issue_rates" in results
    assert "quality_scores" in results
    assert len(results["quality_scores"]) == len(obs_ids)

    # Class 0 should flag obs 101
    assert 101 in results["per_class_issues"]["budding"]


def test_export_label_issues_csv():
    all_obs_ids = [101, 102, 103]
    labels = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    preds = np.array([[0.95, 0.90, 0.05], [0.85, 0.10, 0.10], [0.90, 0.80, 0.10]])
    ranked_obs_ids = [101]

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = export_label_issues_csv(
            corrections_dir=tmp_dir,
            class_name="budding",
            class_idx=0,
            ranked_obs_ids=ranked_obs_ids,
            all_obs_ids=all_obs_ids,
            labels=labels,
            preds=preds,
            version=1,
        )

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["observation_id"] == 101
        assert df.iloc[0]["label_error_id"] == 0
        assert df.iloc[0]["link"] == "https://www.inaturalist.ca/observations/101"
        assert "cleanlab kfold (budding)" in df.iloc[0]["notes"]

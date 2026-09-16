import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from plant_pheno.data import get_model_evaluations, log_model_evaluation
from plant_pheno.train.metrics import EpochMetrics


@pytest.fixture
def sample_epoch_metrics():
    return EpochMetrics(
        roc_auc_macro=0.915,
        pr_auc_macro=0.884,
        pr_norm_excess_macro=0.452,
        f1_macro_best=0.823,
        f1_macro_05=0.795,
        f1_micro_05=0.812,
        f1_weighted_05=0.808,
        exact_match_ratio=0.750,
        hamming_loss=0.085,
        val_loss=0.245,
        roc_auc={"Flowering": 0.94, "Fruiting": 0.89},
        pr_auc={"Flowering": 0.91, "Fruiting": 0.85},
        pr_norm_excess={"Flowering": 0.50, "Fruiting": 0.40},
        best_thresh={"Flowering": 0.45, "Fruiting": 0.52},
        best_f1={"Flowering": 0.85, "Fruiting": 0.80},
        best_prec={"Flowering": 0.88, "Fruiting": 0.82},
        best_recall={"Flowering": 0.82, "Fruiting": 0.78},
        f1_05={"Flowering": 0.83, "Fruiting": 0.76},
        precision_05={"Flowering": 0.85, "Fruiting": 0.80},
        recall_05={"Flowering": 0.81, "Fruiting": 0.73},
        support_pos={"Flowering": 120, "Fruiting": 95},
        support_neg={"Flowering": 380, "Fruiting": 405},
        prefix="val",
    )


def test_log_model_evaluation_with_epoch_metrics(sample_epoch_metrics):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_evals.duckdb"
        dataset_path = "/path/to/data/cv_photos4.parquet"

        eval_id = log_model_evaluation(
            eval_db_path=db_path,
            model_name="cv_inat_test",
            model_version=2,
            dataset_path=dataset_path,
            dataset_table="cv_photos4",
            seed=42,
            is_test_mode=True,
            metrics=sample_epoch_metrics,
        )

        assert eval_id is not None
        assert len(eval_id) > 0

        df = get_model_evaluations(db_path)
        assert not df.empty
        assert len(df) == 1

        row = df.iloc[0]
        assert row["eval_id"] == eval_id
        assert row["model_name"] == "cv_inat_test"
        assert str(row["model_version"]) == "2"
        assert row["dataset_file"] == "cv_photos4.parquet"
        assert row["dataset_path"] == dataset_path
        assert row["dataset_table"] == "cv_photos4"
        assert row["seed"] == 42
        assert bool(row["is_test_mode"]) is True

        # Check scalar metrics
        assert pytest.approx(row["roc_auc_macro"], rel=1e-3) == 0.915
        assert pytest.approx(row["pr_auc_macro"], rel=1e-3) == 0.884
        assert pytest.approx(row["f1_macro_best"], rel=1e-3) == 0.823
        assert pytest.approx(row["val_loss"], rel=1e-3) == 0.245

        # Check dynamic per-class columns
        assert "pr_auc_Flowering" in df.columns
        assert "pr_norm_excess_Flowering" in df.columns
        assert "pr_auc_Fruiting" in df.columns
        assert "pr_norm_excess_Fruiting" in df.columns

        assert pytest.approx(row["pr_auc_Flowering"], rel=1e-3) == 0.91
        assert pytest.approx(row["pr_norm_excess_Flowering"], rel=1e-3) == 0.50
        assert pytest.approx(row["pr_auc_Fruiting"], rel=1e-3) == 0.85
        assert pytest.approx(row["pr_norm_excess_Fruiting"], rel=1e-3) == 0.40

        # Check per-class JSON
        per_class_dict = json.loads(row["per_class_metrics_json"])
        assert "roc_auc" in per_class_dict
        assert per_class_dict["roc_auc"]["Flowering"] == 0.94


def test_dynamic_columns_when_new_classes_arise(sample_epoch_metrics):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_evals.duckdb"

        # Log run 1 with standard 2 classes (Flowering, Fruiting)
        log_model_evaluation(
            eval_db_path=db_path,
            model_name="cv_inat_test",
            model_version="1",
            dataset_path="/data/v1.parquet",
            dataset_table="v1",
            seed=42,
            is_test_mode=False,
            metrics=sample_epoch_metrics,
        )

        # Create a 3-class metrics object including new class 'Budding'
        new_epoch_metrics = EpochMetrics(
            roc_auc_macro=0.930,
            pr_auc_macro=0.890,
            pr_norm_excess_macro=0.480,
            f1_macro_best=0.840,
            f1_macro_05=0.810,
            f1_micro_05=0.820,
            f1_weighted_05=0.815,
            exact_match_ratio=0.770,
            hamming_loss=0.075,
            val_loss=0.210,
            roc_auc={"Flowering": 0.95, "Fruiting": 0.90, "Budding": 0.92},
            pr_auc={"Flowering": 0.92, "Fruiting": 0.86, "Budding": 0.89},
            pr_norm_excess={"Flowering": 0.52, "Fruiting": 0.42, "Budding": 0.47},
            best_thresh={"Flowering": 0.45, "Fruiting": 0.52, "Budding": 0.48},
            best_f1={"Flowering": 0.86, "Fruiting": 0.81, "Budding": 0.83},
            best_prec={"Flowering": 0.89, "Fruiting": 0.83, "Budding": 0.85},
            best_recall={"Flowering": 0.83, "Fruiting": 0.79, "Budding": 0.81},
            f1_05={"Flowering": 0.84, "Fruiting": 0.77, "Budding": 0.80},
            precision_05={"Flowering": 0.86, "Fruiting": 0.81, "Budding": 0.83},
            recall_05={"Flowering": 0.82, "Fruiting": 0.74, "Budding": 0.77},
            support_pos={"Flowering": 125, "Fruiting": 100, "Budding": 80},
            support_neg={"Flowering": 375, "Fruiting": 400, "Budding": 420},
            prefix="val",
        )

        # Log run 2 with the new class
        log_model_evaluation(
            eval_db_path=db_path,
            model_name="cv_inat_test",
            model_version="2",
            dataset_path="/data/v2.parquet",
            dataset_table="v2",
            seed=42,
            is_test_mode=False,
            metrics=new_epoch_metrics,
        )

        df = get_model_evaluations(db_path)
        assert len(df) == 2

        # Verify dynamic columns for Budding were added
        assert "pr_auc_Budding" in df.columns
        assert "pr_norm_excess_Budding" in df.columns

        # Version 2 should have valid values for Budding
        v2_row = df[df["model_version"] == "2"].iloc[0]
        assert pytest.approx(v2_row["pr_auc_Budding"], rel=1e-3) == 0.89
        assert pytest.approx(v2_row["pr_norm_excess_Budding"], rel=1e-3) == 0.47

        # Version 1 should have NaN / None for newly added Budding columns
        v1_row = df[df["model_version"] == "1"].iloc[0]
        assert pd.isna(v1_row["pr_auc_Budding"])

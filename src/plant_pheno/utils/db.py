from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pandas as pd

if TYPE_CHECKING:
    from ..train.metrics import EpochMetrics


CREATE_EVALUATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_evaluations (
    eval_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP,
    model_name VARCHAR,
    model_version VARCHAR,
    dataset_file VARCHAR,
    dataset_path VARCHAR,
    dataset_table VARCHAR,
    seed INTEGER,
    is_test_mode BOOLEAN,
    roc_auc_macro DOUBLE,
    pr_auc_macro DOUBLE,
    pr_norm_excess_macro DOUBLE,
    f1_macro_best DOUBLE,
    f1_macro_05 DOUBLE,
    f1_micro_05 DOUBLE,
    f1_weighted_05 DOUBLE,
    exact_match_ratio DOUBLE,
    hamming_loss DOUBLE,
    val_loss DOUBLE,
    per_class_metrics_json VARCHAR
);
"""


def get_df_from_table(db_path: str, table_name: str) -> pd.DataFrame:
    with duckdb.connect(db_path) as con:
        df = con.execute(f"SELECT * FROM {table_name}").fetch_df()
    return df


def log_model_evaluation(
    eval_db_path: str | Path,
    model_name: str,
    model_version: str | int,
    dataset_path: str | Path,
    dataset_table: str,
    seed: int,
    is_test_mode: bool,
    metrics: EpochMetrics | dict[str, Any],
) -> str:
    """Commit evaluation results for a model version and dataset version into DuckDB.

    Dynamically alters the table schema to add per-class metric columns
    (e.g., pr_auc_Flowering, pr_norm_excess_Flowering) as new classes arise.

    Returns:
        eval_id (str): The unique UUID of the logged evaluation record.
    """
    eval_db_path = Path(eval_db_path)
    if eval_db_path.parent:
        eval_db_path.parent.mkdir(parents=True, exist_ok=True)

    eval_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    dataset_path_str = str(dataset_path)
    dataset_file_str = Path(dataset_path).name

    if hasattr(metrics, "roc_auc_macro"):
        roc_auc_macro = float(metrics.roc_auc_macro)
        pr_auc_macro = float(metrics.pr_auc_macro)
        pr_norm_excess_macro = float(metrics.pr_norm_excess_macro)
        f1_macro_best = float(metrics.f1_macro_best)
        f1_macro_05 = float(metrics.f1_macro_05)
        f1_micro_05 = float(metrics.f1_micro_05)
        f1_weighted_05 = float(metrics.f1_weighted_05)
        exact_match_ratio = float(metrics.exact_match_ratio)
        hamming_loss = float(metrics.hamming_loss)
        val_loss = float(metrics.val_loss)

        per_class_data = {
            "roc_auc": getattr(metrics, "roc_auc", {}),
            "pr_auc": getattr(metrics, "pr_auc", {}),
            "pr_norm_excess": getattr(metrics, "pr_norm_excess", {}),
            "best_thresh": getattr(metrics, "best_thresh", {}),
            "best_f1": getattr(metrics, "best_f1", {}),
            "best_prec": getattr(metrics, "best_prec", {}),
            "best_recall": getattr(metrics, "best_recall", {}),
            "f1_05": getattr(metrics, "f1_05", {}),
            "precision_05": getattr(metrics, "precision_05", {}),
            "recall_05": getattr(metrics, "recall_05", {}),
            "support_pos": getattr(metrics, "support_pos", {}),
            "support_neg": getattr(metrics, "support_neg", {}),
        }
    else:
        roc_auc_macro = float(metrics.get("roc_auc_macro", 0.0))
        pr_auc_macro = float(metrics.get("pr_auc_macro", 0.0))
        pr_norm_excess_macro = float(metrics.get("pr_norm_excess_macro", 0.0))
        f1_macro_best = float(metrics.get("f1_macro_best", 0.0))
        f1_macro_05 = float(metrics.get("f1_macro_05", 0.0))
        f1_micro_05 = float(metrics.get("f1_micro_05", 0.0))
        f1_weighted_05 = float(metrics.get("f1_weighted_05", 0.0))
        exact_match_ratio = float(metrics.get("exact_match_ratio", 0.0))
        hamming_loss = float(metrics.get("hamming_loss", 0.0))
        val_loss = float(metrics.get("val_loss", 0.0))
        per_class_data = metrics.get("per_class", {})

    per_class_json = json.dumps(per_class_data)

    # Base payload dict mapping column_name -> value
    payload: dict[str, Any] = {
        "eval_id": eval_id,
        "timestamp": now,
        "model_name": str(model_name),
        "model_version": str(model_version),
        "dataset_file": dataset_file_str,
        "dataset_path": dataset_path_str,
        "dataset_table": str(dataset_table),
        "seed": int(seed),
        "is_test_mode": bool(is_test_mode),
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
        "pr_norm_excess_macro": pr_norm_excess_macro,
        "f1_macro_best": f1_macro_best,
        "f1_macro_05": f1_macro_05,
        "f1_micro_05": f1_micro_05,
        "f1_weighted_05": f1_weighted_05,
        "exact_match_ratio": exact_match_ratio,
        "hamming_loss": hamming_loss,
        "val_loss": val_loss,
        "per_class_metrics_json": per_class_json,
    }

    # Dynamically extract per-class scalar entries
    #  (e.g. pr_auc_Flowering, pr_norm_excess_Flowering)
    dynamic_column_types: dict[str, str] = {}
    for metric_name, class_dict in per_class_data.items():
        if not isinstance(class_dict, dict):
            continue
        for class_name, val in class_dict.items():
            clean_cls = str(class_name).strip().replace(" ", "_").replace("-", "_")
            col_name = f"{metric_name}_{clean_cls}"

            if isinstance(val, (int, bool)) and not isinstance(val, bool):
                sql_type = "BIGINT"
                typed_val = int(val)
            else:
                sql_type = "DOUBLE"
                typed_val = float(val) if val is not None else None

            payload[col_name] = typed_val
            dynamic_column_types[col_name] = sql_type

    with duckdb.connect(str(eval_db_path)) as con:
        con.execute(CREATE_EVALUATIONS_TABLE_SQL)

        # Inspect current table schema
        existing_cols = {
            row[1]
            for row in con.execute("PRAGMA table_info('model_evaluations')").fetchall()
        }

        # Dynamically add any missing per-class columns to the table schema
        for col_name, sql_type in dynamic_column_types.items():
            if col_name not in existing_cols:
                con.execute(
                    f'ALTER TABLE model_evaluations ADD COLUMN "{col_name}" {sql_type}'
                )
                existing_cols.add(col_name)

        # Insert record with all columns
        cols = list(payload.keys())
        vals = list(payload.values())
        col_sql = ", ".join([f'"{c}"' for c in cols])
        placeholders = ", ".join(["?"] * len(cols))

        con.execute(
            f"INSERT INTO model_evaluations ({col_sql}) VALUES ({placeholders})", vals
        )
        con.close()

    return eval_id


def get_model_evaluations(eval_db_path: str | Path) -> pd.DataFrame:
    """Fetch all recorded model evaluations from DuckDB as a Pandas DataFrame."""
    eval_db_path = Path(eval_db_path)
    if not eval_db_path.exists():
        return pd.DataFrame()
    with duckdb.connect(str(eval_db_path)) as con:
        try:
            return con.execute(
                "SELECT * FROM model_evaluations ORDER BY timestamp DESC"
            ).fetch_df()
        except duckdb.CatalogException:
            return pd.DataFrame()

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold
from torch import nn

from .config import (
    CLASS_ORDER,
    cfg_to_runtime,
    model_params_from_cfg,
    optimizer_params_from_cfg,
    scheduler_params_from_cfg,
    training_params_from_cfg,
)
from .core import build_pipeline_model
from .core.backbone import BACKBONE_REGISTRY
from .core.device import get_device
from .infra import resolve_uri, seed_everything
from .train import (
    build_pipeline_dataloaders,
    build_pipeline_optimizer,
    build_scheduler,
    workflow,
)
from .train.dataset import (
    PhenologyDataset,
    get_mean_img_ratio,
    get_samples,
    resolve_cache_decision,
)
from .utils import get_pos_ratios, get_pos_weights
from .val import execute as val_execute

logger = logging.getLogger(__name__)


def stratified_multilabel_kfold(
    df: pd.DataFrame,
    idx_col: str = "observation_id",
    label_col: str = "label",
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Splits an observation DataFrame into K stratified folds
      based on multi-label combinations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rows grouped by observation (or individual photo rows).
    idx_col : str
        Column name for unique observation ID.
    label_col : str
        Column name containing the multi-label array/list.
    n_splits : int
        Number of CV folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (train_df, val_df) tuples for each fold.
    """
    obs_df = (
        df.groupby(idx_col)
        .agg({label_col: "first"})
        .reset_index(drop=False)
        .sort_values(by=idx_col)
        .reset_index(drop=True)
    )

    def _to_combo(label_val: Any) -> str:
        arr = np.array(label_val).astype(int).tolist()
        return "_".join(str(x) for x in arr)

    combos = obs_df[label_col].apply(_to_combo)
    combo_counts = combos.value_counts()
    rare_combos = set(combo_counts[combo_counts < n_splits].index)

    strat_labels = combos.apply(lambda c: "rare" if c in rare_combos else c)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []

    for train_indices, val_indices in skf.split(obs_df, strat_labels):
        train_obs_ids = set(obs_df.iloc[train_indices][idx_col])
        val_obs_ids = set(obs_df.iloc[val_indices][idx_col])

        train_df = (
            df[df[idx_col].isin(train_obs_ids)]
            .sort_values(by=idx_col)
            .reset_index(drop=True)
        )
        val_df = (
            df[df[idx_col].isin(val_obs_ids)]
            .sort_values(by=idx_col)
            .reset_index(drop=True)
        )
        folds.append((train_df, val_df))

    return folds


def save_oof_predictions_to_duckdb(
    db_path: str | Path,
    obs_ids: list[int],
    labels: np.ndarray,
    preds: np.ndarray,
    table_name: str = "oof_predictions",
) -> None:
    """Persists out-of-fold predictions to DuckDB tables
    (both given table_name and test_set)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db_path)) as con:
        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} (
                obs_id BIGINT,
                labels DOUBLE[],
                preds DOUBLE[]
            )
        """)

        rows = [
            (int(i), label.tolist(), pred.tolist())
            for i, label, pred in zip(obs_ids, labels, preds)
        ]

        con.executemany(
            f"""
            INSERT INTO {table_name} (obs_id, labels, preds)
            VALUES (?, ?, ?)
        """,
            rows,
        )

        if table_name != "test_set":
            con.execute(
                f"CREATE OR REPLACE VIEW test_set AS SELECT * FROM {table_name}"
            )

    logger.info(
        f"Saved {len(obs_ids)} out-of-fold predictions to {db_path} ({table_name})"
    )


def run_cleanlab_analysis(
    obs_ids: list[int],
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Runs cleanlab label issue analysis per class and
    across the multi-label dataset."""
    from cleanlab.filter import find_label_issues
    from cleanlab.multilabel_classification import get_label_quality_scores

    if class_names is None:
        class_names = CLASS_ORDER

    n_classes = labels.shape[1]
    per_class_issues: dict[str, list[int]] = {}
    per_class_issue_rates: dict[str, float] = {}
    obs_ids_arr = np.array(obs_ids)

    for i in range(n_classes):
        c_name = class_names[i]
        labels_c = labels[:, i].astype(int)
        pred_probs_pos = preds[:, i]
        pred_probs_neg = 1.0 - pred_probs_pos
        pred_probs = np.column_stack([pred_probs_neg, pred_probs_pos])

        if len(np.unique(labels_c)) > 1:
            issue_mask = find_label_issues(labels=labels_c, pred_probs=pred_probs)
            issue_rate = float(issue_mask.mean())
            ranked_indices = find_label_issues(
                labels=labels_c,
                pred_probs=pred_probs,
                return_indices_ranked_by="self_confidence",
            )
            per_class_issues[c_name] = [int(obs_ids_arr[idx]) for idx in ranked_indices]
            per_class_issue_rates[c_name] = issue_rate
        else:
            per_class_issues[c_name] = []
            per_class_issue_rates[c_name] = 0.0

    labels_list = labels.astype(int).tolist()
    quality_scores = get_label_quality_scores(labels=labels_list, pred_probs=preds)

    return {
        "per_class_issues": per_class_issues,
        "per_class_issue_rates": per_class_issue_rates,
        "quality_scores": quality_scores,
    }


def export_label_issues_csv(
    corrections_dir: str | Path,
    class_name: str,
    class_idx: int,
    ranked_obs_ids: list[int],
    all_obs_ids: list[int],
    labels: np.ndarray,
    preds: np.ndarray,
    version: int = 1,
) -> Path:
    """Exports ranked cleanlab issues to a CSV formatted for human review."""
    corrections_dir = Path(corrections_dir)
    corrections_dir.mkdir(parents=True, exist_ok=True)

    csv_path = corrections_dir / f"cleanlab_issues_{class_name}.csv"

    obs_idx_map = {obs_id: i for i, obs_id in enumerate(all_obs_ids)}
    rows = []
    for rank, obs_id in enumerate(ranked_obs_ids):
        i = obs_idx_map.get(obs_id)
        given_label = int(labels[i, class_idx]) if i is not None else None
        pred_prob = round(float(preds[i, class_idx]), 4) if i is not None else None
        rows.append(
            {
                "observation_id": obs_id,
                "label_error_id": rank,
                "given_label": given_label,
                "pred_prob": pred_prob,
                "correction": "",
                "link": f"https://www.inaturalist.ca/observations/{obs_id}",
                "notes": f"cleanlab kfold ({class_name})",
                "version": version,
            }
        )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Exported {len(rows)} issues for {class_name} to {csv_path}")
    return csv_path


def load_cleanlab_config(config_path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file for the cleanlab CV pipeline."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_cleanlab_cv(
    config_path: str | Path,
    test_mode: bool = False,
    test_fraction: float = 0.05,
) -> dict[str, Any]:
    """Executes the full K-Fold cross-validation loop and
    runs Cleanlab issue detection."""
    import mlflow

    config_path = Path(config_path)
    cfg_dict = load_cleanlab_config(config_path)

    paths_cfg = cfg_dict.get("paths", {})
    model_cfg = cfg_dict.get("model_params", {})
    train_cfg = cfg_dict.get("training_params", {})
    loader_cfg = cfg_dict.get("dataloader_params", {})
    dataset_cfg = cfg_dict.get("dataset_params", {})
    cleanlab_cfg = cfg_dict.get("cleanlab_params", {})

    seed = train_cfg.get("seed", 42)
    seed_everything(seed, set_cuda_deterministic=False)

    paths_params = PathsParams(**paths_cfg)
    dataset_params = DatasetParams(**dataset_cfg)
    dataloader_params = DataLoadersParams(**loader_cfg)
    hardware_profile = resolve_hardware_profile()

    config = Config(
        config_path=config_path,
        paths_params=paths_params,
        dataset_params=dataset_params,
        dataloaders_params=dataloader_params,
        hardware_profile=hardware_profile,
        git_branch=get_current_git_branch(),
        git_hash=get_git_hash(),
    )
    config.test = test_mode

    device = get_device()
    logger.info(f"Using device: {device.type}")

    model_params = ModelParams(
        backbone=model_cfg.get("backbone", "efficientnet"),
        architecture=model_cfg.get("architecture", "mil_pooling"),
        head_neurons=model_cfg.get("head_neurons", 256),
        head_outputs=model_cfg.get("head_outputs", 1),
        head_dropout_prob=model_cfg.get("head_dropout_prob", 0.5),
        attention_neurons=model_cfg.get("attention_neurons", 128),
        attention_dropout_prob=model_cfg.get("attention_dropout_prob", 0.1),
        start_unfreezed=model_cfg.get("start_unfreezed", 0),
        gated=model_cfg.get("gated", True),
    )

    n_folds = cleanlab_cfg.get("n_folds", 5)
    db_table_name = cleanlab_cfg.get("table_name", "oof_predictions")
    cleanlab_db_path = paths_cfg.get("cleanlab_db_path", "data/cleanlab.duckdb")
    corrections_dir = paths_cfg.get("corrections_dir", "data/label_corrections")
    checkpoint_dir = Path(paths_params.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading full dataset samples...")
    raw_df = get_samples(paths_params, dataset_params)

    # Collapse df by observation
    df = (
        raw_df.groupby(dataset_params.idx_col)
        .agg({"path": list, dataset_params.label_col: "first"})
        .reset_index(drop=False)
        .sort_values(by=dataset_params.idx_col)
        .reset_index(drop=True)
    )

    if test_mode:
        sample_size = max(10, int(len(df) * test_fraction))
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        logger.info(f"Running in test mode: subsampled to {len(df)} observations")

    mean_image_ratio = get_mean_img_ratio(df)
    cache = resolve_cache_decision(
        hardware_profile, len(df), mean_image_ratio, config.max_img_resolution
    )

    transforms = BACKBONE_REGISTRY[model_params.backbone]().get_transforms()
    train_transform, val_transform = transforms

    folds = stratified_multilabel_kfold(
        df=df,
        idx_col=dataset_params.idx_col,
        label_col=dataset_params.label_col,
        n_splits=n_folds,
        seed=seed,
    )

    mlflow.set_tracking_uri(resolve_uri())

    all_oof_obs_ids: list[int] = []
    all_oof_labels: list[np.ndarray] = []
    all_oof_preds: list[np.ndarray] = []

    with mlflow.start_run(run_name="cleanlab_cv"):
        mlflow.log_params(
            {
                "n_folds": n_folds,
                "backbone": model_params.backbone,
                "architecture": model_params.architecture,
                "epochs": train_cfg.get("epochs", 5),
                "seed": seed,
                "total_observations": len(df),
            }
        )

        for fold_idx, (train_df, val_df) in enumerate(folds):
            logger.info(f"\n{'=' * 25} FOLD {fold_idx + 1}/{n_folds} {'=' * 25}")

            train_set = PhenologyDataset.from_profile(
                cache, train_df, train_transform, dataset_params
            )
            val_set = PhenologyDataset.from_profile(
                cache, val_df, val_transform, dataset_params
            )

            train_loader, val_loader, _ = build_pipeline_dataloaders(
                (train_set, val_set, val_set), dataloader_params, seed=seed
            )

            pos_weights = get_pos_weights(train_set, dataset_params, device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction="none")
            pos_ratios = get_pos_ratios(val_set)

            training_params = TrainingParams(
                epochs=train_cfg.get("epochs", 5),
                stopping_patience=train_cfg.get("stopping_patience", 3),
                unfreeze=train_cfg.get("unfreeze", False),
                unfreezing_patience=train_cfg.get("unfreezing_patience", 2),
                unfreezing_cooldown=train_cfg.get("unfreezing_cooldown", 1),
                starting_block=train_cfg.get("starting_block", 0),
                max_stages=train_cfg.get("max_stages", 1),
                block_per_stage=train_cfg.get("block_per_stage", 1),
                best_objective=train_cfg.get("best_objective", 1e-5),
                seed=seed + fold_idx,
                pos_ratios=pos_ratios,
            )

            optim_params = OptimizerParams(base_lr=train_cfg.get("base_lr", 1e-3))
            scheduler_params = SchedulerParams(
                warmup_epochs=train_cfg.get("warmup_epochs", 1),
                total_epoch=training_params.epochs,
            )

            fold_checkpoint_path = str(checkpoint_dir / f"cleanlab_fold_{fold_idx}.pt")

            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                model = build_pipeline_model(device, model_params)
                optimizer = build_pipeline_optimizer(model, optim_params)
                scheduler = build_scheduler(optimizer, scheduler_params)

                workflow.execute(
                    device=device,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    checkpoint_path=fold_checkpoint_path,
                    training_params=training_params,
                )

                # Out-of-fold inference on validation set
                fold_obs_ids, fold_labels, fold_preds = val_execute(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    as_numpy=True,
                )

                all_oof_obs_ids.extend(fold_obs_ids)
                all_oof_labels.append(fold_labels)
                all_oof_preds.append(fold_preds)

        oof_labels_arr = np.concatenate(all_oof_labels, axis=0)
        oof_preds_arr = np.concatenate(all_oof_preds, axis=0)

        # Sort aggregated results deterministically by observation_id
        sort_idx = np.argsort(all_oof_obs_ids)
        sorted_obs_ids = [all_oof_obs_ids[i] for i in sort_idx]
        sorted_labels = oof_labels_arr[sort_idx]
        sorted_preds = oof_preds_arr[sort_idx]

        logger.info(
            "\nRunning Cleanlab issue detection across all out-of-fold predictions..."
        )
        cleanlab_results = run_cleanlab_analysis(
            obs_ids=sorted_obs_ids,
            labels=sorted_labels,
            preds=sorted_preds,
            class_names=CLASS_ORDER,
        )

        save_oof_predictions_to_duckdb(
            db_path=cleanlab_db_path,
            obs_ids=sorted_obs_ids,
            labels=sorted_labels,
            preds=sorted_preds,
            table_name=db_table_name,
        )

        exported_csvs = []
        if cleanlab_cfg.get("export_csv", True):
            for c_idx, c_name in enumerate(CLASS_ORDER):
                issues_for_c = cleanlab_results["per_class_issues"].get(c_name, [])
                csv_p = export_label_issues_csv(
                    corrections_dir=corrections_dir,
                    class_name=c_name,
                    class_idx=c_idx,
                    ranked_obs_ids=issues_for_c,
                    all_obs_ids=sorted_obs_ids,
                    labels=sorted_labels,
                    preds=sorted_preds,
                )
                exported_csvs.append(csv_p)
                mlflow.log_artifact(str(csv_p))

        for c_name, rate in cleanlab_results["per_class_issue_rates"].items():
            mlflow.log_metric(f"cleanlab/{c_name}_issue_rate", rate)
            mlflow.log_metric(
                f"cleanlab/{c_name}_issue_count",
                len(cleanlab_results["per_class_issues"].get(c_name, [])),
            )

        mean_quality = float(np.mean(cleanlab_results["quality_scores"]))
        mlflow.log_metric("cleanlab/mean_quality_score", mean_quality)

    logger.info("\nCleanlab K-Fold CV pipeline completed successfully!")
    logger.info(f"Mean quality score: {mean_quality:.4f}")
    for c_name in CLASS_ORDER:
        count = len(cleanlab_results["per_class_issues"].get(c_name, []))
        rate = cleanlab_results["per_class_issue_rates"].get(c_name, 0.0)
        logger.info(f" - {c_name}: {count} issues ({rate * 100:.1f}%)")

    return {
        "obs_ids": sorted_obs_ids,
        "labels": sorted_labels,
        "preds": sorted_preds,
        "cleanlab_results": cleanlab_results,
        "exported_csvs": exported_csvs,
    }

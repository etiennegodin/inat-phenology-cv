from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Union

import mlflow
import numpy as np
import torch

from .metrics import log_best_artifacts

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from .model import PhenologyModel

logger = logging.getLogger(__name__)


def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy types to native Python types for clean pickling."""
    clean = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            clean[k] = float(v)
        elif isinstance(v, (str, bool)):
            clean[k] = v
    return clean


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    eval_metrics: dict[str, Any],
    to_mlflow=False,
) -> None:
    """Save training checkpoint to disk and MLflow."""
    sanitized_metrics = _sanitize_metrics(eval_metrics)
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_metrics": sanitized_metrics,
        "run_id": run_id,
    }

    checkpoint_path = f"{checkpoint_path}/{run_id}.pth"

    val_loss = sanitized_metrics.get("val/loss", sanitized_metrics.get("val_loss", 0.0))
    logger.info(f"Saving checkpoint for epoch {epoch} with val_loss of {val_loss:.4f}")
    t0 = time.time()
    torch.save(checkpoint, checkpoint_path)
    t1 = time.time() - t0
    logger.debug(f"Took {t1:.3f}s")

    if to_mlflow:
        if mlflow.active_run():
            mlflow.log_artifact(checkpoint_path)
            t2 = time.time() - t1
            logger.debug(f"Took {t2:.3f}s")

        logger.info("Logged to mlflow")

    log_best_artifacts(eval_metrics)


def load_checkpoint(
    checkpoint_path: str,
    model: PhenologyModel,
    optimizer: Optimizer,
) -> tuple[PhenologyModel, Optimizer, int, dict[str, Any], Union[str, None]]:
    """Load model checkpoint safely across PyTorch versions."""
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    checkpoint_path = f"{checkpoint_path}/{run_id}.pth"

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        # Fallback for PyTorch versions prior to weights_only parameter
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    eval_metrics = checkpoint.get("eval_metrics", {})
    run_id = checkpoint.get("run_id", None)

    val_loss = eval_metrics.get("val/loss", eval_metrics.get("val_loss", 0.0))
    logger.info(
        f"Reloading checkpoint run_id={run_id}, epoch={start_epoch}, "
        f"val_loss={val_loss:.4f}"
    )
    return model, optimizer, start_epoch, eval_metrics, run_id

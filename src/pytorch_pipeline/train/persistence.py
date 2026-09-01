from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Union

import mlflow
import torch

from .metrics import EpochMetrics, log_best_artifacts

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from .model import PhenologyModel

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    eval_metrics: EpochMetrics | None = None,
    to_mlflow=False,
) -> None:
    """Save training checkpoint to disk and MLflow."""
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "run_id": run_id,
    }

    if eval_metrics is not None:
        checkpoint["eval_metrics"] = (
            eval_metrics.to_dict() if hasattr(eval_metrics, "to_dict") else eval_metrics
        )
        if hasattr(eval_metrics, "to_dict"):
            log_best_artifacts(eval_metrics)

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    checkpoint_file = f"{checkpoint_path}/{run_id}.pth"

    t0 = time.time()
    torch.save(checkpoint, checkpoint_file)
    t1 = time.time() - t0
    logger.debug(f"Took {t1:.3f}s")

    if to_mlflow:
        if mlflow.active_run():
            mlflow.log_artifact(checkpoint_file)
            t2 = time.time() - t1
            logger.debug(f"Took {t2:.3f}s")


def load_checkpoint(
    checkpoint_path: str,
    model: PhenologyModel,
    optimizer: Optimizer | None = None,
) -> tuple[
    PhenologyModel, Optimizer | None, int, EpochMetrics | None, Union[str, None]
]:
    """Load model checkpoint safely across PyTorch versions."""

    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

    checkpoint_file = f"{checkpoint_path}/{run_id}.pth"

    try:
        checkpoint = torch.load(checkpoint_file, weights_only=False)

    except TypeError:
        # Fallback for PyTorch versions prior to weights_only parameter
        checkpoint = torch.load(checkpoint_file)

    model.load_state_dict(checkpoint["model_state_dict"])
    run_id = checkpoint.get("run_id", None)
    start_epoch = checkpoint.get("epoch", 0)

    if optimizer is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning(
                "Warning, failed to load optimizer_state_dict in loaded checkpoint "
                f"{checkpoint_file}"
            )

    eval_metrics: EpochMetrics | None = None
    if "eval_metrics" in checkpoint:
        raw = checkpoint["eval_metrics"]
        try:
            eval_metrics = EpochMetrics.from_dict(raw)
            logger.info(
                f"Reloading checkpoint run_id={run_id}, epoch={start_epoch}, "
                f"val_loss={eval_metrics.val_loss:.4f}"
            )
        except (TypeError, KeyError) as exc:
            if isinstance(raw, dict):
                eval_metrics = raw
            else:
                logger.warning(
                    f"Could not reconstruct EpochMetrics from checkpoint ({exc}). "
                    "eval_metrics will be None."
                )
    return model, optimizer, start_epoch, eval_metrics, run_id

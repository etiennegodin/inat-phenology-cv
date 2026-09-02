from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow
import torch

from ..utils.misc import get_mlflow_run_id
from ..utils.params import ModelParams
from .factory import build_pipeline_model, get_device
from .metrics import EpochMetrics, log_best_artifacts
from .model import PhenologyModel

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from .model import PhenologyModel

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    model: PhenologyModel
    optimizer: Optimizer | None
    eval_metrics: EpochMetrics | None

    @classmethod
    def from_file(
        cls,
        checkpoint_path: str,
        run_id: str | None = None,
        model: PhenologyModel | None = None,
        model_params: ModelParams | dict | None = None,
        optimizer: Optimizer | None = None,
    ) -> Checkpoint:
        """Reinstates a full Checkpoint object from a raw checkpoint file

        Args:
            checkpoint_path (str): Checkpoint folder path
            run_id (str | None, optional): Mlflow run id. Defaults to None.
            model (PhenologyModel | None, optional): Optionnal model to reinstate.
            Defaults to None.
            model_params (ModelParams | None, optional): Model description.
              Mainly for legacy runs without model params in checkfpoint.
              Defaults to None.
            optimizer (Optimizer | None, optional): Optionnal optimizer to reinstate.
            Defaults to None.

        Raises:
            ValueError: If no model, model_params is provided and absent from checkpoint

        Returns:
            Checkpoint: Checkpoint dataclass instance
        """
        if run_id is None:
            run_id = get_mlflow_run_id()
        checkpoint_file = f"{checkpoint_path}/{run_id}.pth"
        try:
            checkpoint_dict = torch.load(checkpoint_file, weights_only=False)

        except TypeError:
            # Fallback for PyTorch versions prior to weights_only parameter
            checkpoint_dict = torch.load(checkpoint_file)

        # Try to re-instanciate model if not provied
        if model is None:
            # If model params is not provided, read it back from checkpoint
            if model_params is None:
                if "model_params" in checkpoint_dict:
                    model_params = ModelParams(**checkpoint_dict["model_params"])
                else:
                    raise ValueError(
                        "No model params provided and in checkpoint file - "
                        "Can't re-instanciate model to load state dict"
                    )
            elif isinstance(model_params, dict):
                model_params = ModelParams(**model_params)

            device = get_device()
            model = build_pipeline_model(device, model_params)

        model.load_state_dict(checkpoint_dict["model_state_dict"])
        run_id = checkpoint_dict.get("run_id", None)
        start_epoch = checkpoint_dict.get("epoch", 0)

        if optimizer is not None:
            if "optimizer_state_dict" in checkpoint_dict:
                optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
            else:
                logger.warning(
                    "Warning, failed to load optimizer_state_dict in loaded checkpoint "
                    f"{checkpoint_file}"
                )

        eval_metrics: EpochMetrics | None = None
        if "eval_metrics" in checkpoint_dict:
            raw = checkpoint_dict["eval_metrics"]
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

        return Checkpoint(model, optimizer, eval_metrics)

    def save(
        self,
        checkpoint_path: str,
        epoch: int,
        to_mlflow=False,
    ) -> None:
        """Save training checkpoint to disk and MLflow."""
        run_id = get_mlflow_run_id()
        checkpoint_file = f"{checkpoint_path}/{run_id}.pth"

        checkpoint = self.to_dict()
        checkpoint.update(
            {
                "epoch": epoch,
                "run_id": run_id,
            }
        )

        torch.save(checkpoint, checkpoint_file)

        if to_mlflow:
            if mlflow.active_run():
                mlflow.log_artifact(checkpoint_file)

    def to_dict(self) -> dict:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_params": self.model.params.to_dict(),
        }

        if self.eval_metrics is not None:
            checkpoint["eval_metrics"] = (
                self.eval_metrics.to_dict()
                if hasattr(self.eval_metrics, "to_dict")
                else self.eval_metrics
            )
            if hasattr(self.eval_metrics, "to_dict"):
                log_best_artifacts(self.eval_metrics)

        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        return checkpoint

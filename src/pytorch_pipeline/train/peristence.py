from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import mlflow
import torch

from .metrics import log_best_artifacts

if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    eval_metrics: dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_metrics": eval_metrics,
        "run_id": mlflow.active_run().info.run_id,
    }
    print(
        f"Saving checkpoint for epoch {epoch} "
        f"with loss of {eval_metrics['val_loss']:.3f}"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )
    mlflow.log_artifact(
        checkpoint_path,
    )

    log_best_artifacts(eval_metrics)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optimizer,
) -> tuple[torch.nn.Sequential, Optimizer, int, dict[str, Any], Union[str, None]]:
    checkpoint = torch.load(checkpoint_path)
    checkpoint: dict
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    eval_metrics = checkpoint.get("eval_metrics", {})
    run_id = checkpoint.get("run_id", None)

    print(f"Reloading run {run_id}")
    print(f"Previous epoch= {start_epoch} previous_loss={eval_metrics['val_loss']}")
    return model, optimizer, start_epoch, eval_metrics, run_id

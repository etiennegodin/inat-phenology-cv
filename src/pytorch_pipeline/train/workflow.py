from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import mlflow
import torch
from torch.amp import autocast_mode, grad_scaler

from .metrics import get_metrics, log_best_artifacts
from .peristence import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from torch import Tensor, device, nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from ..utils.params import TrainingParams


logger = logging.getLogger(__name__)

scaler = grad_scaler.GradScaler()


def forward_pass(model: nn.Sequential, images: list[Tensor], device: device) -> Tensor:

    indices = []

    # Get indices for split
    for t in images:
        indices.append(t.size()[0])

    # Concatenate all images in one Tensor
    stacked = torch.cat(images)

    # Transfer to device
    stacked = stacked.to(device)

    pooled = []
    # Run backbone on stacked Tensor
    embeddings = model[0](stacked)
    # Split embedding per observation
    chunks = torch.split(embeddings, indices)

    # Append observation pool
    for c in chunks:
        pooled.append(c.mean(0))

    # Stack back in one Tensor
    pooled = torch.stack(pooled)
    return model[1](pooled).squeeze(1)


def train_one_epoch(
    epoch: int,
    model: nn.Sequential,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: device,
):
    total_loss = 0
    model.train()

    for images, labels in dataloader:
        labels: Tensor

        optimizer.zero_grad()

        # Run foward prop and backprop
        if device.type == "cuda":
            with autocast_mode.autocast(device_type=device.type):
                predictions = forward_pass(model, images, device)
                loss = criterion(predictions, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predictions = forward_pass(model, images, device)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(dataloader)

    # MLflow 'step' tells it which epoch this is for the graph
    mlflow.log_metric("train_loss", train_loss, step=epoch)

    return train_loss


def evaluate(
    model: nn.Sequential,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: device,
    epoch: int,
) -> dict[str, Any]:
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_preds_raw = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            labels: Tensor
            # Run foward prop and backprop
            if device.type == "cuda":
                with autocast_mode.autocast(device_type=device.type):
                    predictions = forward_pass(model, images, device)
            else:
                predictions = forward_pass(model, images, device)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            preds_raw = torch.sigmoid(predictions)
            preds = (preds_raw > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds)
            all_labels.append(labels)
            all_preds_raw.append(preds_raw)

    # Convert back to numpy arrays for metrics
    all_preds = torch.cat(all_preds).detach().cpu().numpy()
    all_labels = torch.cat(all_labels).detach().cpu().numpy()
    all_preds_raw = torch.cat(all_preds_raw).detach().cpu().numpy()

    base_metrics = {
        "val_accuracy": correct / total,
        "val_loss": total_loss / len(dataloader),
    }
    mlflow.log_metrics(base_metrics, step=epoch)

    eval_metrics = get_metrics(all_preds, all_labels, all_preds_raw)
    mlflow.log_metric("val_roc_auc", eval_metrics["val_roc_auc"], step=epoch)
    eval_metrics.update(base_metrics)
    return eval_metrics


def execute(
    device: device,
    model: torch.nn.Sequential,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    checkpoint_path: str,
    training_params: TrainingParams,
):

    patience_counter = 0

    for epoch in range(training_params.epochs):
        if (
            training_params.start_epoch is not None
            and epoch <= training_params.start_epoch
        ):
            print(f"Skipping epoch {epoch}")
            continue
        start = time.time()
        train_loss = train_one_epoch(
            epoch,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        eval_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        elapsed = time.time() - start
        val_loss = eval_metrics["val_loss"]
        gap = val_loss - train_loss
        mlflow.log_metric("loss_gap", gap, step=epoch)

        print(
            f"Epoch {epoch}: train={train_loss:.3f} val={eval_metrics['val_loss']:.3f} "
            f"gap={gap:.3f} accuracy={float(eval_metrics['val_accuracy']):.3f} "
            f" roc={float(eval_metrics['val_roc_auc']):.3f} time={elapsed:.3f}s"
        )
        if device.type == "cuda":
            print(
                f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB /"
                f" {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
            )

        if val_loss < training_params.best_loss:
            # Save only if better
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                eval_metrics=eval_metrics,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= training_params.patience:
                print("Early stopping")
                break

    # Reload best model
    checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    log_best_artifacts(checkpoint[3])
    return checkpoint[0]

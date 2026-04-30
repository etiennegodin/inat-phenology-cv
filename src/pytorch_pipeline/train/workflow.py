import logging
import time
from typing import Any

import mlflow
import pandas as pd
import torch
import torch.optim as optim
from torch.amp import autocast_mode, grad_scaler
from torch.utils.data import DataLoader

from ..utils.params import TrainingParams
from .metrics import get_metrics, log_best_artifacts

logger = logging.getLogger(__name__)

scaler = grad_scaler.GradScaler()


def forward_pass(
    model: torch.nn.Sequential, images: list[torch.Tensor], device
) -> torch.Tensor:

    indices = []

    # Get indices for split
    for t in images:
        indices.append(t.size()[0])

    # Concatenate all images in one torch.Tensor
    stacked = torch.cat(images)

    # Transfer to device
    stacked = stacked.to(device)

    pooled = []
    # Run backbone on stacked torch.Tensor
    embeddings = model[0](stacked)
    # Split embedding per observation
    chunks = torch.split(embeddings, indices)

    # Append observation pool
    for c in chunks:
        pooled.append(c.mean(0))

    # Stack back in one torch.Tensor
    pooled = torch.stack(pooled)
    return model[1](pooled).squeeze(1)


def train_one_epoch(
    epoch: int,
    model: torch.nn.Sequential,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
):
    total_loss = 0
    model.train()

    for images, labels in dataloader:
        labels: torch.Tensor

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
    model: torch.nn.Sequential,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
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
            labels: torch.Tensor
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
    device: torch.device,
    model: torch.nn.Sequential,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    checkpoint_path: str,
    training_params: TrainingParams,
):

    patience_counter = 0
    if training_params.reload:
        # Reload previous run
        model, optimizer, start_epoch, eval_metrics = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
        )
        best_loss = eval_metrics["val_loss"]
    else:
        # Fresh run
        best_loss = 1e10
        start_epoch = None

    for epoch in range(training_params.epochs):
        if start_epoch is not None and epoch <= start_epoch:
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

        if val_loss < best_loss:
            best_loss = val_loss
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
    model, optimizer, start_epoch, eval_metrics = load_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer
    )
    log_best_artifacts(eval_metrics)
    return model


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: torch.nn.Sequential,
    optimizer: optim.Optimizer,
    eval_metrics: dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_metrics": eval_metrics,
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


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Sequential,
    optimizer: optim.Optimizer,
) -> tuple[torch.nn.Sequential, optim.Optimizer, int, dict]:
    checkpoint = torch.load(checkpoint_path)
    checkpoint: dict
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    eval_metrics = checkpoint.get("eval_metrics", {})

    print(f"Reloading session at: {checkpoint_path}")
    print(f"Previous epoch= {start_epoch} previous_loss={eval_metrics['val_loss']}")

    return model, optimizer, start_epoch, eval_metrics


def train_report(eval_metrics: dict):
    labels = ["flowering", "non_flowering"]
    df_cm = pd.DataFrame(eval_metrics["cm"], index=labels, columns=labels)
    print(df_cm)

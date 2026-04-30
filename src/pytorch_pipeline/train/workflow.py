import logging
import time

import mlflow
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.amp import autocast_mode, grad_scaler
from torch.utils.data import DataLoader

from ..utils.params import TrainingParams

logger = logging.getLogger(__name__)

scaler = grad_scaler.GradScaler()


def train_one_epoch(
    epoch: int,
    model: nn.Sequential,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):

    total_loss = 0
    model.train()

    def foward_prop(
        tensor: torch.Tensor,
        labels: torch.Tensor,
        indices: list,
    ):
        pooled = []
        # Run backbone on stacked tensor
        embeddings = model[0](tensor)
        # Split embedding per observation
        chunks = torch.split(embeddings, indices)

        # Append observation pool
        for c in chunks:
            pooled.append(c.mean(0))

        # Stack back in one tensor
        pooled = torch.stack(pooled)
        predictions = model[1](pooled).squeeze(1)
        return criterion(predictions, labels)

    for images, labels in dataloader:
        labels: torch.Tensor
        indices = []

        optimizer.zero_grad()

        # Get indices for split
        for t in images:
            indices.append(t.size()[0])

        # Concatenate all images in one tensor
        stacked = torch.cat(images)

        # Transfer to device
        labels = labels.to(device)
        stacked = stacked.to(device)

        # Run foward prop and backprop
        if device.type == "cuda":
            with autocast_mode.autocast(device_type=device.type):
                loss = foward_prop(stacked, labels, indices)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = foward_prop(stacked, labels, indices)
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
    device: torch.device,
    epoch: int,
) -> dict:
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_preds_raw = []
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            labels = labels.to(device)
            predictions = []
            for obs_images in images:
                obs_images = obs_images.to(device)
                embeddings = model[0](obs_images)
                pooled = embeddings.mean(0)
                prediction = model[1](pooled)
                predictions.append(prediction)
            predictions = torch.stack(predictions).squeeze(1)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            preds_raw = torch.sigmoid(predictions)
            preds = (preds_raw > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds)
            all_labels.append(labels)
            all_preds_raw.append(preds_raw)

    accuracy = correct / total
    all_preds = torch.cat(all_preds).detach().cpu().numpy()
    all_labels = torch.cat(all_labels).detach().cpu().numpy()
    all_preds_raw = torch.cat(all_preds_raw).detach().cpu().numpy()

    # Metrics
    val_loss = total_loss / len(dataloader)
    cm = confusion_matrix(all_labels, all_preds)
    roc_auc = float(roc_auc_score(all_labels, all_preds_raw))

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy,
        "cm": cm,
        "roc": roc_auc,
    }

    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_accuracy", accuracy, step=epoch)
    mlflow.log_metric("val_roc", roc_auc, step=epoch)
    mlflow.log_metric("val_accuracy", accuracy, step=epoch)

    return metrics


def execute(
    device: torch.device,
    model: nn.Sequential,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    checkpoint_path: str,
    training_params: TrainingParams,
):
    patience_counter = 0
    if training_params.reload:
        # Reload previous run
        model, optimizer, start_epoch, best_loss = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
        )
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

        metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )
        elapsed = time.time() - start

        val_loss = metrics["loss"]
        gap = val_loss - train_loss
        print(
            f"Epoch {epoch}: train={train_loss:.3f} val={val_loss:.3f} "
            f"gap={gap:.3f} accuracy={float(metrics['accuracy']):.3f} "
            f" roc={float(metrics['roc']):.3f} time={elapsed:.3f}s"
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
                best_val_loss=best_loss,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= training_params.patience:
                print("Early stopping")
                break

    return model, checkpoint_path


def save_checkpoint(
    checkpoint_path: str,
    epoch: int,
    model: nn.Sequential,
    optimizer: optim.Optimizer,
    best_val_loss: float,
) -> None:
    print(f"Saving checkpoint for epoch {epoch} with loss of {best_val_loss:.3f}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Sequential,
    optimizer: optim.Optimizer,
) -> tuple[nn.Sequential, optim.Optimizer, int, float]:
    checkpoint = torch.load(checkpoint_path)
    checkpoint: dict
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    best_val_loss = checkpoint.get("best_val_loss", 1e10)
    print(f"Reloading session at: {checkpoint_path}")
    print(f"Previous epoch= {start_epoch} previous_loss={best_val_loss}")
    return model, optimizer, start_epoch, best_val_loss


def train_report(metrics: dict):
    labels = ["flowering", "non_flowering"]
    df_cm = pd.DataFrame(metrics["cm"], index=labels, columns=labels)
    print(df_cm)

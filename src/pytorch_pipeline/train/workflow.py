from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch
from torch.amp import autocast_mode, grad_scaler

from .metrics import log_best_artifacts, log_metrics
from .persistence import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from torch import Tensor, device, nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from ..utils.params import TrainingParams
    from .model import PhenologyModel


logger = logging.getLogger(__name__)

scaler = grad_scaler.GradScaler()


def mem(msg):
    print(
        msg,
        torch.cuda.memory_allocated() / 1024**2,
        torch.cuda.memory_reserved() / 1024**2,
    )


def train_one_epoch(
    epoch: int,
    model: PhenologyModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: device,
):
    data_time = 0
    compute_time = 0
    t0 = time.time()
    logger.debug(f"Start train one epoch {epoch}")
    total_loss = 0
    model.train()
    t0 = time.time()

    batches = 0
    img_per_batch = []
    obs_per_batch = []

    for images, labels in dataloader:
        indices = []
        for i in images:
            indices.append(i.size()[0])

        # Stats
        total_img = sum(indices)
        obs = len(indices)
        obs_per_batch.append(obs)
        img_per_batch.append(total_img)

        batches += 1

        t1 = time.time()
        data_time += t1 - t0

        labels: Tensor
        images = [t.to(device) for t in images]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Run foward prop and backprop
        if device.type == "cuda":
            with autocast_mode.autocast(device_type=device.type):
                predictions, weights = model(images)
                loss = criterion(predictions, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
        else:
            logger.info("start prediction")
            predictions, weights = model(images)
            logger.info("predictions")
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

        t0 = time.time()
        compute_time += t0 - t1

        total_loss += loss.item()

    train_loss = total_loss / len(dataloader)

    print("total_images", np.sum(img_per_batch))
    print("total_batches", batches)
    print("total obs", len(dataloader))

    print(f"observation per batch mean {np.mean(obs_per_batch):.3f}")
    print(f"observation per batch std {np.std(obs_per_batch):.3f}")
    print(f"imgs per batch mean {np.mean(img_per_batch):.3f}")
    print(f"imgs per batch std {np.std(img_per_batch):.3f}")

    print(f"imgs per observation mean {(np.sum(img_per_batch) / len(dataloader)):.3f}")

    # MLflow 'step' tells it which epoch this is for the graph
    mlflow.log_metric("train_loss", train_loss, step=epoch)

    return train_loss, (data_time, compute_time)


def evaluate(
    model: PhenologyModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: device,
    epoch: int,
) -> tuple[dict[str, Any], tuple[float, float]]:

    data_time = 0
    compute_time = 0
    t0 = time.time()

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
            images = [t.to(device) for t in images]
            labels = labels.to(device)

            t1 = time.time()
            data_time += t1 - t0

            # Run foward prop and backprop
            if device.type == "cuda":
                with autocast_mode.autocast(device_type=device.type):
                    predictions, weights = model(images)
                    torch.cuda.synchronize()

            else:
                predictions, weights = model(images)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            preds_raw = torch.sigmoid(predictions)
            preds = (preds_raw > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds)
            all_labels.append(labels)
            all_preds_raw.append(preds_raw)
            t0 = time.time()
            compute_time += t0 - t1

    # Convert back to numpy arrays for metrics
    all_preds = torch.cat(all_preds).detach().cpu().numpy()
    all_labels = torch.cat(all_labels).detach().cpu().numpy()
    all_preds_raw = torch.cat(all_preds_raw).detach().cpu().numpy()

    base_metrics = {
        "val_loss": total_loss / len(dataloader),
    }
    if mlflow.active_run():
        mlflow.log_metrics(base_metrics, step=epoch)

    eval_metrics = log_metrics(all_preds, all_labels, all_preds_raw, epoch)
    eval_metrics.update(base_metrics)
    return eval_metrics, (data_time, compute_time)


def execute(
    device: device,
    model: PhenologyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler,
    criterion: nn.Module,
    checkpoint_path: str,
    training_params: TrainingParams,
):

    patience_counter = 0
    best_loss = training_params.best_loss

    for epoch in range(training_params.epochs):
        if (
            training_params.start_epoch is not None
            and epoch <= training_params.start_epoch
        ):
            print(f"Skipping epoch {epoch}")
            continue

        train_loss, train_times = train_one_epoch(
            epoch,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        # Stepping though scheduler

        # to_do: resume mechanics need scheduler.load_state_dict() (or equivalent)
        # to stay in sync with start_epoch — revisit after Optuna is working end-to-end
        scheduler.step()
        current_lr = scheduler.get_last_lr()  # list, one value per param group
        mlflow.log_metric("lr_backbone", float(current_lr[0]), step=epoch)
        mlflow.log_metric("lr_attention", float(current_lr[1]), step=epoch)
        mlflow.log_metric("lr_head", float(current_lr[2]), step=epoch)
        mlflow.log_metric("train_data_time", float(train_times[0]), step=epoch)
        mlflow.log_metric("train_compute_time", float(train_times[1]), step=epoch)

        eval_metrics, eval_times = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_loss = eval_metrics["val_loss"]
        gap = val_loss - train_loss
        mlflow.log_metric("loss_gap", gap, step=epoch)
        mlflow.log_metric("eval_data_time", float(eval_times[0]), step=epoch)
        mlflow.log_metric("eval_compute_time", float(eval_times[1]), step=epoch)

        print(
            f"Epoch {epoch}: train={train_loss:.6f} val={eval_metrics['val_loss']:.6f} "
            f"gap={gap:.6f} "
            f"roc_auc_macro={float(eval_metrics['val_roc_auc_macro']):.3f} "
        )

        print(
            f"train data_time={train_times[0]:.1f}s | "
            f"train_compute_time={train_times[1]:.1f}s | "
            f"eval data_time={eval_times[0]:.1f}s | "
            f"eval_compute_time={eval_times[1]:.1f}s"
        )

        if device.type == "cuda" and epoch == 0:
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
    checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    log_best_artifacts(checkpoint[3])
    return checkpoint[0]

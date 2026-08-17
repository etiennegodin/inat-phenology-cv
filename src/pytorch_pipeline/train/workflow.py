from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch
from torch.amp import autocast_mode, grad_scaler
from tqdm import tqdm

from ..utils.configs import CLASS_ORDER
from .metrics import log_attention_metrics, log_best_artifacts, log_metrics
from .persistence import load_checkpoint, save_checkpoint

if TYPE_CHECKING:
    from torch import Tensor, device, nn
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from ..utils.params import TrainingParams
    from .model import PhenologyModel


logger = logging.getLogger(__name__)


dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = None
if dtype == torch.float16:
    scaler = grad_scaler.GradScaler()


def train_one_epoch(
    epoch: int,
    model: PhenologyModel,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: device,
    log_step_interval: int = 10,
) -> tuple[float, tuple[float, float]]:
    """Train the model for one epoch with progress tracking and step logging."""
    data_time = 0.0
    compute_time = 0.0
    total_loss = 0.0
    classes_loss = {la: 0.0 for la in CLASS_ORDER}
    model.train()

    img_per_batch = []
    obs_per_batch = []
    all_obs_weights = {}
    for c in CLASS_ORDER:
        all_obs_weights[c] = []

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [Train]",
        leave=False,
        dynamic_ncols=True,
    )

    t0 = time.time()
    for step, (images, labels) in enumerate(pbar):
        indices = [img.size(0) for img in images]
        total_img = sum(indices)
        obs_count = len(indices)

        obs_per_batch.append(obs_count)
        img_per_batch.append(total_img)

        t1 = time.time()
        data_time += t1 - t0

        labels: Tensor
        images = [t.to(device) for t in images]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # to_do add switch if scaler is needed for amp of previous cards vs new ones
        if device.type == "cuda":
            with autocast_mode.autocast(device_type=device.type, dtype=dtype):
                predictions, class_weights = model(images)
                raw_loss = criterion(predictions, labels)
                class_loss = torch.mean(raw_loss, dim=0)
                for i, c in enumerate(CLASS_ORDER):
                    classes_loss[c] += class_loss[i].item()
                loss = torch.mean(class_loss, dim=0)

                # Scale if fp16
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            torch.cuda.synchronize()
        else:
            predictions, class_weights = model(images)
            raw_loss = criterion(predictions, labels)
            class_loss = torch.mean(raw_loss, dim=0)
            for i, c in enumerate(CLASS_ORDER):
                classes_loss[c] += class_loss[i].item()
            loss = torch.mean(class_loss, dim=0)
            loss.backward()
            optimizer.step()

        t0 = time.time()
        compute_time += t0 - t1

        current_loss = loss.item()
        total_loss += current_loss

        for k, v in class_weights.items():
            all_obs_weights[k].append(v)

        pbar.set_postfix(
            {
                "loss": f"{current_loss:.4f}",
                "avg_loss": f"{total_loss / (step + 1):.4f}",
                "imgs/batch": total_img,
            }
        )

        if (
            mlflow.active_run()
            and log_step_interval > 0
            and step % log_step_interval == 0
        ):
            global_step = epoch * len(dataloader) + step
            mlflow.log_metric("train/batch_loss", current_loss, step=global_step)

    train_loss = total_loss / len(dataloader)
    if mlflow.active_run():
        mlflow.log_metric("train/loss", train_loss, step=epoch)
        for k, v in classes_loss.items():
            c_loss = v / len(dataloader)
            mlflow.log_metric(f"train/{k}_loss", c_loss, step=epoch)
    logger.debug(
        f"Epoch {epoch} Train: Loss={train_loss:.6f} | "
        f"Total Imgs={np.sum(img_per_batch)} | Total Obs={len(dataloader)}"
    )

    # Attention
    log_attention_metrics(epoch, all_obs_weights, prefix="train")

    return train_loss, (data_time, compute_time)


def evaluate(
    model: PhenologyModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: device,
    epoch: int,
) -> tuple[dict[str, Any], tuple[float, float]]:
    """Evaluate the model on validation set with rich multi-label metrics."""
    data_time = 0.0
    compute_time = 0.0

    total_loss = 0.0
    classes_loss = {la: 0.0 for la in CLASS_ORDER}

    all_preds_bin = []
    all_labels = []
    all_preds_raw = []
    all_obs_weights = {}
    for c in CLASS_ORDER:
        all_obs_weights[c] = []

    model.eval()

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [Val]  ",
        leave=False,
        dynamic_ncols=True,
    )

    t0 = time.time()
    with torch.no_grad():
        for step, (images, labels) in enumerate(pbar):
            labels: Tensor
            images = [t.to(device) for t in images]
            labels = labels.to(device)

            t1 = time.time()
            data_time += t1 - t0

            if device.type == "cuda":
                with autocast_mode.autocast(device_type=device.type, dtype=dtype):
                    predictions, class_weights = model(images)
                    torch.cuda.synchronize()
            else:
                predictions, class_weights = model(images)

            raw_loss = criterion(predictions, labels)
            class_loss = torch.mean(raw_loss, dim=0)
            for i, c in enumerate(CLASS_ORDER):
                classes_loss[c] += class_loss[i].item()
            loss = torch.mean(class_loss, dim=0)
            total_loss += loss.item()

            preds_raw = torch.sigmoid(predictions)
            preds_bin = (preds_raw >= 0.5).float()

            all_preds_bin.append(preds_bin.detach().float().cpu())
            all_labels.append(labels.detach().float().cpu())
            all_preds_raw.append(preds_raw.detach().float().cpu())
            for k, v in class_weights.items():
                all_obs_weights[k].append(v)
            t0 = time.time()
            compute_time += t0 - t1

            pbar.set_postfix({"val_loss": f"{total_loss / (step + 1):.4f}"})

    all_preds_bin_np = torch.cat(all_preds_bin).numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    all_preds_raw_np = torch.cat(all_preds_raw).numpy()

    val_loss = total_loss / len(dataloader)
    base_metrics = {
        "val/loss": val_loss,
        "val_loss": val_loss,  # Backward compatibility alias
    }

    if mlflow.active_run():
        mlflow.log_metric("val/loss", val_loss, step=epoch)
        for k, v in classes_loss.items():
            c_loss = v / len(dataloader)
            mlflow.log_metric(f"val/{k}_loss", c_loss, step=epoch)

    eval_metrics = log_metrics(
        all_preds=all_preds_bin_np,
        all_labels=all_labels_np,
        all_preds_raw=all_preds_raw_np,
        epoch=epoch,
        prefix="val",
    )

    log_attention_metrics(epoch, all_obs_weights, prefix="val")

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
    """Execute training pipeline over requested epochs with full logging."""
    patience_counter = 0
    best_loss = training_params.best_loss
    best_eval_metrics = {}

    log_step_interval = getattr(training_params, "log_step_interval", 10)

    logger.info(
        f"Starting training run: total epochs={training_params.epochs}, "
        f"patience={training_params.patience}, device={device.type}"
    )

    for epoch in range(training_params.epochs):
        if (
            training_params.start_epoch is not None
            and epoch <= training_params.start_epoch
        ):
            logger.info(f"Skipping epoch {epoch} (resume requested)")
            continue

        epoch_start_time = time.time()

        train_loss, train_times = train_one_epoch(
            epoch,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            log_step_interval=log_step_interval,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()

        if mlflow.active_run():
            mlflow.log_metric("lr/backbone", float(current_lr[0]), step=epoch)
            mlflow.log_metric("lr/attention", float(current_lr[1]), step=epoch)
            mlflow.log_metric("lr/head", float(current_lr[2]), step=epoch)
            mlflow.log_metric("time/train_data", float(train_times[0]), step=epoch)
            mlflow.log_metric("time/train_compute", float(train_times[1]), step=epoch)

        eval_metrics, eval_times = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_loss = eval_metrics.get("val/loss", eval_metrics.get("val_loss", 0.0))
        gap = val_loss - train_loss
        epoch_duration = time.time() - epoch_start_time

        if mlflow.active_run():
            mlflow.log_metric("val/loss_gap", gap, step=epoch)
            mlflow.log_metric("time/eval_data", float(eval_times[0]), step=epoch)
            mlflow.log_metric("time/eval_compute", float(eval_times[1]), step=epoch)
            mlflow.log_metric("time/epoch_duration", float(epoch_duration), step=epoch)

        roc_macro = eval_metrics.get(
            "val/roc_auc_macro", eval_metrics.get("val_roc_auc_macro", 0.0)
        )
        pr_macro = eval_metrics.get(
            "val/pr_auc_macro", eval_metrics.get("val_pr_auc_macro", 0.0)
        )
        f1_best_macro = eval_metrics.get(
            "val/f1_macro_best", eval_metrics.get("val_f1_macro_best", 0.0)
        )

        logger.info(
            f"Epoch {epoch:02d}/{training_params.epochs:02d} [{epoch_duration:.1f}s] - "
            f"train_loss: {train_loss:.5f} | "
            f"val_loss: {val_loss:.5f} | gap: {gap:.5f} | "
            f"ROC-AUC: {roc_macro:.3f} | PR-AUC: {pr_macro:.3f} "
            f"| Best-F1: {f1_best_macro:.3f}"
        )

        logger.debug(
            f"Timings -> Train Data: {train_times[0]:.1f}s, "
            f"Compute: {train_times[1]:.1f}s | "
            f"Eval Data: {eval_times[0]:.1f}s, "
            f"Compute: {eval_times[1]:.1f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_eval_metrics = eval_metrics
            logger.info(
                f"Val loss improved to {val_loss:.5f}. "
                f"Saving checkpoint to {checkpoint_path}."
            )
            save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                eval_metrics=eval_metrics,
                to_mlflow=False,
            )
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Val loss did not improve. "
                f"Patience: {patience_counter}/{training_params.patience}"
            )
            if patience_counter >= training_params.patience:
                logger.info("Early stopping threshold reached. Terminating training.")
                break

    checkpoint = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer)
    log_best_artifacts(checkpoint[3] if checkpoint[3] else best_eval_metrics)
    return checkpoint[0]

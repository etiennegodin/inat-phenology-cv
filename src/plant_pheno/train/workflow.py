from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import mlflow
import numpy as np
import torch
from torch.amp import autocast_mode, grad_scaler
from tqdm import tqdm

from ..utils import CLASS_ORDER, save_log
from .analysis import error_analysis, log_error_analysis
from .flow_control import ClassesObjectiveState, TrainingState, create_stage_states
from .metrics import (
    EpochMetrics,
    compute_metrics,
    log_attention_metrics,
    log_best_artifacts,
    log_epoch_metrics,
)
from .persistence import Checkpoint, CheckpointSaver

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
    accumulation_steps: int,
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
    logger.debug("")
    logger.debug(f"{'-' * 20} EPOCH {epoch} {'-' * 20} \n")

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [Train]",
        leave=False,
        dynamic_ncols=True,
    )

    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)

    n_batches = len(dataloader)

    for step, (images, labels, obs_ids) in enumerate(pbar):
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

        # True if this micro-batch completes an accumulation window,
        # or if it's the final (possibly partial) batch of the epoch.
        is_step_boundary = ((step + 1) % accumulation_steps == 0) or (
            step + 1 == n_batches
        )

        if device.type == "cuda":
            with autocast_mode.autocast(device_type=device.type, dtype=dtype):
                predictions, class_weights = model(images)
                raw_loss = criterion(predictions, labels)
                class_loss = torch.mean(raw_loss, dim=0)
                for i, c in enumerate(CLASS_ORDER):
                    classes_loss[c] += class_loss[i].item()  # unscaled, for logging
                loss = torch.mean(class_loss, dim=0)

                scaled_loss = loss / accumulation_steps  # normalize BEFORE backward

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                    if is_step_boundary:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                else:
                    scaled_loss.backward()
                    if is_step_boundary:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
        else:
            predictions, class_weights = model(images)
            raw_loss = criterion(predictions, labels)
            class_loss = torch.mean(raw_loss, dim=0)
            for i, c in enumerate(CLASS_ORDER):
                classes_loss[c] += class_loss[i].item()
            loss = torch.mean(class_loss, dim=0)

            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            if is_step_boundary:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        t0 = time.time()
        compute_time += t0 - t1

        current_loss = loss.item()
        total_loss += current_loss

        for class_name, batch_weights in class_weights.items():
            all_obs_weights[class_name].extend(batch_weights)

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
    pos_ratios: list[float],
) -> tuple[EpochMetrics, tuple[float, float]]:
    """Evaluate the model on validation set with rich multi-label metrics."""
    data_time = 0.0
    compute_time = 0.0

    total_loss = 0.0
    classes_loss = {la: 0.0 for la in CLASS_ORDER}

    all_preds_bin = []
    all_labels = []
    all_preds_raw = []
    all_obs_ids = []
    observations_attention_weights: dict[str, list[torch.Tensor]] = {}
    for c in CLASS_ORDER:
        observations_attention_weights[c] = []

    model.eval()

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [Val]  ",
        leave=False,
        dynamic_ncols=True,
    )

    t0 = time.time()
    with torch.no_grad():
        for step, (images, labels, obs_ids) in enumerate(pbar):
            all_obs_ids.extend(obs_ids)
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
            for class_name, batch_weights in class_weights.items():
                observations_attention_weights[class_name].extend(batch_weights)
            t0 = time.time()
            compute_time += t0 - t1
            pbar.set_postfix({"val_loss": f"{total_loss / (step + 1):.4f}"})

    all_labels_np = torch.cat(all_labels).numpy()
    all_preds_raw_np = torch.cat(all_preds_raw).numpy()

    val_loss = total_loss / len(dataloader)

    if mlflow.active_run():
        mlflow.log_metric("val/loss", val_loss, step=epoch)
        for k, v in classes_loss.items():
            c_loss = v / len(dataloader)
            mlflow.log_metric(f"val/{k}_loss", c_loss, step=epoch)

    eval_metrics = compute_metrics(
        all_preds_raw=all_preds_raw_np,
        all_labels=all_labels_np,
        pos_ratios=pos_ratios,
        val_loss=val_loss,
        prefix="val",
    )

    log_epoch_metrics(
        metrics=eval_metrics,
        all_preds_raw=all_preds_raw_np,
        all_labels=all_labels_np,
        epoch=epoch,
    )

    log_attention_metrics(epoch, observations_attention_weights, prefix="val")

    obs_paths_map = dataloader.dataset.get_obs_paths_map()

    error_report = error_analysis(
        all_obs_ids,
        all_preds_raw_np,
        all_labels_np,
        observations_attention_weights,
        eval_metrics,
        all_obs_paths=obs_paths_map,
    )
    log_error_analysis(error_report, epoch)
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
) -> Checkpoint:
    """Execute training pipeline over requested epochs with full logging."""
    best_eval_metrics = {}
    saver = CheckpointSaver()

    training_state = TrainingState(
        classes_states=ClassesObjectiveState(class_count=len(CLASS_ORDER)),
        training_params=training_params,
    )

    training_state.stages_states = create_stage_states(
        training_params=training_params,
        trainable_block_count=len(model.backbone.get_trainable_blocks())
        - training_params.starting_block,
    )

    log_step_interval = getattr(training_params, "log_step_interval", 10)

    logger.info(
        f"Starting training run: total epochs={training_params.epochs}, "
        f"patience={training_params.stopping_patience}, device={device.type}"
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
            accumulation_steps=training_params.accumulation_steps,
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
            pos_ratios=training_params.pos_ratios,
        )

        epoch_duration = time.time() - epoch_start_time

        if mlflow.active_run():
            mlflow.log_metric("time/eval_data", float(eval_times[0]), step=epoch)
            mlflow.log_metric("time/eval_compute", float(eval_times[1]), step=epoch)
            mlflow.log_metric("time/epoch_duration", float(epoch_duration), step=epoch)

        logger.info(
            f"Epoch {epoch:02d}/{training_params.epochs:02d} [{epoch_duration:.1f}s] - "
            f"pr_norm_excess_macro: {eval_metrics.pr_norm_excess_macro:.5f} | "
            f"ROC-AUC: {eval_metrics.roc_auc_macro:.3f} | "
            f"PR-AUC: {eval_metrics.pr_auc_macro:.3f} "
            f"| Best-F1: {eval_metrics.f1_macro_best:.3f}"
        )

        logger.debug(
            f"Timings -> Train Data: {train_times[0]:.1f}s, "
            f"Compute: {train_times[1]:.1f}s | "
            f"Eval Data: {eval_times[0]:.1f}s, "
            f"Compute: {eval_times[1]:.1f}s"
        )

        logger.debug(f"Eval metrics: {eval_metrics.pr_norm_excess_per_class()}")

        training_state.patience_counter(
            classes_metric=eval_metrics.pr_norm_excess_per_class()
        )

        if training_state.checkpoint_condition():
            # If any class improved, checkpoint
            logger.info(f"Saving checkpoint for epoch {epoch} to {checkpoint_path}")
            best_eval_metrics = eval_metrics
            checkpoint = Checkpoint(
                model=model, optimizer=optimizer, eval_metrics=best_eval_metrics
            )
            saver.save(
                checkpoint=checkpoint,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                to_mlflow=False,
                save_optimizer=False,
                async_transfer=True,
            )

        if training_state.stop_condition():
            logger.info(
                f"Pr_excess did not improve. "
                f"Patience: {training_state.classes_states} / "
                f"{training_params.stopping_patience}"
            )
            logger.info("Early stopping threshold reached. Terminating training.")
            break

        if training_params.unfreeze:
            last_lr = scheduler.get_last_lr()[0]
            training_state.unfreeze(
                epoch=epoch, model=model, optimizer=optimizer, last_lr=last_lr
            )

        save_log()

    save_log()
    saver.wait()
    checkpoint = Checkpoint.from_file(checkpoint_path, model=model)
    log_best_artifacts(checkpoint.eval_metrics)
    return checkpoint

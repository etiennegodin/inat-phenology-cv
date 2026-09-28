from dataclasses import replace

import mlflow
import optuna
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from ..config import (
    Config,
    ModelParams,
    OptimizerParams,
    SchedulerParams,
    TrainingParams,
)
from ..infra.seed import seed_everything
from ..train import run_training
from ..train.factory import DataPipeline


def _subsample_train_loader(
    data: DataPipeline, subsample_frac: float, seed: int
) -> DataLoader:
    """Return a DataLoader covering only a random fraction of the training set.

    Faithfully mirrors all settings (collate_fn, worker_init_fn, etc.) from the
    reference train loader so the batch format is identical to the full run.
    Handles both the ``batch_sampler`` path (use_max_images=True) and the plain
    ``batch_size`` path.
    """
    from ..infra.seed import seed_worker
    from ..train.batch_sampler import MaxImagesBatchSampler
    from ..train.dataloader import collate_fn as pheno_collate_fn

    train_set = data.datasets[0]
    ref = data.train_loader
    n_total = len(train_set)
    n_keep = max(1, int(n_total * subsample_frac))
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator)[:n_keep].tolist()

    # Shared kwargs that are always the same regardless of batching strategy
    common = dict(
        collate_fn=pheno_collate_fn,
        num_workers=ref.num_workers,
        pin_memory=ref.pin_memory,
        persistent_workers=ref.persistent_workers if ref.num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    # Detect whether the reference loader was built with a batch_sampler
    if ref.batch_sampler is not None and not isinstance(
        ref.batch_sampler, torch.utils.data.BatchSampler
    ):
        # use_max_images path: replicate MaxImagesBatchSampler on the subset
        subset = torch.utils.data.Subset(train_set, indices)
        bag_sizes = [train_set.bag_sizes[i] for i in indices]
        batch_sampler = MaxImagesBatchSampler(
            bag_sizes,
            max_images=ref.batch_sampler.max_images,
            shuffle=True,
            seed=seed,
        )
        return DataLoader(subset, batch_sampler=batch_sampler, **common)
    else:
        # Plain batch_size path
        sampler = SubsetRandomSampler(indices)
        batch_size = ref.batch_size or 1
        return DataLoader(train_set, batch_size=batch_size, sampler=sampler, **common)


def make_objective(
    configs: Config,
    data: DataPipeline,
    base_args,
    search_space: dict,
    subsample_frac: float = 1.0,
):
    """
    Factory that returns the function Optuna will call for each trial.
    WHAT HAPPENS EACH TRIAL:
    1. Optuna's `trial` object suggests hyperparameter values
    2. We build a fresh pipeline with those values
    3. Cross-validation scores the pipeline on training data
        (no test data touched here — that would be data leakage)
    4. Score is logged to MLflow as a child run
    5. Score is returned to Optuna to inform the next trial
    """

    def objective(trial: optuna.Trial) -> float:
        # ── Seed everything for this trial ────────────────────────────────────
        # Each trial gets a unique but deterministic seed derived from the base
        # seed so that: (a) model init, dropout masks, and data shuffling are
        # all varied across trials, and (b) any individual trial is fully
        # reproducible by re-running with the same base seed and trial number.
        base_seed = getattr(base_args, "seed", 42)
        trial_seed = base_seed + trial.number
        seed_everything(trial_seed, set_cuda_deterministic=False)

        # ── Step 1: Ask Optuna for hyperparameter suggestions ─────────────────
        # trial.suggest_* methods implement Bayesian optimization:
        # early trials explore randomly; later trials focus on promising regions.

        trial_params = {}
        for param_name, spec in search_space.items():
            suggest_type = spec["type"]
            kwargs = {k: v for k, v in spec.items() if k != "type"}

            if suggest_type == "int":
                trial_params[param_name] = trial.suggest_int(param_name, **kwargs)
            elif suggest_type == "float":
                trial_params[param_name] = trial.suggest_float(param_name, **kwargs)
            elif suggest_type == "categorical":
                trial_params[param_name] = trial.suggest_categorical(
                    param_name, **kwargs
                )

        # ── Step 2: Build pipeline with suggested params ──────────────────────

        # Build trial-specific dataclasses

        model_params = ModelParams(
            backbone=getattr(base_args, "backbone", "bioclip"),
            head_neurons=256,
            head_outputs=1,
            head_dropout_prob=trial_params.get("head_dropout_prob", 0.5),
            attention_neurons=trial_params.get("attention_neurons", 128),
            attention_dropout_prob=trial_params.get("attention_dropout_prob", 0.0),
            start_unfreezed=getattr(base_args, "start_unfreezed", 1),
            gated=trial_params.get("gated", True),
        )

        optim_params = OptimizerParams(
            base_lr=trial_params.get("base_lr", 1e-4),
        )
        epochs = getattr(base_args, "epochs", 10)
        warmup_epochs = getattr(base_args, "warmup_epochs", 3)
        scheduler_params = SchedulerParams(
            warmup_epochs=warmup_epochs,
            total_epoch=epochs,
        )

        training_params = TrainingParams(
            epochs=epochs,
            stopping_patience=getattr(base_args, "stopping_patience", 3),
            unfreeze=getattr(base_args, "unfreeze", True),
            unfreezing_patience=getattr(base_args, "unfreezing_patience", 3),
            unfreezing_cooldown=getattr(base_args, "unfreezing_cooldown", 3),
            starting_block=model_params.start_unfreezed,
            max_stages=getattr(base_args, "max_stages", 3),
            block_per_stage=getattr(base_args, "block_per_stage", 1),
            best_objective=1e-5,
            seed=trial_seed,
            log_step_interval=getattr(base_args, "log_step_interval", 10),
            pos_ratios=data.pos_ratios,
            backbone_decay=trial_params.get("backbone_decay", 0.90),
            accumulation_steps=configs.dataloaders_params.gradient_accumulation_steps,
        )

        # ── Step 3: Optionally subsample the training set for this trial ──────
        # Val/test loaders and pos_ratios are always kept at full size so that
        # evaluation is comparable across trials.
        if subsample_frac < 1.0:
            trial_train_loader = _subsample_train_loader(
                data,
                subsample_frac,
                seed=trial_seed,
            )
            trial_data = replace(data, train_loader=trial_train_loader)
        else:
            trial_data = data

        # 4. Execute inside a child nested MLflow run

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_param("trial_seed", trial_seed)
            if subsample_frac < 1.0:
                mlflow.log_param("subsample_frac", subsample_frac)
            try:
                checkpoint, _, _ = run_training(
                    configs=configs,
                    model_params=model_params,
                    optim_params=optim_params,
                    scheduler_params=scheduler_params,
                    training_params=training_params,
                    data=trial_data,
                    trial=trial,
                )
                metric_val = checkpoint.eval_metrics.pr_norm_excess_macro
                mlflow.log_metric("final_pr_norm_excess_macro", metric_val)
                mlflow.set_tag("optuna.state", "COMPLETE")
                return metric_val
            except optuna.TrialPruned as e:
                mlflow.set_tag("optuna.state", "PRUNED")
                raise e

    return objective

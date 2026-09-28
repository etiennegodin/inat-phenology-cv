import mlflow
import optuna

from ..config import (
    Config,
    ModelParams,
    OptimizerParams,
    SchedulerParams,
    TrainingParams,
)
from ..train import run_training
from ..train.factory import DataPipeline


def make_objective(configs: Config, data: DataPipeline, base_args, search_space: dict):
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
            start_epoch=None,
            best_objective=1e-5,
            seed=getattr(base_args, "seed", 42),
            log_step_interval=getattr(base_args, "log_step_interval", 10),
            pos_ratios=data.pos_ratios,
            backbone_decay=trial_params.get("backbone_decay", 0.90),
            accumulation_steps=configs.dataloaders_params.gradient_accumulation_steps,
        )

        # 3. Execute inside a child nested MLflow run

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial_params)
            checkpoint, _, _ = run_training(
                configs=configs,
                model_params=model_params,
                optim_params=optim_params,
                scheduler_params=scheduler_params,
                training_params=training_params,
                data=data,
                trial=trial,
            )
            metric_val = checkpoint.eval_metrics.pr_norm_excess_macro
            mlflow.log_metric("final_pr_norm_excess_macro", metric_val)
            return metric_val

    return objective

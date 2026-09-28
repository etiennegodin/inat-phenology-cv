import mlflow
import optuna

from ..config import (
    Config,
    ModelParams,
    OptimizerParams,
    SchedulerParams,
    TrainingParams,
)
from ..core import build_pipeline_model
from ..core.device import get_device
from . import (
    build_pipeline_optimizer,
    build_scheduler,
    log_experiment_metadata,
    workflow,
)
from .factory import DataPipeline, build_data_pipeline


def run_training(
    configs: Config,
    model_params: ModelParams,
    optim_params: OptimizerParams,
    scheduler_params: SchedulerParams,
    training_params: TrainingParams,
    data: DataPipeline | None = None,
    device=None,
    trial: optuna.Trial | None = None,
):

    # Initialise train modules
    device = get_device()

    # 1. Setup or reuse data
    if data is None:
        data = build_data_pipeline(
            configs, model_params.backbone, device=device, seed=training_params.seed
        )
    # Ensure pos_ratios is populated
    if not training_params.pos_ratios:
        training_params.pos_ratios = data.pos_ratios

    # 2. Build fresh trial components
    model = build_pipeline_model(device, model_params)
    optimizer = build_pipeline_optimizer(model, optim_params)
    scheduler = build_scheduler(optimizer, scheduler_params)

    if mlflow.active_run():
        mlflow.log_params(model_params.to_dict())
        mlflow.log_params(training_params.to_dict())
        mlflow.log_params(optim_params.to_dict())
        mlflow.log_params(scheduler_params.to_dict())

        log_experiment_metadata(
            model=model,
            train_dataset=data.datasets[0],
            val_dataset=data.datasets[1],
        )

    checkpoint = workflow.execute(
        device=device,
        model=model,
        train_loader=data.train_loader,
        val_loader=data.val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=data.criterion,
        checkpoint_path=configs.paths_params.checkpoint_path,
        training_params=training_params,
        trial=trial,
    )
    return checkpoint, model, data

import functools
import logging

import mlflow

logger = logging.getLogger(__name__)


def mlflow_track(model_log_func):
    """
    A decorator to wrap any training function with MLflow.
    model_log_func: the mlflow logging function (e.g., mlflow.sklearn.log_model)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Pull out MLflow-specific args (with defaults)
            experiment_name = kwargs.pop("experiment_name", "temp")
            run_name = kwargs.pop("run_name", "temp")
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

            mlflow.end_run()  # force-close any dangling run
            mlflow.set_experiment(experiment_name=experiment_name)
            with mlflow.start_run(run_name=run_name) as parent_run:
                logger.info(f"Starting ml flow run {parent_run.info.run_id}")
                # mlflow.log_dict(config.to_dict(), "config.json")

                # Log all keyword arguments as params
                # mlflow.log_params(kwargs)
                # Execute the actual training
                model = func(*args, **kwargs)
                # Log the resulting model
                model_log_func(model, "model")
                return model

        return wrapper

    return decorator

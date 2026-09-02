import mlflow

from .train.persistence import Checkpoint
from .utils import resolve_uri

# Set mlflow uri
mlflow.set_tracking_uri(resolve_uri())


def register_model(
    run_id: str, checkpoint_path: str, model_name: str = "my_cool_model"
):
    checkpoint = Checkpoint.from_file(checkpoint_path=checkpoint_path, run_id=run_id)
    try:
        # Re-open the finished run context to safely package and upload the model flavor
        with mlflow.start_run(run_id=run_id):
            # Log the instantiated model into the run
            # using MLflow's native PyTorch flavor
            mlflow.pytorch.log_model(
                pytorch_model=checkpoint.model,
                name="model",
                registered_model_name=model_name,
            )

        print(f"Successfully converted .pth and registered it to run {run_id}")

    except Exception as e:
        print(f"Registration failed: {e}")

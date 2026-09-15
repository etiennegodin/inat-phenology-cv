from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import torch

from .train.persistence import Checkpoint
from .utils import resolve_uri

if TYPE_CHECKING:
    import torch

    from .utils.params import ModelParams

# Set mlflow uri
mlflow.set_tracking_uri(resolve_uri())


class PhenologyPyfunc(mlflow.pyfunc.PythonModel):
    model_params: ModelParams
    class_thresholds: dict
    device: torch.device
    model: Any

    def load_context(self, context):
        from .train.factory import build_pipeline_model, get_device
        from .utils.params import ModelParams

        self.device = get_device()

        with open(context.artifacts["model_params"]) as f:
            self.model_params = ModelParams(**json.load(f))

        with open(context.artifacts["class_thresholds"]) as f:
            self.class_thresholds = json.load(f)

        state_dict_path = context.artifacts["model_state_dict"]

        model = build_pipeline_model(self.device, self.model_params)
        state_dict = torch.load(
            state_dict_path, map_location=self.device, weights_only=True
        )

        model.load_state_dict(state_dict=state_dict)
        model.to(device=self.device)
        model.eval()

        print()
        return super().load_context(context)

    def predict(self, context, model_input, params: dict[str, Any] | None = None):
        return super().predict(context, model_input, params)


def register_model(
    run_id: str,
    checkpoint_path: str,
    model_name: str = "my_cool_model",
    model_params: ModelParams | dict | None = None,
    device_type: str = "cuda",
):

    device = torch.device(device_type)
    checkpoint = Checkpoint.from_file(
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        model_params=model_params,
        device=device,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Model
        model_state_dict_path = Path(temp_dir) / "model_state_dict.pt"
        torch.save(checkpoint.model.state_dict(), model_state_dict_path)

        # Model params
        model_params_path = Path(temp_dir) / "model_params.json"
        with model_params_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint.model.params.to_dict(), f, indent=2)

        # Class thresholds
        class_thresholds_path = Path(temp_dir) / "class_thresholds.json"
        with class_thresholds_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint.eval_metrics.best_thresh, f, indent=2)

        with mlflow.start_run(run_id=run_id):
            model_info = mlflow.pyfunc.log_model(
                name=model_name,
                python_model=PhenologyPyfunc(),
                artifacts={
                    "model_state_dict": str(model_state_dict_path),
                    "model_params": str(model_params_path),
                    "class_thresholds": str(class_thresholds_path),
                },
            )
    print(model_info.model_uri)

    return
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

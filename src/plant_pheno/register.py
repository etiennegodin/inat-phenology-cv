from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import torch

from .inference import PhenologyPyfunc
from .infra import resolve_uri
from .train.persistence import Checkpoint

if TYPE_CHECKING:
    from .config import ModelParams

# Set mlflow uri
mlflow.set_tracking_uri(resolve_uri())


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
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Opening temp folder")

            # Model params
            model_params_path = Path(temp_dir) / "model_params.json"
            with model_params_path.open("w", encoding="utf-8") as f:
                json.dump(checkpoint.model.params.to_dict(), f, indent=2)

            print("Retrieved model params")

            # Class thresholds
            class_thresholds_path = Path(temp_dir) / "class_thresholds.json"
            with class_thresholds_path.open("w", encoding="utf-8") as f:
                json.dump(checkpoint.eval_metrics.best_thresh, f, indent=2)

            print("Retrieved class thresholds")

            # Model
            model_state_dict_path = Path(temp_dir) / "model_state_dict.pt"
            torch.save(checkpoint.model.state_dict(), model_state_dict_path)

            print("Retrieved artifacts")

            with mlflow.start_run(run_id=run_id):
                model_info = mlflow.pyfunc.log_model(
                    python_model=PhenologyPyfunc(),
                    artifacts={
                        "model_state_dict": str(model_state_dict_path),
                        "model_params": str(model_params_path),
                        "class_thresholds": str(class_thresholds_path),
                    },
                    registered_model_name=model_name,
                )

            print(f"Successfully converted .pth and registered it to {run_id}")
            print(model_info.model_uri)

    except Exception as e:
        print(f"Registration failed: {e}")

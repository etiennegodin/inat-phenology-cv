from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch
from PIL import Image

from .train.factory import build_pipeline_model
from .train.persistence import Checkpoint
from .utils import CLASS_ORDER, resolve_uri
from .utils.params import ModelParams

if TYPE_CHECKING:
    import pandas as pd
    import torch
    from torchvision.transforms.v2 import Compose

    from .train.model import PhenologyModel
    from .utils.params import ModelParams

# Set mlflow uri
mlflow.set_tracking_uri(resolve_uri())


class PhenologyPyfunc(mlflow.pyfunc.PythonModel):
    model_params: ModelParams
    class_thresholds: np.ndarray
    device: torch.device
    model: PhenologyModel
    transfrom: Compose

    @classmethod
    def from_inat_id(cls):
        raise NotImplementedError

    @classmethod
    def from_paths(cls):
        raise NotImplementedError

    @classmethod
    def from_batch(cls):
        raise NotImplementedError

    def load_context(self, context):

        requested_device = os.getenv("MODEL_DEVICE", "auto").lower()

        if requested_device == "cpu":
            self.device = torch.device("cpu")
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MODEL_DEVICE=cuda, but CUDA is unavailable")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(context.artifacts["model_params"]) as f:
            self.model_params = ModelParams(**json.load(f))

        with open(context.artifacts["class_thresholds"]) as f:
            threshold_dict = json.load(f)

        self.class_thresholds = np.array([v for v in threshold_dict.values()])

        # Rebuild model from params
        self.model = build_pipeline_model(self.device, self.model_params)

        # Set transforms
        _, self.transform = self.model.backbone.get_transforms()

        # Load weights
        state_dict_path = context.artifacts["model_state_dict"]
        state_dict = torch.load(
            state_dict_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict=state_dict)
        self.model.to(device=self.device)
        self.model.eval()

    def predict(
        self, context, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ):
        observations_attention_weights: dict[str, list[torch.Tensor]] = {}
        for c in CLASS_ORDER:
            observations_attention_weights[c] = []
        # Load images
        observations_ids = []
        image_bags = []
        for _, row in model_input.iterrows():
            observations_ids.append(row["observation_id"])
            image_paths = row["paths"]
            images = [self.load_image(p) for p in image_paths]
            image_bags.append(torch.stack(images))

        # Inference
        with torch.inference_mode():
            predictions, class_weights = self.model(image_bags)

        # Class predictions
        preds_raw = torch.sigmoid(predictions).detach().float().cpu().numpy()
        preds_bin = (preds_raw >= self.class_thresholds).astype(int)

        # Attention weights
        for class_name, batch_weights in class_weights.items():
            observations_attention_weights[class_name].extend(batch_weights)

        return preds_bin, observations_attention_weights

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.transform(image)


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

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch
from PIL import Image

from ..config import ModelParams
from ..core import build_pipeline_model

if TYPE_CHECKING:
    import pandas as pd
    import torch
    from torchvision.transforms.v2 import Compose

    from ..core.model import PhenologyModel


class PhenologyPyfunc(mlflow.pyfunc.PythonModel):
    model_params: ModelParams
    classes = list[str]
    class_thresholds: np.ndarray
    device: torch.device
    model: PhenologyModel
    transfrom: Compose

    def load_context(self, context):

        self._set_device()

        with open(context.artifacts["model_params"]) as f:
            self.model_params = ModelParams(**json.load(f))

        with open(context.artifacts["class_thresholds"]) as f:
            threshold_dict = json.load(f)

        self.classes = [[v for v in threshold_dict.keys()]]
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
    ) -> tuple[tuple[np.ndarray, np.ndarray], list[dict[str, list[float]]]]:
        # Load images
        observations_ids = []
        image_bags = []
        for _, row in model_input.iterrows():
            observations_ids.append(row["observation_id"])
            image_paths = row["paths"]
            images = [self._load_image(p) for p in image_paths]
            image_bags.append(torch.stack(images))

        # Inference
        with torch.inference_mode():
            predictions, class_weights = self.model.forward(image_bags)

        # Class predictions
        preds_raw = torch.sigmoid(predictions).detach().float().cpu().numpy()
        preds_bin = (preds_raw >= self.class_thresholds).astype(int)

        observations_attention_weights: list[dict[str, list[float]]] = []
        # Attention weights
        for o in range(preds_bin.shape[0]):
            obs_dict: dict[str, list] = {}
            for class_name, batch_weights in class_weights.items():
                obs_dict[class_name] = (
                    batch_weights[o].detach().cpu().squeeze(-1).tolist()
                )
            observations_attention_weights.append(obs_dict)

        return (preds_bin, preds_raw), observations_attention_weights

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def _set_device(self):
        requested_device = os.getenv("MODEL_DEVICE", "auto").lower()

        if requested_device == "cpu":
            self.device = torch.device("cpu")
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MODEL_DEVICE=cuda, but CUDA is unavailable")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

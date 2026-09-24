from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mlflow

from ..infra import resolve_uri

if TYPE_CHECKING:
    from ..config import InferenceParams

logger = logging.getLogger(__name__)


class BaseInferenceClient(ABC):
    model: mlflow.pyfunc.PythonModel
    model_uri: str
    params: InferenceParams

    def __init__(self, params: InferenceParams) -> None:
        self.params = params
        self._load_model()

    def _load_model(self):

        mlflow.set_tracking_uri(resolve_uri())
        # Construct the model URI
        self.model_uri = f"models:/{self.params.model_name}/{self.params.model_version}"

        # Load the model
        self.model = mlflow.pyfunc.load_model(self.model_uri)

    def _predict(self, model_input):
        """Validate"""
        return self.model.predict(model_input)

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Construct and validate model input"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mlflow

from ..infra import resolve_uri

if TYPE_CHECKING:
    from ..config import PathsParams

logger = logging.getLogger(__name__)


class BaseInferenceClient(ABC):
    model: mlflow.pyfunc.PythonModel
    model_name: str
    model_version: int
    paths: PathsParams

    def __init__(
        self, model_name: str, model_version: int, paths_params: PathsParams
    ) -> None:
        logger.debug("hhh")
        self.model_name = model_name
        self.model_version = model_version
        self.paths = paths_params
        self._load_model()

    def _load_model(self):

        mlflow.set_tracking_uri(resolve_uri())
        # Construct the model URI
        model_uri = f"models:/{self.model_name}/{self.model_version}"

        # Load the model
        self.model = mlflow.pyfunc.load_model(model_uri)

    def _predict(self, model_input):
        """Validate"""
        return self.model.predict(model_input)

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Construct and validate model input"""

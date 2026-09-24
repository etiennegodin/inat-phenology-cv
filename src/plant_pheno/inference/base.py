from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import mlflow
import numpy as np

from ..config import CLASS_ORDER
from ..data import DuckDBAdapter
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

    @abstractmethod
    def execute(self, *args, **kwargs):
        """Construct and validate model input"""

    def _predict(
        self, model_input
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, list[float]]]]:
        """Validate"""
        return self.model.predict(model_input)

    def _load_model(self):

        mlflow.set_tracking_uri(resolve_uri())
        # Construct the model URI
        self.model_uri = f"models:/{self.params.model_name}/{self.params.model_version}"

        # Load the model
        self.model = mlflow.pyfunc.load_model(self.model_uri)

    def _log_predictions(
        self,
        observation_ids: list[int],
        preds_raw: np.ndarray,
        preds_bin: np.ndarray,
        attention_weights_list: list[dict[str, list]],
    ):
        model_id = self._resolve_model_id()

        prediction_rows = []
        attention_rows = []

        for i, obs_id in enumerate(observation_ids):
            prediction_rows.append(
                (
                    obs_id,
                    model_id,
                    preds_raw[i].tolist(),
                    preds_bin[i].tolist(),
                )
            )
            attention_weights = attention_weights_list[i]
            for class_name in CLASS_ORDER:
                w = attention_weights[class_name]
                attention_rows.append((obs_id, model_id, class_name, w))

        with DuckDBAdapter(self.params.db_path) as con:
            con.executemany(
                """
                INSERT INTO serving.predictions
                    (observation_id, model_id, raw_preds, bin_preds)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (observation_id, model_id) DO UPDATE SET
                    raw_preds = EXCLUDED.raw_preds,
                    bin_preds = EXCLUDED.bin_preds,
                    predicted_at = now()
                """,
                prediction_rows,
            )
            con.executemany(
                """
                INSERT INTO serving.prediction_attention_weights
                    (observation_id, model_id, class_name, weights)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (observation_id, model_id, class_name) DO UPDATE SET
                weights = EXCLUDED.weights,
                predicted_at = now()
                """,
                attention_rows,
            )

    def _resolve_model_id(self) -> int:
        """Map this client's loaded model to a row in serving.models,
        keyed on model_uri. Inserts one the first time this exact model_uri
        is served from; cached on self so repeat execute() calls on the
        same client don't re-query."""
        if getattr(self, "_model_id", None) is not None:
            return self._model_id

        with DuckDBAdapter(self.params.db_path) as con:
            row = con.execute(
                "SELECT model_id FROM serving.models WHERE model_uri = ?",
                [self.model_uri],
            ).fetchone()

            if row is None:
                new_id = con.execute(
                    "SELECT COALESCE(MAX(model_id), 0) + 1 FROM serving.models"
                ).fetchone()[0]
                con.execute(
                    """
                    INSERT INTO serving.models
                        (model_id, model_name, model_version, model_uri, thresholds)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        new_id,
                        self.params.model_name,
                        self.params.model_version,
                        self.model_uri,
                        self.model._model_impl.python_model.class_thresholds,
                    ],
                )
                row = (new_id,)

        self._model_id = row[0]
        return self._model_id

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import InferenceParams, IngestPhotosParams
from .predictor import PhenologyPredictor
from .sync import InatDataSynchronizer

if TYPE_CHECKING:
    import mlflow


class InatInferencePipeline:
    """
    High-level orchestrator that takes observation URLs, ensures images
    are present locally, executes model inference, and returns predictions.
    """

    def __init__(
        self, sync: InatDataSynchronizer, predictor: mlflow.pyfunc.PythonModel
    ) -> None:

        self.sync = sync
        self.predictor = predictor

    @classmethod
    def from_params(cls, params: InferenceParams) -> InatInferencePipeline:
        return cls(
            sync=InatDataSynchronizer(
                db_path=params.db_path,
                photo_target_dir=params.photo_target_dir,
                sql_dir=params.sql_dir,
            ),
            predictor=PhenologyPredictor(
                model_name=params.model_name,
                model_version=params.model_version,
                db_path=params.db_path,
            ),
        )

    def execute(
        self,
        urls: list[str] | str,
        photos_params: IngestPhotosParams | None = None,
        rate: int = 10,
    ):
        if isinstance(urls, str):
            urls = [urls]
        if photos_params is None:
            photos_params = IngestPhotosParams()

        # 1. Sync data and photos to local disk
        df_ready = self.sync.prepare_observation_images(
            urls=urls,
            photos_params=photos_params,
            rate=rate,
        )

        # 2. Run model inference & log to DuckDB
        return self.predictor.predict_and_log(df_ready)

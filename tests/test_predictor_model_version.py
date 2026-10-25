from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from mlflow import MlflowException

from plant_pheno.config import PredictParams
from plant_pheno.inference.predictor import PhenologyPredictor


def test_predict_defaults_to_latest_registered_model_version():
    client = Mock()
    client.search_model_versions.return_value = [
        SimpleNamespace(version="10"),
        SimpleNamespace(version="4"),
        SimpleNamespace(version="2"),
    ]
    load_model = Mock()
    mlflow_mock = SimpleNamespace(
        set_tracking_uri=Mock(),
        pyfunc=SimpleNamespace(load_model=load_model),
    )

    with (
        patch("plant_pheno.inference.predictor.MlflowClient", return_value=client),
        patch("plant_pheno.inference.predictor.mlflow", mlflow_mock),
    ):
        predictor = PhenologyPredictor("cv_inat", None, "inference.duckdb")
        predictor._load_model()

    client.search_model_versions.assert_called_once_with(
        "name='cv_inat'",
        max_results=1,
        order_by=["version_number DESC"],
    )
    assert predictor.model_version == 10
    assert predictor.model_uri == "models:/cv_inat/10"
    load_model.assert_called_once_with("models:/cv_inat/10")


def test_predict_preserves_explicit_model_version():
    mlflow_mock = SimpleNamespace(
        set_tracking_uri=Mock(),
        pyfunc=SimpleNamespace(load_model=Mock()),
    )

    with (
        patch("plant_pheno.inference.predictor.MlflowClient") as client,
        patch("plant_pheno.inference.predictor.mlflow", mlflow_mock),
    ):
        predictor = PhenologyPredictor("cv_inat", 3, "inference.duckdb")
        predictor._load_model()

    client.assert_not_called()
    assert predictor.model_version == 3
    assert predictor.model_uri == "models:/cv_inat/3"


def test_predict_errors_when_model_has_no_registered_versions():
    client = Mock()
    client.search_model_versions.return_value = []
    mlflow_mock = SimpleNamespace(set_tracking_uri=Mock())

    with (
        patch("plant_pheno.inference.predictor.MlflowClient", return_value=client),
        patch("plant_pheno.inference.predictor.mlflow", mlflow_mock),
        pytest.raises(MlflowException, match="No registered versions.*cv_inat"),
    ):
        PhenologyPredictor("cv_inat", None, "inference.duckdb")._load_model()


def test_predict_config_defaults_to_latest_but_accepts_version_override():
    default_config = PredictParams()
    explicit_config = PredictParams(model_name="cv_inat_v2", model_version=3)

    assert default_config.model_name == "cv_pheno_bioclip"
    assert default_config.model_version is None
    assert explicit_config.model_name == "cv_inat_v2"
    assert explicit_config.model_version == 3

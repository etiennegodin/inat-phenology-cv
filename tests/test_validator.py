import pytest
from PIL import Image
from pydantic import ValidationError

from plant_pheno.inference.validator import ObservationRecord, validate_records


@pytest.fixture
def mock_image_path(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 32), color=(0, 128, 255)).save(path)
    return path


def test_validate_records():
    input_records = [
        {"observation_id": 12, "ancestor_ids": [1, 2, 3]},
        {"observation_id": "12", "ancestor_ids": [1, 2, 3]},
        {"observation_id": 12, "ancestor_ids": []},
    ]

    validate_records(input_records)


def test_validate_records_raise():
    with pytest.raises(ValidationError):
        ObservationRecord.model_validate(
            {
                "observation_id": "obs-1",
                "value": "not-a-number",
            }
        )


def test_is_flowering_plant_record(mock_image_path):
    input_record = {
        "observation_id": 12,
        "ancestor_ids": [47125],
        "uuid": "1234",
        "paths": [mock_image_path],
    }
    record = ObservationRecord(**input_record)
    assert record.is_flowering_plant


def test_is_not_flowering_plant_record(mock_image_path):
    input_record = {
        "observation_id": 12,
        "ancestor_ids": [1],
        "uuid": "1234",
        "paths": [mock_image_path],
    }
    record = ObservationRecord(**input_record)
    assert not record.is_flowering_plant

import numpy as np
import pandas as pd
import torch

from plant_pheno.core.pyfunc import PhenologyPyfunc


class DeviceCheckingModel:
    def __init__(self, device):
        self.device = device

    def forward(self, image_bags):
        assert all(images.device == self.device for images in image_bags)
        predictions = torch.tensor([[2.0]])
        weights = {"class": torch.tensor([[[0.4], [0.6]]])}
        return predictions, weights


def test_predict_moves_image_bags_to_model_device():
    pyfunc = PhenologyPyfunc()
    pyfunc.device = torch.device("meta")
    pyfunc.model = DeviceCheckingModel(pyfunc.device)
    pyfunc.class_thresholds = np.array([0.5])
    pyfunc._load_image = lambda _: torch.ones(3, 2, 2)

    predictions, _, _ = pyfunc.predict(
        None,
        pd.DataFrame([{"observation_id": 1, "paths": ["image.jpg"]}]),
    )

    np.testing.assert_array_equal(predictions, np.array([[1]]))

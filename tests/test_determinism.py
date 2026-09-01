import random

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2

from pytorch_pipeline.train.batch_sampler import MaxImagesBatchSampler
from pytorch_pipeline.train.dataset import split_dataset
from pytorch_pipeline.utils.params import DatasetParams
from pytorch_pipeline.utils.seed import seed_everything, seed_worker


def test_seed_everything():
    seed_everything(1234, set_cuda_deterministic=False)
    py_rand1 = random.random()
    np_rand1 = np.random.rand()
    torch_rand1 = torch.rand(1).item()

    seed_everything(1234, set_cuda_deterministic=False)
    py_rand2 = random.random()
    np_rand2 = np.random.rand()
    torch_rand2 = torch.rand(1).item()

    assert py_rand1 == py_rand2
    assert np_rand1 == np_rand2
    assert torch_rand1 == torch_rand2


def test_split_dataset_determinism():
    data = {
        "observation_id": list(range(100)),
        "label": [[1, 0, 0]] * 100,
        "path": ["/path/to/img.jpg"] * 100,
    }
    df = pd.DataFrame(data)
    params = DatasetParams(train_frac=0.8, val_frac=0.1, test_frac=0.1)

    train1, val1, test1 = split_dataset(df, params, seed=42)
    train2, val2, test2 = split_dataset(df, params, seed=42)
    train3, val3, test3 = split_dataset(df, params, seed=999)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)
    assert not train1["observation_id"].equals(train3["observation_id"])


def test_max_images_batch_sampler_determinism():
    bag_sizes = [2, 5, 1, 4, 3, 6, 2, 1, 5, 3]
    sampler1 = MaxImagesBatchSampler(bag_sizes, max_images=6, shuffle=True, seed=42)
    sampler2 = MaxImagesBatchSampler(bag_sizes, max_images=6, shuffle=True, seed=42)
    sampler3 = MaxImagesBatchSampler(bag_sizes, max_images=6, shuffle=True, seed=99)

    batches1 = list(sampler1)
    batches2 = list(sampler2)
    batches3 = list(sampler3)

    assert batches1 == batches2
    assert batches1 != batches3


def test_v2_transforms_worker_determinism():
    transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=45),
        ]
    )

    dummy_image = torch.ones(3, 32, 32)

    # Seed worker process simulation
    torch.manual_seed(42)
    seed_worker(0)
    out1 = transform(dummy_image.clone())

    torch.manual_seed(42)
    seed_worker(0)
    out2 = transform(dummy_image.clone())

    assert torch.equal(out1, out2)

import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from plant_pheno import cache as cache_module
from plant_pheno.config import DataLoadersParams, DatasetParams, ModelParams
from plant_pheno.core.model import CrossAttentionModel, MilAttentionModel
from plant_pheno.train import dataset as dataset_module
from plant_pheno.train import factory as factory_module
from plant_pheno.train.batch_sampler import MaxImagesBatchSampler
from plant_pheno.train.dataloader import collate_fn_cached
from plant_pheno.train.dataset import (
    CachedEmbeddingDataset,
    UncachedPhenologyDataset,
    build_cached_datasets,
    build_full_dataset,
)


def test_cached_embedding_dataset_and_collate():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create dummy cached .pt files
        # Step 0: 2 observations (obs 1: 3 images, obs 2: 2 images) => total 5 images
        payload_0 = {
            "embeddings": torch.randn(5, 768, dtype=torch.bfloat16),
            "bag_sizes": [3, 2],
            "obs_ids": [101, 102],
            "kind": "cls",
            "to_layer": "full",
        }
        torch.save(payload_0, tmp_path / "0.pt")

        # Step 1: 1 observation (obs 3: 4 images) => total 4 images
        payload_1 = {
            "embeddings": torch.randn(4, 768, dtype=torch.bfloat16),
            "bag_sizes": [4],
            "obs_ids": [103],
            "kind": "cls",
            "to_layer": "full",
        }
        torch.save(payload_1, tmp_path / "1.pt")

        # Create DataFrame with labels
        df = pd.DataFrame(
            {
                "observation_id": [101, 102, 103, 104],
                "label": [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
            }
        )

        dataset_params = DatasetParams(
            idx_col="observation_id",
            label_col="label",
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )

        dataset = CachedEmbeddingDataset(
            cache_dir=tmp_path,
            obs_df=df,
            dataset_params=dataset_params,
        )

        assert len(dataset) == 3
        assert dataset.bag_sizes == [3, 2, 4]

        emb, label, obs_id = dataset[0]
        assert emb.shape == (3, 768)
        assert emb.dtype == torch.bfloat16
        assert obs_id == 101
        assert torch.equal(label, torch.tensor([1.0, 0.0, 0.0]))

        # Test MaxImagesBatchSampler compatibility
        sampler = MaxImagesBatchSampler(
            dataset.bag_sizes,
            max_images=5,
            shuffle=False,
        )
        batches = list(sampler)
        # obs 0 has 3 imgs, obs 1 has 2 imgs -> batch [0, 1] (sum 5)
        # obs 2 has 4 imgs -> batch [2] (sum 4)
        assert batches == [[0, 1], [2]]

        # Test collate_fn_cached
        batch_items = [dataset[i] for i in batches[0]]
        batch_embeddings, batch_labels, batch_obs_ids = collate_fn_cached(batch_items)
        assert len(batch_embeddings) == 2
        assert batch_embeddings[0].shape == (3, 768)
        assert batch_embeddings[1].shape == (2, 768)
        assert batch_labels.shape == (2, 3)
        assert batch_obs_ids == [101, 102]


def test_mil_attention_model_forward_from_embeddings():
    params = ModelParams(backbone="bioclip", architecture="mil_pooling")
    model = MilAttentionModel(params)
    model.eval()

    # Create dummy embeddings for 2 observations
    # obs 1: 2 images, obs 2: 3 images
    embed_dim = model.backbone.output_dim
    embeddings = [torch.randn(2, embed_dim), torch.randn(3, embed_dim)]

    preds, weights = model.forward_from_embeddings(embeddings)
    assert preds.shape == (2, 3)
    assert len(weights) == 3
    for c, w_list in weights.items():
        assert len(w_list) == 2
        assert w_list[0].shape == (2, 1)
        assert w_list[1].shape == (3, 1)


def test_cross_attention_model_forward_from_embeddings():
    params = ModelParams(
        backbone="bioclip",
        architecture="cross_attention",
        proj_dim=64,
        cross_attn_heads=2,
    )
    model = CrossAttentionModel(params)
    model.eval()

    # Pre-computed patch embeddings: shape (N_i, num_patches, patch_dim)
    # BioClip patch_dim is 768, assume 4 patches
    patch_embeddings = [
        torch.randn(2, 4, 768),
        torch.randn(1, 4, 768),
    ]

    preds, weights = model.forward_from_embeddings(patch_embeddings)
    assert preds.shape == (2, 3)
    assert len(weights) == 3


def test_cached_dataset_raises_on_nonexistent_or_empty_dir():
    df = pd.DataFrame(
        {
            "observation_id": [101],
            "label": [[1, 0, 0]],
        }
    )
    dataset_params = DatasetParams(
        idx_col="observation_id",
        label_col="label",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )

    # 1. Non-existent directory
    non_existent_path = Path("/tmp/does_not_exist_cached_emb_12345")
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        CachedEmbeddingDataset(
            cache_dir=non_existent_path,
            obs_df=df,
            dataset_params=dataset_params,
        )

    # 2. Existing but empty directory (no .pt files)
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(FileNotFoundError, match="No cached embedding"):
            CachedEmbeddingDataset(
                cache_dir=Path(empty_dir),
                obs_df=df,
                dataset_params=dataset_params,
            )


def test_cached_splits_load_embeddings_once_and_share_storage(tmp_path, monkeypatch):
    observation_ids = list(range(10))
    dataset_params = DatasetParams(
        idx_col="observation_id",
        label_col="label",
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
    )
    df = pd.DataFrame(
        {
            "observation_id": observation_ids,
            "label": [[1, 0, 0]] * len(observation_ids),
            "path": [["image.jpg"]] * len(observation_ids),
        }
    )
    torch.save(
        {
            "embeddings": torch.randn(len(observation_ids), 2, 4, dtype=torch.bfloat16),
            "bag_sizes": [1] * len(observation_ids),
            "obs_ids": observation_ids,
        },
        tmp_path / "0.pt",
    )

    load_calls = 0
    original_load = CachedEmbeddingDataset._load

    def count_loads(self, cache_dir, obs_df, params):
        nonlocal load_calls
        load_calls += 1
        return original_load(self, cache_dir, obs_df, params)

    monkeypatch.setattr(CachedEmbeddingDataset, "_load", count_loads)
    monkeypatch.setattr(
        dataset_module,
        "_prepare_build_dataset",
        lambda configs, seed: (df, False),
    )

    train_set, val_set, test_set = build_cached_datasets(
        SimpleNamespace(dataset_params=dataset_params), tmp_path
    )

    assert load_calls == 1
    assert len(train_set) + len(val_set) + len(test_set) == len(df)
    assert all(
        obs.embeddings.dtype == torch.bfloat16
        for dataset in (train_set, val_set, test_set)
        for obs in dataset._obs
    )
    backing_storages = {
        obs.embeddings.untyped_storage().data_ptr()
        for dataset in (train_set, val_set, test_set)
        for obs in dataset._obs
    }
    assert len(backing_storages) == 1


def test_full_dataset_for_embedding_cache_streams_images(monkeypatch):
    dataset_params = DatasetParams(
        idx_col="observation_id",
        label_col="label",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )
    df = pd.DataFrame(
        {
            "observation_id": [101],
            "label": [[1, 0, 0]],
            "path": [["image.jpg"]],
        }
    )
    prepare_args = {}

    def prepare_dataset(configs, seed, cache_images):
        prepare_args["cache_images"] = cache_images
        return df, False

    monkeypatch.setattr(dataset_module, "_prepare_build_dataset", prepare_dataset)

    dataset = build_full_dataset(
        SimpleNamespace(dataset_params=dataset_params), (None, None)
    )

    assert isinstance(dataset, UncachedPhenologyDataset)
    assert prepare_args["cache_images"] is False


def test_cache_execute_saves_embeddings_as_bfloat16(tmp_path, monkeypatch):
    class FakeBackbone:
        trainable_block_count = 12
        has_patch_support = True

        def get_transforms(self):
            return None, None

        def encode_patches(self, images, n_blocks=None):
            return torch.ones(images.size(0), 3, 768, dtype=torch.float32)

    class FakeSaver:
        payload = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def save(self, payload, path):
            self.payload = payload

    cache_dir = tmp_path / "cached_embeddings" / "dinov3" / "patches" / "to_layer_full"
    cache_dir.mkdir(parents=True)
    stale_file = cache_dir / "stale.pt"
    stale_file.touch()
    saver = FakeSaver()
    monkeypatch.setattr(cache_module, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        cache_module,
        "build_backbone",
        lambda *args, **kwargs: FakeBackbone(),
    )
    monkeypatch.setattr(
        cache_module,
        "build_full_dataset",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        cache_module,
        "build_single_dataloader",
        lambda **kwargs: [([torch.ones(2, 3, 224, 224)], torch.zeros(1, 3), [101])],
    )
    monkeypatch.setattr(cache_module, "LocalTorchSaver", lambda max_pending: saver)
    monkeypatch.setattr(cache_module, "cuda_cleanup", lambda: nullcontext())

    configs = SimpleNamespace(
        dataset_params=SimpleNamespace(
            cache=SimpleNamespace(
                cache_to_layer=0,
                cache_views=1,
                force_recache=True,
            )
        ),
        paths_params=SimpleNamespace(data_root=str(tmp_path)),
        dataloaders_params=DataLoadersParams(
            batch_size=256,
            max_images=1024,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            use_max_images=True,
        ),
    )
    model_params = SimpleNamespace(
        backbone="dinov3",
        architecture="cross_attention",
        dinov3=SimpleNamespace(checkpoint_path=None),
    )

    cache_module.execute(configs, model_params, seed=42, force_recache=True)

    assert saver.payload["embeddings"].dtype == torch.bfloat16
    assert not stale_file.exists()


def test_cached_data_pipeline_disables_workers_without_mutating_config(
    monkeypatch, tmp_path
):
    dataset_params = DatasetParams(
        idx_col="observation_id",
        label_col="label",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )
    loader_params = DataLoadersParams(
        batch_size=256,
        max_images=1024,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        use_max_images=True,
    )
    configs = SimpleNamespace(
        dataset_params=dataset_params,
        dataloaders_params=loader_params,
    )
    datasets = (object(), object(), object())
    seen = {}

    monkeypatch.setattr(
        factory_module, "build_cached_datasets", lambda *args, **kwargs: datasets
    )
    monkeypatch.setattr(
        factory_module, "get_pos_weights", lambda *args, **kwargs: torch.ones(3)
    )
    monkeypatch.setattr(factory_module, "get_pos_ratios", lambda dataset: [])

    def capture_loader_params(datasets, params, **kwargs):
        seen["params"] = params
        return "train", "val", "test"

    monkeypatch.setattr(
        factory_module, "build_pipeline_dataloaders", capture_loader_params
    )

    factory_module.build_cached_data_pipeline(configs, tmp_path, device="cpu")

    assert seen["params"].num_workers == 0
    assert seen["params"].persistent_workers is False
    assert loader_params.num_workers == 2
    assert loader_params.persistent_workers is True

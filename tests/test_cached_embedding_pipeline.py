import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from plant_pheno import cache as cache_module
from plant_pheno.config import DataLoadersParams, DatasetParams, ModelParams
from plant_pheno.core import model as model_module
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


@pytest.mark.parametrize(
    ("architecture", "model_cls"),
    [
        ("mil_pooling", MilAttentionModel),
        ("cross_attention", CrossAttentionModel),
    ],
)
def test_partial_cached_forward_trains_only_uncached_tail(
    monkeypatch, architecture, model_cls
):
    class TinyBackbone(torch.nn.Module):
        output_dim = 4
        patch_dim = 4
        has_patch_support = True
        supports_token_continuation = True
        trainable_block_count = 3

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(self.trainable_block_count)]
            )

        def get_trainable_blocks(self):
            return self.encoder.blocks

        def continue_intermediate_tokens(self, tokens, start_block, *, use_patches):
            for block in self.get_trainable_blocks()[start_block:]:
                tokens = block(tokens)
            return tokens[:, 1:] if use_patches else tokens[:, 0]

    monkeypatch.setattr(
        model_module, "build_backbone", lambda *args, **kwargs: TinyBackbone()
    )
    params = ModelParams(
        backbone="test",
        architecture=architecture,
        start_unfreezed=1,
        proj_dim=4,
        cross_attn_heads=2,
    )
    model = model_cls(params, cache_to_layer=1)
    model.train()
    cached_tokens = [
        torch.randn(2, 3, 4),
        torch.randn(1, 3, 4),
    ]

    predictions, _ = model.forward_from_embeddings(cached_tokens)
    predictions.sum().backward()

    blocks = model.backbone.get_trainable_blocks()
    assert blocks[0].weight.grad is None
    assert blocks[1].weight.grad is None
    assert blocks[2].weight.grad is not None
    assert blocks[2].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("model_cls", [MilAttentionModel, CrossAttentionModel])
def test_partial_cache_rejects_more_trainable_blocks_than_tail(monkeypatch, model_cls):
    class TinyBackbone(torch.nn.Module):
        output_dim = 4
        patch_dim = 4
        has_patch_support = True
        supports_token_continuation = True
        trainable_block_count = 3

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(self.trainable_block_count)]
            )

        def get_trainable_blocks(self):
            return self.encoder.blocks

    monkeypatch.setattr(
        model_module, "build_backbone", lambda *args, **kwargs: TinyBackbone()
    )
    params = ModelParams(backbone="test", start_unfreezed=2)

    with pytest.raises(ValueError, match="exceeds the 1 uncached tail blocks"):
        model_cls(params, cache_to_layer=1)


def test_full_depth_cached_model_freezes_backbone(monkeypatch):
    class TinyBackbone(torch.nn.Module):
        output_dim = 4
        trainable_block_count = 2
        supports_token_continuation = False

        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.blocks = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(self.trainable_block_count)]
            )

        def get_trainable_blocks(self):
            return self.encoder.blocks

        def log_trainable_blocks(self):
            pass

    monkeypatch.setattr(
        model_module, "build_backbone", lambda *args, **kwargs: TinyBackbone()
    )
    model = model_module.build_pipeline_model(
        torch.device("cpu"),
        ModelParams(backbone="test", start_unfreezed=1),
        cache_to_layer=0,
    )

    assert all(
        not parameter.requires_grad for parameter in model.backbone.encoder.parameters()
    )


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


def test_cached_dataset_rejects_legacy_partial_cache_format(tmp_path):
    torch.save(
        {
            "embeddings": torch.randn(1, 2, 4),
            "bag_sizes": [1],
            "obs_ids": [101],
        },
        tmp_path / "0.pt",
    )
    df = pd.DataFrame({"observation_id": [101], "label": [[1, 0, 0]]})
    params = DatasetParams(
        idx_col="observation_id",
        label_col="label",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )

    with pytest.raises(ValueError, match="expected 'tokens_v1'.*force_recache"):
        CachedEmbeddingDataset(
            cache_dir=tmp_path,
            obs_df=df,
            dataset_params=params,
            expected_cache_format="tokens_v1",
        )


def test_cached_dataset_rejects_partial_cache_with_wrong_depth(tmp_path):
    torch.save(
        {
            "embeddings": torch.randn(1, 2, 4),
            "bag_sizes": [1],
            "obs_ids": [101],
            "cache_format": "tokens_v1",
            "cache_to_layer": 2,
        },
        tmp_path / "0.pt",
    )
    df = pd.DataFrame({"observation_id": [101], "label": [[1, 0, 0]]})
    params = DatasetParams(
        idx_col="observation_id",
        label_col="label",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
    )

    with pytest.raises(ValueError, match="expected 1.*force_recache"):
        CachedEmbeddingDataset(
            cache_dir=tmp_path,
            obs_df=df,
            dataset_params=params,
            expected_cache_format="tokens_v1",
            expected_cache_to_layer=1,
        )


def test_cache_command_writes_intermediate_token_payload(tmp_path, monkeypatch):
    class TinyBackbone(torch.nn.Module):
        trainable_block_count = 3
        supports_token_continuation = True
        has_patch_support = False

        def get_transforms(self):
            return None, None

        def encode_intermediate_tokens(self, images, n_blocks):
            assert n_blocks == 2
            return images.mean(dim=(-2, -1)).unsqueeze(1)

    backbone = TinyBackbone()
    monkeypatch.setattr(cache_module, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        cache_module,
        "build_backbone",
        lambda *args, **kwargs: backbone,
    )
    monkeypatch.setattr(
        cache_module, "build_full_dataset", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        cache_module,
        "build_single_dataloader",
        lambda **kwargs: [
            (
                [torch.randn(2, 3, 4, 4), torch.randn(1, 3, 4, 4)],
                None,
                [101, 102],
            )
        ],
    )
    configs = SimpleNamespace(
        dataset_params=SimpleNamespace(
            cache=SimpleNamespace(cache_to_layer=1, cache_views=1)
        ),
        paths_params=SimpleNamespace(data_root=str(tmp_path)),
        dataloaders_params=DataLoadersParams(
            batch_size=2,
            max_images=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            use_max_images=False,
        ),
    )

    cache_module.execute(
        configs=configs,
        model_params=ModelParams(backbone="tiny"),
        seed=42,
    )

    cache_file = (
        tmp_path
        / "cached_embeddings"
        / "tiny"
        / "cls"
        / "to_layer_1"
        / "tokens_v1"
        / "0.pt"
    )
    payload = torch.load(cache_file, map_location="cpu")
    assert payload["embeddings"].shape == (3, 1, 3)
    assert payload["cache_format"] == "tokens_v1"
    assert payload["cache_to_layer"] == 1
    assert payload["bag_sizes"] == [2, 1]


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

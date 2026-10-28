import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch

from plant_pheno.config import DatasetParams, ModelParams
from plant_pheno.core.model import CrossAttentionModel, MilAttentionModel
from plant_pheno.train.batch_sampler import MaxImagesBatchSampler
from plant_pheno.train.dataloader import collate_fn_cached
from plant_pheno.train.dataset import CachedEmbeddingDataset


def test_cached_embedding_dataset_and_collate():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create dummy cached .pt files
        # Step 0: 2 observations (obs 1: 3 images, obs 2: 2 images) => total 5 images
        payload_0 = {
            "embeddings": torch.randn(5, 768),
            "bag_sizes": [3, 2],
            "obs_ids": [101, 102],
            "kind": "cls",
            "to_layer": "full",
        }
        torch.save(payload_0, tmp_path / "0.pt")

        # Step 1: 1 observation (obs 3: 4 images) => total 4 images
        payload_1 = {
            "embeddings": torch.randn(4, 768),
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

import torch

from pytorch_pipeline.train.factory import (
    build_pipeline_model,
    build_pipeline_optimizer,
)
from pytorch_pipeline.train.persistence import load_checkpoint, save_checkpoint
from pytorch_pipeline.utils.params import ModelParams, OptimizerParams


def test_save_and_load_full_checkpoint(tmp_path):
    device = torch.device("cpu")
    model_params = ModelParams(
        backbone="efficientnet",
        head_neurons=32,
        head_outputs=3,
        head_dropout_prob=0.1,
        attention_neurons=16,
        last_blocks=0,
    )
    model = build_pipeline_model(device, model_params)
    optimizer = build_pipeline_optimizer(model, OptimizerParams())

    ckpt_path = str(tmp_path)
    eval_metrics = {"val/loss": 0.42, "val/roc_auc_macro": 0.88}

    save_checkpoint(
        ckpt_path,
        epoch=2,
        model=model,
        optimizer=optimizer,
        eval_metrics=eval_metrics,
    )

    # Reload checkpoint
    model_loaded, opt_loaded, epoch, metrics, run_id = load_checkpoint(
        ckpt_path, model=model, optimizer=optimizer
    )

    assert epoch == 2
    assert metrics["val/loss"] == 0.42
    assert metrics["val/roc_auc_macro"] == 0.88


def test_save_and_load_checkpoint_no_optimizer(tmp_path):
    device = torch.device("cpu")
    model_params = ModelParams(
        backbone="efficientnet",
        head_neurons=32,
        head_outputs=3,
        head_dropout_prob=0.1,
        attention_neurons=16,
        last_blocks=0,
    )
    model = build_pipeline_model(device, model_params)

    ckpt_path = str(tmp_path)
    eval_metrics = {"val/loss": 0.42, "val/roc_auc_macro": 0.88}

    save_checkpoint(
        ckpt_path,
        epoch=2,
        model=model,
        eval_metrics=eval_metrics,
    )
    # Reload checkpoint
    model_loaded, opt_loaded, epoch, metrics, run_id = load_checkpoint(
        ckpt_path,
        model=model,
    )

    assert epoch == 2
    assert metrics["val/loss"] == 0.42
    assert metrics["val/roc_auc_macro"] == 0.88


def test_save_and_load_checkpoint_no_metrics(tmp_path):
    device = torch.device("cpu")
    model_params = ModelParams(
        backbone="efficientnet",
        head_neurons=32,
        head_outputs=3,
        head_dropout_prob=0.1,
        attention_neurons=16,
        last_blocks=0,
    )
    model = build_pipeline_model(device, model_params)
    optimizer = build_pipeline_optimizer(model, OptimizerParams())

    ckpt_path = str(tmp_path)

    save_checkpoint(ckpt_path, epoch=2, model=model, optimizer=optimizer)
    # Reload checkpoint
    model_loaded, opt_loaded, epoch, metrics, run_id = load_checkpoint(
        ckpt_path, model=model, optimizer=optimizer
    )

    assert epoch == 2

import torch

from pytorch_pipeline.train.factory import (
    build_pipeline_model,
    build_pipeline_optimizer,
)
from pytorch_pipeline.train.metrics import EpochMetrics
from pytorch_pipeline.train.persistence import Checkpoint
from pytorch_pipeline.utils.params import ModelParams, OptimizerParams


def create_dummy_model():
    device = torch.device("cpu")
    model_params = ModelParams(
        backbone="efficientnet",
        head_neurons=32,
        head_outputs=3,
        head_dropout_prob=0.1,
        attention_neurons=16,
        attention_dropout_prob=0.1,
        last_blocks=0,
    )
    return build_pipeline_model(device, model_params)


def create_dummy_metrics() -> EpochMetrics:
    return EpochMetrics(
        roc_auc_macro=0.88,
        pr_auc_macro=0.75,
        pr_norm_excess_macro=0.30,
        f1_macro_best=0.80,
        f1_macro_05=0.78,
        f1_micro_05=0.79,
        f1_weighted_05=0.79,
        exact_match_ratio=0.60,
        hamming_loss=0.15,
        val_loss=0.42,
        roc_auc={"c1": 0.88},
        pr_auc={"c1": 0.75},
        pr_norm_excess={"c1": 0.30},
        best_thresh={"c1": 0.5},
        best_f1={"c1": 0.80},
        best_prec={"c1": 0.82},
        best_recall={"c1": 0.78},
        f1_05={"c1": 0.78},
        precision_05={"c1": 0.80},
        recall_05={"c1": 0.76},
        support_pos={"c1": 50},
        support_neg={"c1": 50},
    )


def test_save_and_load_full_checkpoint(tmp_path):
    model = create_dummy_model()
    optimizer = build_pipeline_optimizer(model, OptimizerParams())
    metrics = create_dummy_metrics()

    ckpt = Checkpoint(model=model, optimizer=optimizer, eval_metrics=metrics)

    ckpt_dir = str(tmp_path)
    ckpt.save(ckpt_dir, epoch=2)

    # Reload checkpoint
    reloaded_ckpt = Checkpoint.from_file(
        checkpoint_path=ckpt_dir,
        model=model,
        optimizer=optimizer,
    )

    assert reloaded_ckpt.model is not None
    assert reloaded_ckpt.optimizer is not None
    assert isinstance(reloaded_ckpt.eval_metrics, EpochMetrics)
    assert reloaded_ckpt.eval_metrics.val_loss == 0.42
    assert reloaded_ckpt.eval_metrics.roc_auc_macro == 0.88


def test_save_and_load_checkpoint_no_optimizer(tmp_path):
    model = create_dummy_model()
    metrics = create_dummy_metrics()

    ckpt = Checkpoint(model=model, optimizer=None, eval_metrics=metrics)

    ckpt_dir = str(tmp_path)
    ckpt.save(ckpt_dir, epoch=2)

    # Reload checkpoint without passing optimizer
    reloaded_ckpt = Checkpoint.from_file(
        checkpoint_path=ckpt_dir,
        model=model,
    )

    assert reloaded_ckpt.model is not None
    assert reloaded_ckpt.optimizer is None
    assert isinstance(reloaded_ckpt.eval_metrics, EpochMetrics)
    assert reloaded_ckpt.eval_metrics.val_loss == 0.42


def test_save_and_load_checkpoint_no_metrics(tmp_path):
    model = create_dummy_model()
    optimizer = build_pipeline_optimizer(model, OptimizerParams())

    ckpt = Checkpoint(model=model, optimizer=optimizer, eval_metrics=None)

    ckpt_dir = str(tmp_path)
    ckpt.save(ckpt_dir, epoch=2)

    # Reload checkpoint
    reloaded_ckpt = Checkpoint.from_file(
        checkpoint_path=ckpt_dir,
        model=model,
        optimizer=optimizer,
    )

    assert reloaded_ckpt.model is not None
    assert reloaded_ckpt.optimizer is not None
    assert reloaded_ckpt.eval_metrics is None


def test_save_and_load_dict_metrics(tmp_path):
    model = create_dummy_model()
    dict_metrics = {"val_loss": 0.42, "roc_auc_macro": 0.88}

    ckpt = Checkpoint(model=model, optimizer=None, eval_metrics=dict_metrics)

    ckpt_dir = str(tmp_path)
    ckpt.save(ckpt_dir, epoch=2)

    reloaded_ckpt = Checkpoint.from_file(
        checkpoint_path=ckpt_dir,
        model=model,
    )

    assert reloaded_ckpt.eval_metrics == dict_metrics


def test_checkpoint_from_file_instantiates_model(tmp_path):
    model = create_dummy_model()
    ckpt = Checkpoint(model=model, optimizer=None, eval_metrics=None)

    ckpt_dir = str(tmp_path)
    ckpt.save(ckpt_dir, epoch=1)

    # Reload checkpoint without passing pre-instantiated model
    reloaded_ckpt = Checkpoint.from_file(
        checkpoint_path=ckpt_dir,
    )

    assert reloaded_ckpt.model is not None
    assert hasattr(reloaded_ckpt.model, "backbone")


def test_checkpoint_to_dict(tmp_path):
    model = create_dummy_model()
    optimizer = build_pipeline_optimizer(model, OptimizerParams())
    metrics = create_dummy_metrics()

    ckpt = Checkpoint(model=model, optimizer=optimizer, eval_metrics=metrics)
    d = ckpt.to_dict()

    assert "model_state_dict" in d
    assert "model_params" in d
    assert "optimizer_state_dict" in d
    assert "eval_metrics" in d

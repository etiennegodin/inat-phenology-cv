import math

import pytest

from plant_pheno.train.profiling import EpochProfiling, MetricsProfiling, StepProfiling


@pytest.fixture
def step_profile():
    return StepProfiling(data_time=0.1, compute_time=0.2)


@pytest.fixture
def metrics_profile():
    return MetricsProfiling(
        compute_metrics_time=0.1,
        log_metrics_time=0.2,
        log_attention_metrics_time=0.3,
        error_analysis_time=0.4,
    )


def test_step_profile(step_profile):
    assert math.isclose(step_profile.total(), 0.3)
    assert step_profile.data_time == 0.1
    assert step_profile.compute_time == 0.2


def test_metrics_profile(metrics_profile):
    assert math.isclose(metrics_profile.total(), 1.0)


def test_epoch_metrics(step_profile, metrics_profile):
    e = EpochProfiling(
        train=step_profile,
        epoch_scheduler_time=0.1,
        eval=step_profile,
        metrics=metrics_profile,
    )
    assert math.isclose(e.epoch_duration, 1.7)

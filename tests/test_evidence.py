import json
import numpy as np
import pytest
from pydantic import ValidationError

from modelbreaker.evidence import Evidence, compute_metrics


def make(**overrides):
    base = dict(
        attack_name="gaussian_noise",
        params={"sigma": 0.5},
        seed=0,
        n_samples=100,
        metrics_before={"accuracy": 0.95, "balanced_accuracy": 0.94, "f1_macro": 0.94},
        metrics_after={"accuracy": 0.80, "balanced_accuracy": 0.78, "f1_macro": 0.77},
    )
    base.update(overrides)
    return Evidence(**base)


def test_drop_is_before_minus_after():
    assert make().primary_drop == pytest.approx(0.16)


def test_json_roundtrip():
    e = make()
    data = json.loads(e.model_dump_json())
    assert "drops" in data
    assert data["attack_name"] == "gaussian_noise"


def test_is_immutable():
    with pytest.raises(ValidationError):
        make().seed = 5


def test_mismatched_metric_keys_rejected():
    with pytest.raises(ValidationError):
        make(metrics_after={"accuracy": 0.8, "balanced_accuracy": 0.78})


def test_missing_primary_metric_rejected():
    m = {"accuracy": 0.9, "f1_macro": 0.9}
    with pytest.raises(ValidationError):
        make(metrics_before=m, metrics_after=m)


def test_nan_metric_rejected():
    bad = {"accuracy": 0.8, "balanced_accuracy": float("nan"), "f1_macro": 0.7}
    with pytest.raises(ValidationError):
        make(metrics_after=bad)


def test_numpy_param_rejected():
    with pytest.raises(ValidationError):
        make(params={"sigma": np.float32(0.5)})


def test_zero_samples_rejected():
    with pytest.raises(ValidationError):
        make(n_samples=0)


def test_compute_metrics_perfect_and_keys():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert set(m) == {"accuracy", "balanced_accuracy", "f1_macro"}
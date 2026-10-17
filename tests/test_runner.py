import json

import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelbreaker.runner import run_gaussian_noise


@pytest.fixture(scope="module")
def setup():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    return model, X, y


def test_zero_sigma_gives_zero_drop(setup):
    model, X, y = setup
    e = run_gaussian_noise(model, X, y, sigma=0.0, seed=0)
    assert e.primary_drop == 0.0
    assert e.extra["flip_rate"] == 0.0


def test_large_sigma_hurts(setup):
    model, X, y = setup
    e = run_gaussian_noise(model, X, y, sigma=2.0, seed=0)
    assert e.primary_drop > 0.05
    assert e.extra["flip_rate"] > 0.0


def test_same_seed_same_evidence(setup):
    model, X, y = setup
    a = run_gaussian_noise(model, X, y, sigma=1.0, seed=3)
    b = run_gaussian_noise(model, X, y, sigma=1.0, seed=3)
    assert a == b


def test_evidence_is_json_serializable(setup):
    model, X, y = setup
    e = run_gaussian_noise(model, X, y, sigma=1.0, seed=0, features=["mean radius"])
    data = json.loads(e.model_dump_json())
    assert data["params"]["features"] == ["mean radius"]


def test_length_mismatch_raises(setup):
    model, X, y = setup
    with pytest.raises(ValueError):
        run_gaussian_noise(model, X, y.iloc[:10], sigma=1.0, seed=0)
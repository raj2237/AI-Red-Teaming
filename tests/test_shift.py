import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelbreaker.attacks import feature_shift_attack
from modelbreaker.runner import run_feature_shift


@pytest.fixture(scope="module")
def setup():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    return model, X, y


def test_shift_amount_is_k_times_std(setup):
    _, X, _ = setup
    out = feature_shift_attack(X, features=["mean radius"], k=2.0)
    expected = 2.0 * X["mean radius"].std(ddof=0)
    assert np.allclose(out["mean radius"] - X["mean radius"], expected)


def test_negative_k_shifts_down(setup):
    _, X, _ = setup
    out = feature_shift_attack(X, features=["mean radius"], k=-1.0)
    assert (out["mean radius"] < X["mean radius"]).all()


def test_k_zero_is_identity(setup):
    _, X, _ = setup
    out = feature_shift_attack(X, features=["mean radius"], k=0.0)
    pd.testing.assert_frame_equal(out, X.astype(float))


def test_other_columns_untouched(setup):
    _, X, _ = setup
    out = feature_shift_attack(X, features=["mean radius"], k=3.0)
    pd.testing.assert_frame_equal(out.drop(columns="mean radius"),
                                  X.drop(columns="mean radius").astype(float))


def test_input_not_mutated(setup):
    _, X, _ = setup
    before = X.copy()
    feature_shift_attack(X, features=["mean radius"], k=3.0)
    pd.testing.assert_frame_equal(X, before)


def test_constant_column_unchanged():
    X = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    out = feature_shift_attack(X, features=["a"], k=5.0)
    assert (out["a"] == 1.0).all()


def test_returns_same_type(setup):
    _, X, _ = setup
    assert isinstance(feature_shift_attack(X, features=[0], k=1.0), pd.DataFrame)
    assert isinstance(feature_shift_attack(X.to_numpy(), features=[0], k=1.0), np.ndarray)


def test_invalid_inputs_raise(setup):
    _, X, _ = setup
    with pytest.raises(ValueError):
        feature_shift_attack(X, features=[], k=1.0)
    with pytest.raises(ValueError):
        feature_shift_attack(X, features=["nope"], k=1.0)
    with pytest.raises(ValueError):
        feature_shift_attack(X, features=[0], k=float("nan"))


def test_runner_zero_shift_zero_drop(setup):
    model, X, y = setup
    e = run_feature_shift(model, X, y, features=["mean radius"], k=0.0)
    assert e.primary_drop == 0.0


def test_runner_big_shift_hurts_most_important_feature(setup):
    model, X, y = setup
    e = run_feature_shift(model, X, y, features=list(X.columns), k=3.0)
    assert e.primary_drop > 0.1
    assert e.extra["deterministic"] is True
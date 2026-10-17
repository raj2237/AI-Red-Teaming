import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modelbreaker.attacks import feature_ablation_attack
from modelbreaker.runner import run_feature_ablation, sweep_single_feature_ablation


@pytest.fixture(scope="module")
def setup():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    return model, X, y


def test_only_selected_columns_change(setup):
    _, X, _ = setup
    out = feature_ablation_attack(X, features=["mean radius"], mode="zero")
    assert (out["mean radius"] == 0.0).all()
    pd.testing.assert_frame_equal(out.drop(columns="mean radius"), X.drop(columns="mean radius"))


def test_mean_mode_sets_column_mean(setup):
    _, X, _ = setup
    out = feature_ablation_attack(X, features=["mean radius"], mode="mean")
    assert np.allclose(out["mean radius"], X["mean radius"].mean())


def test_input_not_mutated(setup):
    _, X, _ = setup
    before = X.copy()
    feature_ablation_attack(X, features=["mean radius"], mode="zero")
    pd.testing.assert_frame_equal(X, before)


def test_returns_same_type(setup):
    _, X, _ = setup
    assert isinstance(feature_ablation_attack(X, features=[0]), pd.DataFrame)
    assert isinstance(feature_ablation_attack(X.to_numpy(), features=[0]), np.ndarray)


def test_invalid_inputs_raise(setup):
    _, X, _ = setup
    with pytest.raises(ValueError):
        feature_ablation_attack(X, features=[])
    with pytest.raises(ValueError):
        feature_ablation_attack(X, features=["nope"])
    with pytest.raises(ValueError):
        feature_ablation_attack(X, features=[0], mode="median")


def test_ablating_everything_destroys_the_model(setup):
    model, X, y = setup
    e = run_feature_ablation(model, X, y, features=list(X.columns), mode="mean")
    assert e.primary_drop > 0.3          # constant input -> constant prediction


def test_sweep_is_sorted_and_complete(setup):
    model, X, y = setup
    results = sweep_single_feature_ablation(model, X, y)
    assert len(results) == X.shape[1]
    drops = [r.primary_drop for r in results]
    assert drops == sorted(drops, reverse=True)
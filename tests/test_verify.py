import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from modelbreaker.suite import ExperimentSpec, build_fixed_suite, execute, run_fixed_suite
from modelbreaker.verify import verify_all, verify_finding


@pytest.fixture(scope="module")
def stump_setup():
    """Planted weakness: a depth-1 tree depends on exactly one feature."""
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    return model, X, y, X.columns[model.tree_.feature[0]]


@pytest.fixture(scope="module")
def logreg_setup():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    return model, X, y


def _ablate(model, X, y, feature):
    return execute(model, X, y, ExperimentSpec(
        attack="feature_ablation", params={"features": [feature]}))


def test_real_deterministic_finding_is_verified_by_bootstrap(stump_setup):
    model, X, y, used = stump_setup
    r = verify_finding(model, X, y, _ablate(model, X, y, used), n_runs=30)
    assert r.method == "bootstrap"
    assert r.verified
    assert r.agree_fraction == 1.0


def test_borderline_finding_is_rejected(stump_setup):
    """Threshold set exactly at the observed drop: resamples land on both sides."""
    model, X, y, used = stump_setup
    e = _ablate(model, X, y, used)
    r = verify_finding(model, X, y, e, threshold=e.primary_drop, n_runs=30)
    assert r.agree_fraction < 0.9
    assert not r.verified


def test_stochastic_finding_uses_reseed(logreg_setup):
    model, X, y = logreg_setup
    e = execute(model, X, y, ExperimentSpec(
        attack="gaussian_noise", params={"sigma": 4.0}, seed=0))
    r = verify_finding(model, X, y, e, n_runs=10)
    assert r.method == "reseed"
    assert len(r.drops) == 10
    assert r.verified, r.drops   # if this fails, read the drops, don't weaken the test


def test_non_candidate_cannot_be_verified(logreg_setup):
    model, X, y = logreg_setup
    e = execute(model, X, y, ExperimentSpec(
        attack="gaussian_noise", params={"sigma": 0.01}, seed=0))
    with pytest.raises(ValueError):
        verify_finding(model, X, y, e)


def test_verification_is_reproducible(stump_setup):
    model, X, y, used = stump_setup
    e = _ablate(model, X, y, used)
    assert verify_finding(model, X, y, e, seed=1) == verify_finding(model, X, y, e, seed=1)


def test_inputs_not_mutated(stump_setup):
    model, X, y, used = stump_setup
    X_before, y_before = X.copy(), y.copy()
    verify_finding(model, X, y, _ablate(model, X, y, used))
    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)


def test_invalid_arguments_raise(stump_setup):
    model, X, y, used = stump_setup
    e = _ablate(model, X, y, used)
    with pytest.raises(ValueError):
        verify_finding(model, X, y, e, n_runs=1)
    with pytest.raises(ValueError):
        verify_finding(model, X, y, e, min_agree=0.0)


def test_verify_all_covers_every_finding(stump_setup):
    model, X, y, used = stump_setup
    result = run_fixed_suite(model, X, y, build_fixed_suite(list(X.columns)))
    verified = verify_all(model, X, y, result, n_runs=10)
    assert len(verified) == len(result.findings)
    ablation_of_used = [v for v in verified
                        if v.evidence.attack_name == "feature_ablation"
                        and v.evidence.params["features"] == [used]]
    assert len(ablation_of_used) == 1 and ablation_of_used[0].verified
import pytest
from pydantic import ValidationError
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

from modelbreaker.suite import (
    ExperimentSpec, build_fixed_suite, execute, is_finding, run_fixed_suite,
)


@pytest.fixture(scope="module")
def stump_setup():
    """Planted weakness: a depth-1 tree uses exactly ONE feature."""
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    model = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)
    used = X.columns[model.tree_.feature[0]]
    return model, X, y, used


def test_spec_rejects_unknown_attack():
    with pytest.raises(ValidationError):
        ExperimentSpec(attack="rm_rf", params={})


def test_spec_rejects_missing_and_unknown_params():
    with pytest.raises(ValidationError):
        ExperimentSpec(attack="gaussian_noise", params={})
    with pytest.raises(ValidationError):
        ExperimentSpec(attack="gaussian_noise", params={"sigma": 1.0, "evil": 1})


def test_suite_size_and_determinism(stump_setup):
    _, X, _, _ = stump_setup
    cols = list(X.columns)
    suite = build_fixed_suite(cols)
    assert len(suite) == 4 + len(cols) + 2 * len(cols)
    assert suite == build_fixed_suite(cols)


def test_suite_run_is_reproducible(stump_setup):
    model, X, y, _ = stump_setup
    specs = build_fixed_suite(list(X.columns))[:10]
    a = run_fixed_suite(model, X, y, specs)
    b = run_fixed_suite(model, X, y, specs)
    assert a == b
    assert a.n_experiments == 10


def test_threshold_boundary(stump_setup):
    model, X, y, used = stump_setup
    e = execute(model, X, y, ExperimentSpec(
        attack="feature_ablation", params={"features": [used]}))
    assert is_finding(e, threshold=e.primary_drop)           # >= is inclusive
    assert not is_finding(e, threshold=e.primary_drop + 1e-9)


def test_cumulative_curve_is_monotone(stump_setup):
    model, X, y, _ = stump_setup
    result = run_fixed_suite(model, X, y, build_fixed_suite(list(X.columns)))
    curve = result.cumulative_findings()
    assert len(curve) == result.n_experiments
    assert curve == sorted(curve)
    assert curve[-1] == len(result.findings)


def test_ground_truth_ablation_finds_exactly_the_used_feature(stump_setup):
    """The stump depends on one feature, so only ablating that one can hurt."""
    model, X, y, used = stump_setup
    specs = [s for s in build_fixed_suite(list(X.columns))
             if s.attack == "feature_ablation"]
    result = run_fixed_suite(model, X, y, specs)
    found = {e.params["features"][0] for e in result.findings}
    assert found == {used}
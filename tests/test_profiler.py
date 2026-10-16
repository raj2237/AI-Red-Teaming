
import json

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from modelbreaker.profiler import profile_model, ProfilingError


@pytest.fixture
def dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


@pytest.fixture
def fitted_model(dataset):
    X_train, X_test, y_train, y_test = dataset

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def test_data(dataset):
    _, X_test, _, y_test = dataset
    return X_test, y_test


def test_profile_with_numpy_input(fitted_model, test_data):
    X_test, y_test = test_data

    profile = profile_model(
        fitted_model,
        X_test.to_numpy(),
        y_test.to_numpy(),
    )

    assert profile["model_type"] == "LogisticRegression"
    assert profile["number_of_samples"] == len(X_test)
    assert profile["number_of_features"] == X_test.shape[1]


def test_profile_with_dataframe_input(fitted_model, test_data):
    X_test, y_test = test_data

    profile = profile_model(fitted_model, X_test, y_test)

    assert profile["model_type"] == "LogisticRegression"
    assert profile["number_of_samples"] == len(X_test)
    assert profile["number_of_features"] == X_test.shape[1]


def test_baseline_accuracy_matches_sklearn(fitted_model, test_data):
    X_test, y_test = test_data

    profile = profile_model(fitted_model, X_test, y_test)

    expected_accuracy = accuracy_score(
        y_test,
        fitted_model.predict(X_test),
    )

    assert profile["baseline_accuracy"] == pytest.approx(expected_accuracy)


def test_profile_is_json_serializable(fitted_model, test_data):
    X_test, y_test = test_data

    profile = profile_model(fitted_model, X_test, y_test)

    serialized = json.dumps(profile)

    assert isinstance(serialized, str)
    assert json.loads(serialized)["model_type"] == "LogisticRegression"


def test_unfitted_model_raises_profiling_error(test_data):
    X_test, y_test = test_data
    model = LogisticRegression(max_iter=5000)

    with pytest.raises(ProfilingError):
        profile_model(model, X_test, y_test)


def test_wrong_number_of_features_raises(fitted_model, test_data):
    X_test, y_test = test_data

    wrong_X = X_test.iloc[:, :-1]

    with pytest.raises(ProfilingError) as exc_info:
        profile_model(fitted_model, wrong_X, y_test)

    message = str(exc_info.value)
    assert str(X_test.shape[1]) in message
    assert str(wrong_X.shape[1]) in message


def test_mismatched_sample_counts_raise(fitted_model, test_data):
    X_test, y_test = test_data

    with pytest.raises(ProfilingError):
        profile_model(fitted_model, X_test, y_test.iloc[:-1])


def test_regressor_raises(test_data):
    X_test, y_test = test_data

    regressor = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )
    regressor.fit(X_test, y_test)

    with pytest.raises(ProfilingError):
        profile_model(regressor, X_test, y_test)


def test_string_column_dataframe_raises(fitted_model, test_data):
    X_test, y_test = test_data

    bad_X = X_test.copy()
    bad_X["malicious_string_feature"] = "not-a-number"

    with pytest.raises(ProfilingError):
        profile_model(fitted_model, bad_X, y_test)

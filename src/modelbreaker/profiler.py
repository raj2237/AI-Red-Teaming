import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted


class ProfilingError(ValueError):
    """Raised when a model/dataset pair cannot uube profiled."""


def profile_model(model, X, y) -> dict:
    """Inspect a fitted classifier and return a JSON-serializable profile."""
    try:
        check_is_fitted(model)
    except (NotFittedError, TypeError) as e:
        raise ProfilingError(f"Model is not fitted: {e}") from e

    if not hasattr(model, "classes_"):
        raise ProfilingError("Only fitted classifiers are supported (no classes_ attribute).")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.ndim != 2:
        raise ProfilingError(f"X must be 2D, got {X_arr.ndim}D.")
    if y_arr.ndim != 1:
        raise ProfilingError(f"y must be 1D, got {y_arr.ndim}D.")
    if X_arr.shape[0] == 0:
        raise ProfilingError("X and y cannot be empty.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ProfilingError(
            f"X has {X_arr.shape[0]} samples but y has {y_arr.shape[0]}."
        )
    if not (np.issubdtype(X_arr.dtype, np.number) or X_arr.dtype == bool):
        raise ProfilingError(f"X must be numeric, got dtype {X_arr.dtype}.")

    n_samples, n_features = X_arr.shape
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != n_features:
        raise ProfilingError(
            f"Model expects {expected} features, X has {n_features}."
        )

    if isinstance(X, pd.DataFrame):
        feature_names = [str(c) for c in X.columns]
        feature_types = [str(d) for d in X.dtypes]
    else:
        feature_names = [f"feature_{i}" for i in range(n_features)]
        feature_types = [str(X_arr.dtype)] * n_features

    try:
        preds = np.asarray(model.predict(X))
    except Exception as e:
        raise ProfilingError(f"model.predict failed: {e}") from e

    def _counts(arr):
        labels, counts = np.unique(arr, return_counts=True)
        return {str(l): int(c) for l, c in zip(labels, counts)}

    return {
        "model_type": type(model).__name__,
        "number_of_samples": int(n_samples),
        "number_of_features": int(n_features),
        "feature_names": feature_names,
        "feature_types": feature_types,
        "class_distribution": _counts(y_arr),
        "predicted_distribution": _counts(preds),
        "baseline_accuracy": float(accuracy_score(y_arr, preds)),
        "prediction_classes": [str(c) for c in model.classes_],
    }
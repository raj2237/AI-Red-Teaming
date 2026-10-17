import numpy as np
import pandas as pd


def _to_float_matrix(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        if not all(pd.api.types.is_numeric_dtype(d) for d in X.dtypes):
            raise ValueError("All features must be numeric")
        values = X.to_numpy(dtype=float)
    else:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"X must be numeric: {e}") from e
    if values.ndim != 2:
        raise ValueError(f"X must be 2D, got {values.ndim}D")
    if values.shape[0] == 0:
        raise ValueError("X cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError("X must contain only finite values")
    return values


def _feature_mask(X, features, n_features: int) -> np.ndarray:
    if features is None:
        return np.ones(n_features, dtype=bool)
    mask = np.zeros(n_features, dtype=bool)
    for f in features:
        if isinstance(X, pd.DataFrame) and f in X.columns:
            idx = X.columns.get_loc(f)
        elif isinstance(f, (int, np.integer)) and 0 <= f < n_features:
            idx = int(f)
        else:
            raise ValueError(f"Unknown feature: {f!r}")
        mask[idx] = True
    return mask


def gaussian_noise_attack(X, *, sigma: float, seed: int, features=None):
    """Add Gaussian noise scaled by each feature's std. Does not modify X.

    sigma: noise std as a multiple of each feature's std (0 = no change).
    features: optional column names (DataFrame) or indices to perturb.
    Returns the same type it received (DataFrame or ndarray).
    """
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative")

    values = _to_float_matrix(X)
    mask = _feature_mask(X, features, values.shape[1])

    rng = np.random.default_rng(seed)
    # Always draw the full matrix so the same seed gives the same noise
    # on a given column regardless of which features are selected.
    noise = rng.standard_normal(values.shape) * (sigma * values.std(axis=0))
    noise[:, ~mask] = 0.0

    attacked = values + noise
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(attacked, columns=X.columns, index=X.index)
    return attacked

def feature_ablation_attack(X, *, features, mode: str = "mean"):
    """Replace the selected features with a constant. Does not modify X.

    mode="mean": each selected column is set to its mean in X.
    mode="zero": each selected column is set to 0.0.
    Deterministic: there is no randomness, so no seed.
    Returns the same type it received (DataFrame or ndarray).
    """
    if mode not in ("mean", "zero"):
        raise ValueError(f"mode must be 'mean' or 'zero', got {mode!r}")
    if features is None or len(features) == 0:
        raise ValueError("features must be a non-empty selection")

    values = _to_float_matrix(X)
    mask = _feature_mask(X, features, values.shape[1])

    attacked = values.copy()
    if mode == "mean":
        attacked[:, mask] = values.mean(axis=0)[mask]
    else:
        attacked[:, mask] = 0.0

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(attacked, columns=X.columns, index=X.index)
    return attacked


def feature_shift_attack(X, *, features, k: float):
    """Shift the selected features by k standard deviations. Does not modify X.

    k may be negative (shift downward). k=0 returns the values unchanged.
    A zero-variance feature has std 0, so it is left unchanged.
    Deterministic: there is no randomness, so no seed.
    Returns the same type it received (DataFrame or ndarray).
    """
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    if features is None or len(features) == 0:
        raise ValueError("features must be a non-empty selection")

    values = _to_float_matrix(X)
    mask = _feature_mask(X, features, values.shape[1])

    attacked = values.copy()
    attacked[:, mask] += k * values.std(axis=0)[mask]

    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(attacked, columns=X.columns, index=X.index)
    return attacked
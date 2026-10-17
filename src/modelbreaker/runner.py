"""Runner: executes one attack against a model and returns Evidence."""
from __future__ import annotations

import numpy as np

from modelbreaker.attacks import feature_ablation_attack, gaussian_noise_attack,feature_shift_attack
from modelbreaker.evidence import Evidence, compute_metrics

def _plain_features(features) -> list | None:
    """Convert feature selectors to JSON-safe Python types."""
    if features is None:
        return None
    return [int(f) if isinstance(f, (int, np.integer)) else str(f) for f in features]


def _build_evidence(model, X, y_arr, X_attacked, *, attack_name, params, seed, extra=None):
    preds_before = np.asarray(model.predict(X))
    preds_after = np.asarray(model.predict(X_attacked))
    return Evidence(
        attack_name=attack_name,
        params=params,
        seed=int(seed),
        n_samples=int(len(y_arr)),
        metrics_before=compute_metrics(y_arr, preds_before),
        metrics_after=compute_metrics(y_arr, preds_after),
        extra={"flip_rate": float(np.mean(preds_before != preds_after)), **(extra or {})},
    )


def _check_lengths(X, y) -> np.ndarray:
    y_arr = np.asarray(y)
    if len(X) != len(y_arr):
        raise ValueError(f"X has {len(X)} samples but y has {len(y_arr)}.")
    return y_arr


def run_gaussian_noise(model, X, y, *, sigma: float, seed: int, features=None) -> Evidence:
    y_arr = _check_lengths(X, y)
    X_attacked = gaussian_noise_attack(X, sigma=sigma, seed=seed, features=features)
    return _build_evidence(
        model, X, y_arr, X_attacked,
        attack_name="gaussian_noise",
        params={"sigma": float(sigma), "features": _plain_features(features)},
        seed=seed,
    )


def run_feature_ablation(model, X, y, *, features, mode: str = "mean") -> Evidence:
    y_arr = _check_lengths(X, y)
    X_attacked = feature_ablation_attack(X, features=features, mode=mode)
    return _build_evidence(
        model, X, y_arr, X_attacked,
        attack_name="feature_ablation",
        params={"mode": mode, "features": _plain_features(features)},
        seed=0,  # deterministic attack; seed recorded only to satisfy the schema
        extra={"deterministic": True},
    )


def sweep_single_feature_ablation(model, X, y, *, mode: str = "mean") -> list[Evidence]:
    """Ablate each feature on its own. Returns Evidence sorted by worst drop first."""
    names = list(X.columns) if hasattr(X, "columns") else list(range(np.asarray(X).shape[1]))
    results = [run_feature_ablation(model, X, y, features=[n], mode=mode) for n in names]
    return sorted(results, key=lambda e: e.primary_drop, reverse=True)


def run_feature_shift(model, X, y, *, features, k: float) -> Evidence:
    y_arr = _check_lengths(X, y)
    X_attacked = feature_shift_attack(X, features=features, k=k)
    return _build_evidence(
        model, X, y_arr, X_attacked,
        attack_name="feature_shift",
        params={"k": float(k), "features": _plain_features(features)},
        seed=0,
        extra={"deterministic": True},
    )
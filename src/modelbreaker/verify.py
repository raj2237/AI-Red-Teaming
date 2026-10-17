"""Verification: re-test candidate findings before calling them verified."""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from modelbreaker.evidence import Evidence
from modelbreaker.suite import (
    FINDING_THRESHOLD,
    ExperimentSpec,
    SuiteResult,
    execute,
    is_finding,
)

# Attacks with randomness get re-seeded; everything else gets bootstrapped.
# When you add a stochastic attack, add it here.
STOCHASTIC_ATTACKS = {"gaussian_noise"}


def spec_from_evidence(e: Evidence) -> ExperimentSpec:
    """Rebuild the experiment that produced a piece of Evidence."""
    return ExperimentSpec(attack=e.attack_name, params=dict(e.params), seed=e.seed)


def _stratified_bootstrap_indices(y_arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample with replacement inside each class, so no resample loses a class
    and class proportions stay fixed."""
    parts = []
    for label in np.unique(y_arr):
        idx = np.flatnonzero(y_arr == label)
        parts.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(parts)


def _take(X, idx: np.ndarray):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].reset_index(drop=True)
    return np.asarray(X)[idx]


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence: Evidence
    method: Literal["reseed", "bootstrap"]
    n_runs: int
    drops: list[float]
    threshold: float
    min_agree: float
    mean_drop: float
    agree_fraction: float
    verified: bool


def verify_finding(
    model,
    X,
    y,
    evidence: Evidence,
    *,
    threshold: float = FINDING_THRESHOLD,
    n_runs: int = 10,
    min_agree: float = 0.9,
    seed: int = 0,
) -> VerificationResult:
    """Verified iff, over n_runs independent re-tests:
    (1) at least min_agree of runs individually show drop >= threshold, AND
    (2) the mean drop is >= threshold.
    The original run is excluded, so the re-tests are independent of it."""
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2")
    if not 0 < min_agree <= 1:
        raise ValueError("min_agree must be in (0, 1]")
    if not is_finding(evidence, threshold):
        raise ValueError("Only candidate findings can be verified "
                         f"(drop {evidence.primary_drop:.4f} < threshold {threshold}).")

    y_arr = np.asarray(y)
    spec = spec_from_evidence(evidence)
    drops: list[float] = []

    if spec.attack in STOCHASTIC_ATTACKS:
        method = "reseed"
        for i in range(n_runs):
            s = spec.model_copy(update={"seed": evidence.seed + 1 + i})
            drops.append(execute(model, X, y_arr, s).primary_drop)
    else:
        method = "bootstrap"
        rng = np.random.default_rng(seed)
        for _ in range(n_runs):
            idx = _stratified_bootstrap_indices(y_arr, rng)
            drops.append(execute(model, _take(X, idx), y_arr[idx], spec).primary_drop)

    mean_drop = float(np.mean(drops))
    agree = float(np.mean([d >= threshold for d in drops]))
    return VerificationResult(
        evidence=evidence,
        method=method,
        n_runs=n_runs,
        drops=[float(d) for d in drops],
        threshold=threshold,
        min_agree=min_agree,
        mean_drop=mean_drop,
        agree_fraction=agree,
        verified=bool(agree >= min_agree and mean_drop >= threshold),
    )


def verify_all(model, X, y, result: SuiteResult, **kwargs) -> list[VerificationResult]:
    """Verify every candidate finding in a suite result."""
    return [
        verify_finding(model, X, y, e, threshold=result.threshold, **kwargs)
        for e in result.findings
    ]
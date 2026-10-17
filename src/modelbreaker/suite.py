"""Experiment specs, the definition of a finding, and the fixed-suite baseline."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from modelbreaker.evidence import Evidence
from modelbreaker.runner import (
    run_feature_ablation,
    run_feature_shift,
    run_gaussian_noise,
)

# ---- Catalogue: which attacks exist and which params each requires ----------
# Phase D validates the agent's choices against this same table.
REQUIRED_PARAMS: dict[str, set[str]] = {
    "gaussian_noise": {"sigma"},          # optional: features
    "feature_ablation": {"features"},     # optional: mode
    "feature_shift": {"features", "k"},
}
OPTIONAL_PARAMS: dict[str, set[str]] = {
    "gaussian_noise": {"features"},
    "feature_ablation": {"mode"},
    "feature_shift": set(),
}

AttackName = Literal["gaussian_noise", "feature_ablation", "feature_shift"]


class ExperimentSpec(BaseModel):
    """One planned experiment. Immutable and validated against the catalogue."""

    model_config = ConfigDict(frozen=True)

    attack: AttackName
    params: dict[str, Any]
    seed: int = 0

    @model_validator(mode="after")
    def _check_params(self) -> "ExperimentSpec":
        given = set(self.params)
        required = REQUIRED_PARAMS[self.attack]
        allowed = required | OPTIONAL_PARAMS[self.attack]
        if not required <= given:
            raise ValueError(f"{self.attack} missing params: {sorted(required - given)}")
        if not given <= allowed:
            raise ValueError(f"{self.attack} unknown params: {sorted(given - allowed)}")
        return self


def execute(model, X, y, spec: ExperimentSpec) -> Evidence:
    p = spec.params
    if spec.attack == "gaussian_noise":
        return run_gaussian_noise(model, X, y, sigma=p["sigma"], seed=spec.seed,
                                  features=p.get("features"))
    if spec.attack == "feature_ablation":
        return run_feature_ablation(model, X, y, features=p["features"],
                                    mode=p.get("mode", "mean"))
    return run_feature_shift(model, X, y, features=p["features"], k=p["k"])


# ---- Definition of a finding -------------------------------------------------
FINDING_THRESHOLD = 0.05  # primary-metric (balanced accuracy) drop


def is_finding(evidence: Evidence, threshold: float = FINDING_THRESHOLD) -> bool:
    """A CANDIDATE finding: primary drop >= threshold. Not verified until Phase C."""
    return evidence.primary_drop >= threshold


# ---- Fixed suite ---------------------------------------------------------------
def build_fixed_suite(
    feature_names: list,
    sigmas: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0),
    ks: tuple[float, ...] = (2.0, -2.0),
) -> list[ExperimentSpec]:
    """Every attack at preset strengths. The order is fixed and deterministic."""
    specs: list[ExperimentSpec] = []
    for s in sigmas:
        specs.append(ExperimentSpec(attack="gaussian_noise", params={"sigma": s}, seed=0))
    for f in feature_names:
        specs.append(ExperimentSpec(attack="feature_ablation",
                                    params={"features": [f], "mode": "mean"}))
    for f in feature_names:
        for k in ks:
            specs.append(ExperimentSpec(attack="feature_shift",
                                        params={"features": [f], "k": k}))
    return specs


class SuiteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float
    evidence: list[Evidence]

    @property
    def n_experiments(self) -> int:
        return len(self.evidence)

    @property
    def findings(self) -> list[Evidence]:
        return [e for e in self.evidence if is_finding(e, self.threshold)]

    def cumulative_findings(self) -> list[int]:
        """Findings found after 1, 2, ..., N experiments. This is the efficiency
        curve the agent will be compared against."""
        out, count = [], 0
        for e in self.evidence:
            count += is_finding(e, self.threshold)
            out.append(count)
        return out


def run_fixed_suite(model, X, y, specs: list[ExperimentSpec],
                    threshold: float = FINDING_THRESHOLD) -> SuiteResult:
    return SuiteResult(threshold=threshold,
                       evidence=[execute(model, X, y, s) for s in specs])
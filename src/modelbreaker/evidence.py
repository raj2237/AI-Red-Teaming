"""Evidence: the single structured result type every attack returns."""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PRIMARY_METRIC = "balanced_accuracy"


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Metrics recorded before and after every attack.

    balanced_accuracy is primary because plain accuracy hides failures
    on imbalanced data.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


class Evidence(BaseModel):
    """Immutable record of one attack run. Must be JSON-serializable."""

    model_config = ConfigDict(frozen=True)

    attack_name: str = Field(min_length=1)
    params: dict[str, Any]
    seed: int
    n_samples: int = Field(gt=0)
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    extra: dict[str, Any] = Field(default_factory=dict)  # e.g. {"flip_rate": 0.12}

    @model_validator(mode="after")
    def _validate(self) -> "Evidence":
        if set(self.metrics_before) != set(self.metrics_after):
            raise ValueError("metrics_before and metrics_after must have the same keys.")
        if PRIMARY_METRIC not in self.metrics_before:
            raise ValueError(f"Missing primary metric '{PRIMARY_METRIC}'.")
        for name, v in {**self.metrics_before, **self.metrics_after}.items():
            if not math.isfinite(v):
                raise ValueError(f"Metric '{name}' is not finite: {v}")
        try:
            json.dumps({"params": self.params, "extra": self.extra})
        except TypeError as e:
            raise ValueError(
                f"params/extra must be JSON-serializable (cast numpy types): {e}"
            ) from e
        return self

    @computed_field
    @property
    def drops(self) -> dict[str, float]:
        """metric_before - metric_after for every metric. Positive = got worse."""
        return {k: self.metrics_before[k] - self.metrics_after[k] for k in self.metrics_before}

    @property
    def primary_drop(self) -> float:
        return self.drops[PRIMARY_METRIC]
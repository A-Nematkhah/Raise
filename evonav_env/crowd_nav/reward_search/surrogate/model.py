"""
Fit / predict / uncertainty API for the surrogate (stub).

Persist under ``artifacts/surrogate/`` (model + metrics JSON).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SurrogatePrediction:
    """Point estimate plus a crude uncertainty score in ``[0, +inf)``."""

    y_hat: Dict[str, float]
    uncertainty: float
    model_id: str = ""


class SurrogateModel:
    """Offline-trained predictor. Train via ``fit``, load via ``load``."""

    def fit(
        self,
        features: Sequence[Dict[str, Any]],
        labels: Sequence[Dict[str, Any]],
        *,
        target_keys: Sequence[str] = ("scalar",),
    ) -> Dict[str, Any]:
        raise NotImplementedError("SurrogateModel.fit — see surrogate/PLAN.md")

    def predict(self, features: Dict[str, Any]) -> SurrogatePrediction:
        raise NotImplementedError("SurrogateModel.predict — see surrogate/PLAN.md")

    def save(self, directory: str) -> None:
        raise NotImplementedError("SurrogateModel.save — see surrogate/PLAN.md")

    @classmethod
    def load(cls, directory: str) -> "SurrogateModel":
        raise NotImplementedError("SurrogateModel.load — see surrogate/PLAN.md")


def default_model_dir() -> str:
    return "artifacts/surrogate"

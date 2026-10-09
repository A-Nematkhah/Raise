"""Score1 result container (computed by the domain pack's Stage I scorer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Score1Result:
    """Score1 scalar plus degeneracy / holdout diagnostics for reporting."""

    score: float
    degenerate_fraction: float
    n_pairs: int = 0
    n_degenerate: int = 0
    rejected: bool = False
    reject_reason: Optional[str] = None
    raw_score: Optional[float] = None
    train_score: Optional[float] = None
    holdout_score: Optional[float] = None
    # Per-scenario mean Spearman (train split).
    scenario_scores: Optional[Dict[str, float]] = None

    def __float__(self) -> float:
        return float(self.score)

    def __gt__(self, other: object) -> bool:
        return float(self) > float(other)  # type: ignore[arg-type]

    def __ge__(self, other: object) -> bool:
        return float(self) >= float(other)  # type: ignore[arg-type]

    def __lt__(self, other: object) -> bool:
        return float(self) < float(other)  # type: ignore[arg-type]

    def __le__(self, other: object) -> bool:
        return float(self) <= float(other)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Score1Result):
            return self.as_dict() == other.as_dict()
        if isinstance(other, (int, float)):
            return float(self) == float(other)
        return NotImplemented

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "degenerate_fraction": float(self.degenerate_fraction),
            "n_pairs": int(self.n_pairs),
            "n_degenerate": int(self.n_degenerate),
            "rejected": bool(self.rejected),
            "reject_reason": self.reject_reason,
            "raw_score": (
                None if self.raw_score is None else float(self.raw_score)
            ),
            "train_score": (
                None if self.train_score is None else float(self.train_score)
            ),
            "holdout_score": (
                None if self.holdout_score is None else float(self.holdout_score)
            ),
            "scenario_scores": (
                None
                if self.scenario_scores is None
                else {str(k): float(v) for k, v in self.scenario_scores.items()}
            ),
        }

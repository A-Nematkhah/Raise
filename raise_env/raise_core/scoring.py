"""Score_1 result container (EvoNav-style Stage I scorer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Score1Result:
    """EvoNav Score_1 scalar plus diagnostics for Stage I ranking / LLM feedback."""

    score: float
    per_scenario: Optional[Dict[str, float]] = None
    n_nan_frames: int = 0
    n_frames: int = 0
    error: Optional[str] = None
    rejected: bool = False
    reject_reason: Optional[str] = None

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
            "per_scenario": (
                None
                if self.per_scenario is None
                else {str(k): float(v) for k, v in self.per_scenario.items()}
            ),
            "n_nan_frames": int(self.n_nan_frames),
            "n_frames": int(self.n_frames),
            "error": self.error,
            "rejected": bool(self.rejected),
            "reject_reason": self.reject_reason,
        }

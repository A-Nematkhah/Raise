"""The evolvable reward individual shared by every RAISE stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from raise_core.sandbox.runtime import RewardFunction


@dataclass
class RewardCandidate:
    """One evolvable reward individual."""

    candidate_id: str
    code: str
    reward_fn: Optional[RewardFunction] = None
    score: Optional[float] = None
    valid: bool = False
    origin: str = "unknown"  # initial | mutation | crossover | random
    parent_ids: tuple = ()
    validation_error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        return self.valid and self.reward_fn is not None

    def as_reward_function(self) -> RewardFunction:
        """Return the validated reward callable (Stage I Score1 entry point)."""
        if self.reward_fn is None:
            raise ValueError(f"Candidate {self.candidate_id} has no reward_fn")
        return self.reward_fn

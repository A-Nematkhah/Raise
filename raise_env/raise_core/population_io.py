"""Candidate population <-> JSON rows (shared by closed-loop and Stage III checkpoints)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from domains.crowdnav.reporting import candidate_to_dict, load_candidate_dict
from raise_core.explore import RewardCandidate


def serialize_population(population: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """Candidates (or already-serialized dict rows) -> JSON-safe rows."""
    if not population:
        return []
    out: List[Dict[str, Any]] = []
    for cand in population:
        if hasattr(cand, "candidate_id"):
            out.append(candidate_to_dict(cand))
        elif isinstance(cand, dict):
            out.append(dict(cand))
    return out


def deserialize_population(
    rows: Optional[Sequence[Dict[str, Any]]],
    *,
    validator: Any = None,
) -> List[RewardCandidate]:
    """Rebuild ``RewardCandidate`` objects (re-validates code with ``validator``)."""
    if not rows:
        return []
    return [
        load_candidate_dict(dict(row), validator=validator)
        for row in rows
        if isinstance(row, dict)
    ]


__all__ = ["deserialize_population", "serialize_population"]

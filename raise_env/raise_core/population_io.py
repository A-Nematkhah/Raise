"""Candidate population <-> JSON rows (shared by closed-loop and Stage III checkpoints)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from raise_core.candidate import RewardCandidate


def candidate_to_dict(cand: RewardCandidate) -> Dict[str, Any]:
    return {
        "candidate_id": cand.candidate_id,
        "code": cand.code,
        "score": cand.score,
        "valid": cand.valid,
        "origin": cand.origin,
        "parent_ids": list(cand.parent_ids),
        "validation_error": cand.validation_error,
        "metadata": dict(cand.metadata or {}),
    }


def load_candidate_dict(d: Dict[str, Any], *, validator: Any) -> RewardCandidate:
    """Rebuild one candidate; ``validator`` re-validates the stored code."""
    code = d["code"]
    reward_fn, err = validator.try_validate(code)
    return RewardCandidate(
        candidate_id=d.get("candidate_id", "loaded"),
        code=code,
        reward_fn=reward_fn,
        score=d.get("score"),
        valid=reward_fn is not None,
        origin=d.get("origin", "loaded"),
        parent_ids=tuple(d.get("parent_ids") or ()),
        validation_error=err or d.get("validation_error"),
        metadata=dict(d.get("metadata") or {}),
    )


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
    validator: Any,
) -> List[RewardCandidate]:
    """Rebuild ``RewardCandidate`` objects (re-validates code with ``validator``)."""
    if not rows:
        return []
    return [
        load_candidate_dict(dict(row), validator=validator)
        for row in rows
        if isinstance(row, dict)
    ]


__all__ = [
    "candidate_to_dict",
    "deserialize_population",
    "load_candidate_dict",
    "serialize_population",
]

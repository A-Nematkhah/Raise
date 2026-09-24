"""
Feature extraction for the Stage-II surrogate.

See ``PLAN.md`` §3 (schema v1, locked 2026-09-20).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence

from raise_core.surrogate import FEATURE_SCHEMA_VERSION

# Domain smoke states for behavior fingerprint (set by closed-loop / bootstrap).
_SMOKE_STATES_OVERRIDE: Optional[Sequence[Any]] = None


def set_behavior_smoke_states(states: Optional[Sequence[Any]]) -> None:
    """Override CrowdNav default smoke states for fingerprinting (highway, etc.)."""
    global _SMOKE_STATES_OVERRIDE
    _SMOKE_STATES_OVERRIDE = None if states is None else tuple(states)


def feature_schema_version() -> str:
    return FEATURE_SCHEMA_VERSION


def normalize_reward_code(code: str) -> str:
    """Stable whitespace normalization before hashing / length."""
    return "\n".join(line.rstrip() for line in str(code).strip().splitlines()).strip()


def code_sha256(code: str) -> str:
    normalized = normalize_reward_code(code)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _score1_fields(score1_result: Any) -> Dict[str, Any]:
    if score1_result is None:
        return {
            "score1": None,
            "score1_raw": None,
            "score1_holdout": None,
            "score1_train": None,
            "score1_degen": None,
            "score1_rejected": None,
            "score1_reject_reason": None,
            "score1_worst_k": None,
            "score1_worst_mean": None,
        }

    score = getattr(score1_result, "score", None)
    score_f = _finite_or_none(score)
    # Keep explicit -inf as null for JSON; rejected flag carries the signal.
    if score is not None and score_f is None:
        try:
            if float(score) == float("-inf"):
                score_f = None
        except (TypeError, ValueError):
            pass

    scenario_scores = getattr(score1_result, "scenario_scores", None) or None
    worst_k: Optional[List[Dict[str, Any]]] = None
    worst_mean: Optional[float] = None
    if isinstance(scenario_scores, dict) and scenario_scores:
        ranked = sorted(
            ((str(sid), float(rho)) for sid, rho in scenario_scores.items()),
            key=lambda item: item[1],
        )[:3]
        worst_k = [{"sid": sid, "rho": rho} for sid, rho in ranked]
        worst_mean = sum(rho for _, rho in ranked) / float(len(ranked))

    return {
        "score1": score_f,
        "score1_raw": _finite_or_none(getattr(score1_result, "raw_score", None)),
        "score1_holdout": _finite_or_none(getattr(score1_result, "holdout_score", None)),
        "score1_train": _finite_or_none(getattr(score1_result, "train_score", None)),
        "score1_degen": _finite_or_none(getattr(score1_result, "degenerate_fraction", None)),
        "score1_rejected": bool(getattr(score1_result, "rejected", False)),
        "score1_reject_reason": getattr(score1_result, "reject_reason", None),
        "score1_worst_k": worst_k,
        "score1_worst_mean": worst_mean,
    }


def _behavior_fingerprint(
    candidate: Any,
    *,
    smoke_states: Optional[Sequence[Any]] = None,
) -> List[float]:
    from raise_core.sandbox.runtime import default_smoke_states

    reward_fn = getattr(candidate, "reward_fn", None)
    if reward_fn is None and hasattr(candidate, "as_reward_function"):
        try:
            reward_fn = candidate.as_reward_function()
        except Exception:  # noqa: BLE001
            reward_fn = None
    if reward_fn is None:
        return []

    states = smoke_states
    if states is None:
        states = _SMOKE_STATES_OVERRIDE
    if states is None:
        states = default_smoke_states()

    values: List[float] = []
    for state in states:
        try:
            if hasattr(reward_fn, "reset"):
                reward_fn.reset()
            raw = reward_fn.compute(state)
            number = float(raw)
            if not math.isfinite(number):
                values.append(0.0)
            else:
                values.append(number)
        except Exception:  # noqa: BLE001
            values.append(0.0)
    return values


def extract_candidate_features(
    candidate: Any,
    *,
    score1_result: Any = None,
    extra: Dict[str, Any] | None = None,
    label_budget: str = "stage2_short",
    smoke_states: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable feature dict for one reward candidate.
    """
    code = str(getattr(candidate, "code", "") or "")
    normalized = normalize_reward_code(code)
    digest = code_sha256(code)
    payload: Dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "candidate_id": str(getattr(candidate, "candidate_id", "") or ""),
        "code_hash": digest,
        "code_len": int(len(normalized)),
        "label_budget": str(label_budget),
        "behavior_fingerprint": _behavior_fingerprint(
            candidate, smoke_states=smoke_states
        ),
    }
    payload.update(_score1_fields(score1_result))
    if extra:
        for key, value in extra.items():
            if key not in payload:
                payload[key] = value
    return payload


def example_id_for_features(features: Dict[str, Any]) -> str:
    """``{code_hash[:12]}_{label_budget}`` as in PLAN.md §5 Resume."""
    digest = str(features.get("code_hash") or "")
    budget = str(features.get("label_budget") or "stage2_short")
    if len(digest) < 12:
        raise ValueError("features.code_hash must be a SHA256 hex string")
    return f"{digest[:12]}_{budget}"

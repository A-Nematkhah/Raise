"""
Feature extraction for the Stage-II/III surrogate (stub).

Tomorrow: implement ``extract_candidate_features`` per PLAN.md §3.
"""

from __future__ import annotations

from typing import Any, Dict

from crowd_nav.reward_search.surrogate import FEATURE_SCHEMA_VERSION


def extract_candidate_features(
    candidate: Any,
    *,
    score1_result: Any = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable feature dict for one reward candidate.

    Raises
    ------
    NotImplementedError
        Until the bootstrap implementation lands.
    """
    raise NotImplementedError(
        "surrogate.features.extract_candidate_features — see surrogate/PLAN.md"
    )


def feature_schema_version() -> str:
    return FEATURE_SCHEMA_VERSION

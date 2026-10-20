"""Optional sanity flags for already-computed candidate evidence."""

from __future__ import annotations

import math
from typing import Any, Mapping

from raise_core.evo_selection.config import SelectionConfig


def candidate_guard_flags(candidate: Any, config: SelectionConfig) -> list[str]:
    if not config.guards:
        return []

    flags: list[str] = []
    if not bool(getattr(candidate, "valid", False)):
        flags.append("invalid_reward_code")

    score = getattr(candidate, "score", None)
    if score is not None:
        try:
            if not math.isfinite(float(score)):
                flags.append("non_finite_score")
        except (TypeError, ValueError):
            flags.append("invalid_score")

    metadata = getattr(candidate, "metadata", None) or {}
    metrics = metadata.get("last_metrics")
    if not isinstance(metrics, Mapping):
        return flags
    holdout = metrics.get("holdout")
    values = holdout if isinstance(holdout, Mapping) else metrics

    fitness = metadata.get("fitness")
    if fitness is not None:
        try:
            if not math.isfinite(float(fitness)):
                flags.append("non_finite_fitness")
        except (TypeError, ValueError):
            flags.append("invalid_fitness")

    try:
        success_rate = float(values.get("SR", values.get("sr")))
        path_length = float(values.get("path_length", values.get("PL")))
    except (TypeError, ValueError):
        return flags
    if (
        math.isfinite(success_rate)
        and math.isfinite(path_length)
        and success_rate >= config.guard_success_threshold
        and path_length <= config.guard_min_path_length
    ):
        flags.append("high_success_near_zero_progress")
    return flags

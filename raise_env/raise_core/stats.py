"""Domain-agnostic fitness SE helpers used by Stage II / III selection.

``fitness_standard_error`` and ``paired_difference`` live here so
``raise_core.selection`` never soft-depends on ``domains.crowdnav``. CrowdNav
re-exports the same symbols from ``domains.crowdnav.metrics`` for existing
imports.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "fitness_standard_error",
    "paired_difference",
]


def fitness_standard_error(selection_scores: Sequence[float]) -> float:
    """Standard error of the mean (0.0 for fewer than two samples)."""
    n = len(selection_scores)
    if n < 2:
        return 0.0
    mean = sum(float(x) for x in selection_scores) / n
    var = sum((float(x) - mean) ** 2 for x in selection_scores) / (n - 1)
    return float(math.sqrt(var) / math.sqrt(n))


def _episode_value(row: Mapping[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None and key == "selection_score":
        outcome = str(row.get("outcome", ""))
        scores = {"success": 1.0, "collision": -1.0, "timeout": -0.5}
        if outcome not in scores:
            return None
        value = scores[outcome]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _by_case_averaged(
    eps: Sequence[Mapping[str, Any]], *, key: str
) -> Optional[Dict[Any, float]]:
    """Map ``case -> mean(value)``; average duplicate cases instead of failing."""
    buckets: Dict[Any, List[float]] = {}
    for row in eps:
        if not isinstance(row, Mapping) or "case" not in row:
            return None
        value = _episode_value(row, key)
        if value is None:
            return None
        buckets.setdefault(row["case"], []).append(value)
    return {case: sum(vals) / len(vals) for case, vals in buckets.items()}


def paired_difference(
    eps_a: Sequence[Mapping[str, Any]],
    eps_b: Sequence[Mapping[str, Any]],
    *,
    key: str = "selection_score",
) -> Optional[Dict[str, float]]:
    """Mean per-case difference and its SE when both lists share the same cases.

    Duplicate ``case`` values (e.g. eval cycles when ``n_episodes > case_set_size``)
    are averaged per case rather than rejecting the pairing. Returns ``None``
    when cases differ or fewer than two pairs are available.
    """
    map_a = _by_case_averaged(eps_a, key=key)
    map_b = _by_case_averaged(eps_b, key=key)
    if map_a is None or map_b is None:
        return None
    if set(map_a) != set(map_b) or len(map_a) < 2:
        return None
    diffs = [map_b[c] - map_a[c] for c in sorted(map_a, key=lambda x: (str(type(x)), x))]
    mean = sum(diffs) / len(diffs)
    se = fitness_standard_error(diffs)
    return {"mean": float(mean), "se": float(se), "n": float(len(diffs))}

"""
Fitness helpers for Stage II / III selection.

Highway trainers cache ``fitness`` (``highway_fitness`` of the holdout eval,
alias ``selection_scalar``) on candidate metadata; that scalar ranks parents
and picks the final best. Metrics without a highway ``domain`` tag (stub
trainers) fall back to ``SR - CR - 0.5 * TR``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from domains.highway.metrics import highway_fitness
from raise_core.explore import RewardCandidate


def is_highway_metrics(metrics: Mapping[str, Any]) -> bool:
    """Canonical check: Stage-II metrics dict is from the highway domain pack."""
    return str(metrics.get("domain", "")).strip().lower() == "highway"


def navigation_scalar(sr: float, cr: float, tr: float) -> float:
    """Fallback scalar for metrics without a highway fitness."""
    return float(sr) - float(cr) - 0.5 * float(tr)


def navigation_scalar_from_dict(metrics: Optional[Mapping[str, Any]]) -> float:
    if not metrics:
        return float("-inf")
    for key in ("fitness", "selection_scalar"):
        if metrics.get(key) is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                pass
    if is_highway_metrics(metrics):
        return float(highway_fitness(metrics))
    sr = float(metrics.get("SR", metrics.get("sr", 0.0)))
    cr = float(metrics.get("CR", metrics.get("cr", 0.0)))
    tr = float(metrics.get("TR", metrics.get("tr", 0.0)))
    return navigation_scalar(sr, cr, tr)


def candidate_nav_scalar(candidate: RewardCandidate) -> float:
    md = candidate.metadata or {}
    for key in ("fitness", "selection_scalar"):
        if md.get(key) is not None:
            try:
                return float(md[key])
            except (TypeError, ValueError):
                pass
    return navigation_scalar_from_dict(md.get("last_metrics"))


def candidate_fitness(candidate: RewardCandidate) -> float:
    """Fitness used for ranking (cached highway holdout fitness when present)."""
    return candidate_nav_scalar(candidate)


def pick_best_trained(
    snapshots: Sequence[RewardCandidate],
) -> Optional[RewardCandidate]:
    """Return the trained snapshot with highest fitness, if any."""
    scored = [c for c in snapshots if candidate_nav_scalar(c) > float("-inf")]
    if not scored:
        return None
    return max(scored, key=candidate_nav_scalar)


def is_same_genome(a: RewardCandidate, b: RewardCandidate) -> bool:
    return a.code.strip() == b.code.strip()

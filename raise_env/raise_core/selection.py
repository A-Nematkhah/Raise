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
from raise_core.candidate import RewardCandidate

_FITNESS_KEYS = ("fitness", "selection_scalar")


def is_highway_metrics(metrics: Mapping[str, Any]) -> bool:
    """Canonical check: Stage-II metrics dict is from the highway domain pack."""
    return str(metrics.get("domain", "")).strip().lower() == "highway"


def _cached(source: Mapping[str, Any]) -> Optional[float]:
    for key in _FITNESS_KEYS:
        if source.get(key) is not None:
            try:
                return float(source[key])
            except (TypeError, ValueError):
                pass
    return None


def metrics_fitness(metrics: Optional[Mapping[str, Any]]) -> float:
    """Fitness of a Stage II metrics dict (``-inf`` without metrics)."""
    if not metrics:
        return float("-inf")
    cached = _cached(metrics)
    if cached is not None:
        return cached
    if is_highway_metrics(metrics):
        return float(highway_fitness(metrics))
    sr = float(metrics.get("SR", metrics.get("sr", 0.0)))
    cr = float(metrics.get("CR", metrics.get("cr", 0.0)))
    tr = float(metrics.get("TR", metrics.get("tr", 0.0)))
    return sr - cr - 0.5 * tr


def candidate_fitness(candidate: RewardCandidate) -> float:
    """Fitness used for ranking (cached highway holdout fitness when present)."""
    md = candidate.metadata or {}
    cached = _cached(md)
    return cached if cached is not None else metrics_fitness(md.get("last_metrics"))


def pick_best_trained(
    snapshots: Sequence[RewardCandidate],
) -> Optional[RewardCandidate]:
    """Return the trained snapshot with highest fitness, if any."""
    scored = [c for c in snapshots if candidate_fitness(c) > float("-inf")]
    if not scored:
        return None
    return max(scored, key=candidate_fitness)

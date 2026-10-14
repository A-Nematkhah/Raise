"""
Fitness helpers for Stage II / III selection.

Domain trainers cache ``fitness`` (the pack's official holdout objective,
alias ``selection_scalar``) on candidate metadata; that scalar ranks parents
and picks the final best. Metrics without a cached fitness (stub trainers)
fall back to ``SR - CR - 0.5 * TR``.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence

from raise_core.candidate import RewardCandidate

_FITNESS_KEYS = ("fitness", "selection_scalar")


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
    sr = float(metrics.get("SR", metrics.get("sr", 0.0)))
    cr = float(metrics.get("CR", metrics.get("cr", 0.0)))
    tr = float(metrics.get("TR", metrics.get("tr", 0.0)))
    return sr - cr - 0.5 * tr


def candidate_fitness(candidate: RewardCandidate) -> float:
    """Fitness used for ranking (cached domain holdout fitness when present)."""
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


def candidate_fitness_se(candidate: RewardCandidate) -> float:
    """Bootstrap SE of the cached fitness (0.0 when the trainer reports none)."""
    metrics = (candidate.metadata or {}).get("last_metrics") or {}
    try:
        se = float(metrics.get("fitness_se"))
    except (TypeError, ValueError):
        return 0.0
    return se if math.isfinite(se) and se > 0.0 else 0.0


def statistical_ties(
    snapshots: Sequence[RewardCandidate],
    best: RewardCandidate,
    *,
    z: float = 2.0,
) -> List[str]:
    """Ids whose fitness is within ``z * max(se_best, se_i)`` of ``best``.

    A gap inside that band is evaluation noise, so the argmax among these
    candidates is not a meaningful ranking.
    """
    best_fit = candidate_fitness(best)
    best_se = candidate_fitness_se(best)
    ties: List[str] = []
    for cand in snapshots:
        if cand.candidate_id == best.candidate_id:
            continue
        fit = candidate_fitness(cand)
        if fit == float("-inf"):
            continue
        band = z * max(best_se, candidate_fitness_se(cand))
        if best_fit - fit <= band:
            ties.append(str(cand.candidate_id))
    return ties

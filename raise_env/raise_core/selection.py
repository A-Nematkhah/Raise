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


def _holdout_episodes(candidate: RewardCandidate) -> Optional[Sequence[Mapping[str, Any]]]:
    metrics = (candidate.metadata or {}).get("last_metrics") or {}
    if not isinstance(metrics, Mapping):
        return None
    holdout = metrics.get("holdout") if isinstance(metrics.get("holdout"), Mapping) else metrics
    if not isinstance(holdout, Mapping):
        return None
    eps = holdout.get("holdout_episodes", metrics.get("holdout_episodes"))
    return eps if isinstance(eps, list) else None


def statistical_ties(
    snapshots: Sequence[RewardCandidate],
    best: RewardCandidate,
    *,
    z: float = 2.0,
) -> List[str]:
    """Ids whose fitness is within ``z`` SE of ``best``.

    When both candidates carry ``holdout_episodes`` with identical ``case``
    lists, the band is the paired SE of the per-case task-return difference
    (tighter). Otherwise falls back to ``z * max(se_best, se_i)`` with
    independent SEs.
    """
    best_fit = candidate_fitness(best)
    best_se = candidate_fitness_se(best)
    best_eps = _holdout_episodes(best)
    ties: List[str] = []
    for cand in snapshots:
        if cand.candidate_id == best.candidate_id:
            continue
        fit = candidate_fitness(cand)
        if fit == float("-inf"):
            continue
        band: Optional[float] = None
        if best_eps is not None:
            other_eps = _holdout_episodes(cand)
            if other_eps is not None:
                try:
                    from domains.crowdnav.metrics import paired_difference
                except ImportError:  # pragma: no cover - domain optional in unit stubs
                    paired_difference = None  # type: ignore[assignment]
                if paired_difference is not None:
                    paired = paired_difference(other_eps, best_eps, key="task_return")
                    if paired is None:
                        paired = paired_difference(other_eps, best_eps, key="disc_return")
                    if paired is not None:
                        band = z * float(paired["se"])
                        # Paired mean may differ slightly from cached fitness gap;
                        # still use the fitness gap for the decision, paired SE only.
        if band is None:
            band = z * max(best_se, candidate_fitness_se(cand))
        if best_fit - fit <= band:
            ties.append(str(cand.candidate_id))
    return ties

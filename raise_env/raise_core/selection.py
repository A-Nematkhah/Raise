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

from raise_core import console
from raise_core.candidate import RewardCandidate
from raise_core.stats import paired_difference

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
    """Return the trained snapshot with highest fitness, if any.

    Candidates with ``partial_seeds=True`` rank below full-seed candidates at
    equal fitness (so means over different seed counts are not compared as equals).
    """
    scored = [c for c in snapshots if candidate_fitness(c) > float("-inf")]
    if not scored:
        return None

    def _key(c: RewardCandidate) -> tuple:
        md = c.metadata or {}
        partial = bool(md.get("partial_seeds"))
        return (0 if partial else 1, candidate_fitness(c))

    return max(scored, key=_key)


def candidate_fitness_se(candidate: RewardCandidate) -> float:
    """Bootstrap SE of the cached fitness (0.0 when the trainer reports none)."""
    metrics = (candidate.metadata or {}).get("last_metrics") or {}
    try:
        se = float(metrics.get("fitness_se"))
    except (TypeError, ValueError):
        return 0.0
    return se if math.isfinite(se) and se > 0.0 else 0.0


def candidate_fitness_se_source(candidate: RewardCandidate) -> str:
    """``episodes_only`` or ``across_seeds`` (default episodes_only)."""
    metrics = (candidate.metadata or {}).get("last_metrics") or {}
    if not isinstance(metrics, Mapping):
        return "episodes_only"
    src = str(metrics.get("fitness_se_source") or "episodes_only")
    return src if src in ("episodes_only", "across_seeds") else "episodes_only"


def _holdout_episodes(candidate: RewardCandidate) -> Optional[Sequence[Mapping[str, Any]]]:
    metrics = (candidate.metadata or {}).get("last_metrics") or {}
    if not isinstance(metrics, Mapping):
        return None
    holdout = metrics.get("holdout") if isinstance(metrics.get("holdout"), Mapping) else metrics
    if not isinstance(holdout, Mapping):
        return None
    eps = holdout.get("holdout_episodes", metrics.get("holdout_episodes"))
    return eps if isinstance(eps, list) else None


def _single_seed(candidate: RewardCandidate) -> bool:
    """True when fitness is from one seed (paired-episode band is meaningful)."""
    md = candidate.metadata or {}
    n = md.get("stage3_n_seeds") or md.get("stage2_n_seeds") or 1
    try:
        return int(n) <= 1
    except (TypeError, ValueError):
        return True


def statistical_ties(
    snapshots: Sequence[RewardCandidate],
    best: RewardCandidate,
    *,
    z: float = 2.0,
) -> List[str]:
    """Ids whose fitness is within ``z`` SE of ``best``.

    Uses the paired per-episode band only when both fitnesses are single-seed
    on identical cases. Otherwise uses ``z * sqrt(se_a**2 + se_b**2)`` from the
    cached ``fitness_se`` values (cross-seed SE when ``fitness_se_source ==
    "across_seeds"``). Emits a console warning when ties rely on ``episodes_only``.
    """
    best_fit = candidate_fitness(best)
    best_se = candidate_fitness_se(best)
    best_src = candidate_fitness_se_source(best)
    best_eps = _holdout_episodes(best)
    best_single = _single_seed(best)
    ties: List[str] = []
    used_episodes_only = False
    for cand in snapshots:
        if cand.candidate_id == best.candidate_id:
            continue
        fit = candidate_fitness(cand)
        if fit == float("-inf"):
            continue
        band: Optional[float] = None
        if best_single and _single_seed(cand) and best_eps is not None:
            other_eps = _holdout_episodes(cand)
            if other_eps is not None:
                paired = paired_difference(
                    other_eps, best_eps, key="selection_score"
                )
                if paired is not None:
                    band = z * float(paired["se"])
        if band is None:
            other_se = candidate_fitness_se(cand)
            band = z * math.sqrt(best_se * best_se + other_se * other_se)
            if (
                best_src == "episodes_only"
                or candidate_fitness_se_source(cand) == "episodes_only"
            ):
                used_episodes_only = True
        if best_fit - fit <= band:
            ties.append(str(cand.candidate_id))
    if ties and used_episodes_only:
        console.warn(
            "statistical_ties used episodes_only SE (not cross-seed); "
            "tie set may be optimistic when fitness is a mean over seeds",
            stage="selection",
        )
    return ties

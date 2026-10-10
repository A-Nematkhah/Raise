"""Order genomes for closed-loop ``_next_generation``.

``scalar`` (default): trained candidates by ``highway_fitness`` of their Stage II
evaluation; untrained seats follow, ordered by Score1.
``score1``: analytical Score1 only. ``hybrid``: normalized blend of both.
"""

from __future__ import annotations

from typing import List, Literal, Sequence, Tuple

from raise_core.candidate import RewardCandidate
from raise_core.selection import candidate_fitness, candidate_nav_scalar

EvolveRankMode = Literal["score1", "scalar", "hybrid"]

_VALID = frozenset({"score1", "scalar", "hybrid"})


def parse_evolve_rank(value: object) -> EvolveRankMode:
    """Default ``scalar`` when unset/empty/unknown."""
    raw = str(value or "").strip().lower()
    if raw in _VALID:
        return raw  # type: ignore[return-value]
    return "scalar"


def _score1(c: RewardCandidate) -> float:
    if c.score is None:
        return float("-inf")
    try:
        return float(c.score)
    except (TypeError, ValueError):
        return float("-inf")


def _has_proxy(c: RewardCandidate) -> bool:
    md = c.metadata or {}
    if md.get("fitness") is not None or md.get("selection_scalar") is not None:
        return True
    metrics = md.get("last_metrics")
    return isinstance(metrics, dict) and bool(metrics)


def rank_population_for_evolution(
    population: Sequence[RewardCandidate],
    *,
    mode: EvolveRankMode = "scalar",
    hybrid_score1_weight: float = 0.4,
) -> List[RewardCandidate]:
    """Return a new list sorted best-first for crossover/mutation parents."""
    mode_key = parse_evolve_rank(mode)
    pop = list(population)
    if not pop:
        return pop
    if mode_key == "score1":
        return sorted(pop, key=_score1, reverse=True)

    labeled = [c for c in pop if _has_proxy(c)]
    unlabeled = [c for c in pop if not _has_proxy(c)]

    if mode_key == "scalar":

        def _scalar_key(c: RewardCandidate) -> Tuple[float, float]:
            return (candidate_fitness(c), _score1(c))

        ranked_l = sorted(labeled, key=_scalar_key, reverse=True)
        ranked_u = sorted(unlabeled, key=_score1, reverse=True)
        return ranked_l + ranked_u

    # hybrid
    w = float(hybrid_score1_weight)
    w = min(1.0, max(0.0, w))
    if not labeled:
        return sorted(pop, key=_score1, reverse=True)

    s1_vals = [_score1(c) for c in labeled]
    sc_vals = [candidate_nav_scalar(c) for c in labeled]
    s1_lo, s1_hi = min(s1_vals), max(s1_vals)
    sc_lo, sc_hi = min(sc_vals), max(sc_vals)

    def _norm(v: float, lo: float, hi: float) -> float:
        if not (v == v):  # NaN
            return 0.0
        if hi <= lo + 1e-12:
            return 0.5
        return (v - lo) / (hi - lo)

    def _hybrid_key(c: RewardCandidate) -> float:
        s1n = _norm(_score1(c), s1_lo, s1_hi)
        scn = _norm(candidate_nav_scalar(c), sc_lo, sc_hi)
        return w * s1n + (1.0 - w) * scn

    ranked_l = sorted(labeled, key=_hybrid_key, reverse=True)
    ranked_u = sorted(unlabeled, key=_score1, reverse=True)
    return ranked_l + ranked_u

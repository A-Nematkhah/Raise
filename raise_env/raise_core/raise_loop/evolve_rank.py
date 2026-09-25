"""Order genomes for closed-loop ``_next_generation``.

CrowdNav / paper path keeps Score1-only ranking (``evolve_rank=score1``).

Highway defaults to ``evolve_rank=scalar``: labeled parents ordered by
``highway_fitness`` (holdout) so each generation breeds from the best
navigation objective. Unlabeled seats stay Score1-ordered.

``pareto`` remains available as an explicit diagnostic mode (not the default).
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence, Tuple

from raise_core.explore import RewardCandidate
from raise_core.selection import candidate_fitness, candidate_nav_scalar

EvolveRankMode = Literal["score1", "scalar", "hybrid", "pareto"]

_VALID = frozenset({"score1", "scalar", "hybrid", "pareto"})


def parse_evolve_rank(value: object, *, domain: str = "crowdnav") -> EvolveRankMode:
    """Default: score1 for crowdnav; scalar (fitness) for highway when unset/empty."""
    raw = str(value or "").strip().lower()
    if raw in _VALID:
        return raw  # type: ignore[return-value]
    if str(domain).strip().lower() == "highway":
        return "scalar"
    return "score1"


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
    if md.get("pareto_rank") is not None:
        return True
    metrics = md.get("last_metrics")
    return isinstance(metrics, dict) and bool(metrics)


def _metrics_blob(c: RewardCandidate) -> Optional[dict]:
    md = c.metadata or {}
    m = md.get("last_metrics")
    return dict(m) if isinstance(m, dict) else None


def rank_population_pareto(
    population: Sequence[RewardCandidate],
    *,
    reference_speed_samples: Any = None,
    reference_cr: Optional[float] = None,
    reference_tr: Optional[float] = None,
    ref: Any = None,
) -> List[RewardCandidate]:
    """
    Order labeled candidates via highway Pareto pipeline; unlabeled by Score1
    after all labeled. Diagnostic / optional — not the highway breeding default.
    """
    from domains.highway.pareto_rank import (
        calibrate_from_population,
        calibrate_from_reference_rollout,
        metrics_from_mapping,
        rank_population,
        stamp_pareto_ranks,
    )

    pop = list(population)
    labeled = [c for c in pop if _has_proxy(c) and _metrics_blob(c)]
    labeled_ids = {id(c) for c in labeled}
    unlabeled = [c for c in pop if id(c) not in labeled_ids]

    if not labeled:
        return sorted(pop, key=_score1, reverse=True)

    metrics_list = []
    for c in labeled:
        blob = _metrics_blob(c) or {}
        metrics_list.append(metrics_from_mapping(str(c.candidate_id), blob))

    if ref is None:
        if reference_speed_samples is not None:
            ref = calibrate_from_reference_rollout(
                reference_speed_samples,
                float(reference_cr or 0.0),
                float(reference_tr or 0.0),
            )
        if ref is None:
            ref = calibrate_from_population(metrics_list)

    ordered_m = rank_population(metrics_list, ref=ref)
    stamp_pareto_ranks(labeled, ordered_m, ref=ref)

    by_id = {str(m.candidate_id): i for i, m in enumerate(ordered_m)}
    ranked_l = sorted(
        labeled,
        key=lambda c: by_id.get(str(c.candidate_id), 10**9),
    )
    ranked_u = sorted(unlabeled, key=_score1, reverse=True)
    return ranked_l + ranked_u


def rank_population_for_evolution(
    population: Sequence[RewardCandidate],
    *,
    mode: EvolveRankMode = "score1",
    hybrid_score1_weight: float = 0.4,
    reference_speed_samples: Any = None,
    reference_cr: Optional[float] = None,
    reference_tr: Optional[float] = None,
    pareto_ref: Any = None,
) -> List[RewardCandidate]:
    """
    Return a new list sorted best-first for crossover/mutation parents.

    Modes
    -----
    score1
        Analytical Score1 only (CrowdNav / Alg.1 default).
    scalar
        Highway default: labeled by ``highway_fitness`` / nav scalar; unlabeled
        by Score1 after all labeled.
    pareto
        Diagnostic: auto-calibrated feasibility + Pareto / crowding.
    hybrid
        Among labeled: ``w·norm(Score1) + (1-w)·norm(fitness)``.
    """
    mode_key = str(mode).strip().lower()
    if mode_key not in _VALID:
        mode_key = "score1"
    pop = list(population)
    if not pop:
        return pop
    if mode_key == "score1":
        return sorted(pop, key=_score1, reverse=True)

    if mode_key == "pareto":
        return rank_population_pareto(
            pop,
            reference_speed_samples=reference_speed_samples,
            reference_cr=reference_cr,
            reference_tr=reference_tr,
            ref=pareto_ref,
        )

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

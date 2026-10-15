"""Order genomes for closed-loop ``_next_generation``.

Trained candidates by domain fitness of their Stage II evaluation (Score1
breaks ties); candidates without Stage II evidence follow, ordered by Score1.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from raise_core.candidate import RewardCandidate
from raise_core.selection import candidate_fitness


def _score1(c: RewardCandidate) -> float:
    if c.score is None:
        return float("-inf")
    try:
        value = float(c.score)
    except (TypeError, ValueError):
        return float("-inf")
    if value != value:  # NaN
        return float("-inf")
    return value


def _has_proxy(c: RewardCandidate) -> bool:
    md = c.metadata or {}
    if md.get("fitness") is not None or md.get("selection_scalar") is not None:
        return True
    metrics = md.get("last_metrics")
    return isinstance(metrics, dict) and bool(metrics)


def rank_population_for_evolution(
    population: Sequence[RewardCandidate],
) -> List[RewardCandidate]:
    """Return a new list sorted best-first for crossover/mutation parents."""

    def _fitness_key(c: RewardCandidate) -> Tuple[float, float]:
        return (candidate_fitness(c), _score1(c))

    labeled = [c for c in population if _has_proxy(c)]
    unlabeled = [c for c in population if not _has_proxy(c)]
    return (
        sorted(labeled, key=_fitness_key, reverse=True)
        + sorted(unlabeled, key=_score1, reverse=True)
    )

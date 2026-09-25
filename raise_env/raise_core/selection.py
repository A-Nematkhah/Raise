"""
Fitness / navigation helpers for Stage II / III selection.

Paper (Alg. 1): final rankings R2 / R3 use LLM evaluation of multi-objective
M(r). Pipeline default is ``final_rank=llm`` (see ``ranking.py``).

Highway uses an explicit ``fitness`` (``highway_fitness`` on holdout metrics).
CrowdNav keeps the engineering scalar:

    SR - CR - 0.5 * TR

``candidate_fitness`` / ``candidate_nav_scalar`` read cached ``fitness`` or
``selection_scalar`` when present.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from raise_core.explore import RewardCandidate


def navigation_scalar(sr: float, cr: float, tr: float) -> float:
    """Baseline elite scalar (not the paper's LLM multi-objective rank)."""
    return float(sr) - float(cr) - 0.5 * float(tr)


def navigation_scalar_from_dict(metrics: Optional[Mapping[str, Any]]) -> float:
    if not metrics:
        return float("-inf")
    # Highway trainers stash official fitness (alias: selection_scalar).
    for key in ("fitness", "selection_scalar"):
        if metrics.get(key) is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                pass
    if str(metrics.get("domain", "")).strip().lower() == "highway":
        from domains.highway.metrics import highway_fitness

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
    """Official fitness for ranking (highway: holdout fitness; CrowdNav: nav scalar)."""
    return candidate_nav_scalar(candidate)



def pick_best_trained(
    snapshots: Sequence[RewardCandidate],
) -> Optional[RewardCandidate]:
    """Return the trained snapshot with highest navigation scalar, if any."""
    scored = [c for c in snapshots if candidate_nav_scalar(c) > float("-inf")]
    if not scored:
        return None
    return max(scored, key=candidate_nav_scalar)


def is_same_genome(a: RewardCandidate, b: RewardCandidate) -> bool:
    return a.code.strip() == b.code.strip()

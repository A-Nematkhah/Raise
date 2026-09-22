"""
Navigation fitness helpers for Stage II / III selection.

Paper (Alg. 1): final rankings R2 / R3 use LLM evaluation of multi-objective
M(r). Pipeline default is ``final_rank=llm`` (see ``ranking.py``).

This module's scalar is still used for *within-round* best-ever tracking and
optional engineering elitism:

    SR - CR - 0.5 * TR

NT, PL, ITR, and SD are logged on ProxyMetrics but are not part of this scalar.
Use ``--final-rank scalar`` only when you intentionally want this engineering
order instead of paper R2/R3.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from crowd_nav.reward_search.explore import RewardCandidate


def navigation_scalar(sr: float, cr: float, tr: float) -> float:
    """Baseline elite scalar (not the paper's LLM multi-objective rank)."""
    return float(sr) - float(cr) - 0.5 * float(tr)


def navigation_scalar_from_dict(metrics: Optional[Mapping[str, Any]]) -> float:
    if not metrics:
        return float("-inf")
    sr = float(metrics.get("SR", metrics.get("sr", 0.0)))
    cr = float(metrics.get("CR", metrics.get("cr", 0.0)))
    tr = float(metrics.get("TR", metrics.get("tr", 0.0)))
    return navigation_scalar(sr, cr, tr)


def candidate_nav_scalar(candidate: RewardCandidate) -> float:
    md = candidate.metadata or {}
    return navigation_scalar_from_dict(md.get("last_metrics"))


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

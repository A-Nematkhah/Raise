"""
EvoNav Stage I analytical rules (Figure 3).

Six preference categories (best → worst):
  Success + short nav length
  Success + long nav length
  Other (timeout) + short dist-to-goal
  Other + long dist-to-goal
  Fail (collision) + short dist-to-goal
  Fail + long dist-to-goal

Rule ranks prioritize Success > Other > Fail; tie-breaks use nav length
(for Success) or distance-to-goal at the ranked frame (for Other/Fail).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Sequence, Tuple

import numpy as np


class TrajectoryCategory(IntEnum):
    """Final episode label mapped onto Figure 3 buckets."""

    SUCCESS = 0  # ReachGoal
    OTHER = 1  # Timeout / incomplete
    FAIL = 2  # Collision


def label_to_category(label: str) -> TrajectoryCategory:
    key = str(label).strip().lower()
    if key in ("success", "reachgoal", "reach_goal", "goal"):
        return TrajectoryCategory.SUCCESS
    if key in ("fail", "collision", "collide"):
        return TrajectoryCategory.FAIL
    if key in ("other", "timeout", "time_out", "truncation"):
        return TrajectoryCategory.OTHER
    raise ValueError(f"Unknown trajectory label: {label!r}")


def dist_to_goal(px: float, py: float, gx: float, gy: float) -> float:
    return float(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)


def rule_preference_score(
    category: TrajectoryCategory,
    *,
    nav_length: float,
    dist_goal: float,
) -> float:
    """
    Higher = preferred by analytical rules (aligned with cumulative reward rank).

    Category dominates; within Success shorter nav_length is better; within
    Other/Fail smaller dist_goal is better.
    """
    base = {
        TrajectoryCategory.SUCCESS: 1e6,
        TrajectoryCategory.OTHER: 0.0,
        TrajectoryCategory.FAIL: -1e6,
    }[category]
    if category == TrajectoryCategory.SUCCESS:
        return float(base - float(nav_length))
    return float(base - float(dist_goal))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """
    Average-rank transform (scipy.stats.rankdata method='average'), NumPy only.

    Ranks are 1..n (higher value → higher rank). Ties share the mean rank.
    """
    a = np.asarray(list(values), dtype=np.float64).reshape(-1)
    n = a.size
    if n == 0:
        return a
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Spearman ρ via Pearson on average ranks (no scipy).

    Returns NaN if n < 2 or either side is constant (undefined correlation).
    """
    a = np.asarray(list(x), dtype=np.float64).reshape(-1)
    b = np.asarray(list(y), dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("x and y must have the same shape")
    if a.size < 2:
        return float("nan")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def pairwise_rule_scores(
    categories: Sequence[TrajectoryCategory],
    nav_lengths: Sequence[float],
    dist_goals: Sequence[float],
) -> Tuple[float, ...]:
    return tuple(
        rule_preference_score(cat, nav_length=nav, dist_goal=dist)
        for cat, nav, dist in zip(categories, nav_lengths, dist_goals)
    )

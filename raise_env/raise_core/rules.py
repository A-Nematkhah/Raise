"""
Stage I analytical rule helpers (episode categories + Spearman).

Preference order: Success > Other (timeout) > Fail (collision); the domain
pack supplies within-category tie-breaks.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Sequence

import numpy as np


class TrajectoryCategory(IntEnum):
    """Final episode label bucket."""

    SUCCESS = 0
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

"""Aggregate fitness across independent PPO train seeds (Stage II / III)."""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from raise_core.stats import fitness_standard_error

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_seed_fits",
    "median_seed_checkpoint",
    "partial_seeds_flag",
    "MIN_SEEDS_FOR_ACROSS_SEEDS_SE",
]

# Cross-seed SE needs enough independent seeds to be meaningful.
MIN_SEEDS_FOR_ACROSS_SEEDS_SE: int = 3


def aggregate_seed_fits(
    ok_fits: Sequence[float],
    *,
    episode_se: float = 0.0,
) -> Dict[str, Any]:
    """Mean fitness and SE across successful seeds.

    ``fitness_se_source`` is ``"across_seeds"`` only when at least
    ``MIN_SEEDS_FOR_ACROSS_SEEDS_SE`` seeds succeed. With 2 <= n < that
    threshold, SE is ``max(se_across, episode_se/sqrt(n))`` but the source
    stays ``"episodes_only"`` and a warning is logged. With a single seed the
    per-episode SE is kept.
    """
    fits = [float(x) for x in ok_fits]
    if not fits:
        return {
            "fitness": float("-inf"),
            "fitness_se": 0.0,
            "fitness_se_source": "episodes_only",
            "n_ok_seeds": 0,
        }
    mean_fit = float(statistics.fmean(fits))
    n = len(fits)
    ep_se = float(episode_se) if math.isfinite(float(episode_se)) else 0.0
    if n >= MIN_SEEDS_FOR_ACROSS_SEEDS_SE:
        se = float(fitness_standard_error(fits))
        source = "across_seeds"
    elif n > 1:
        se_across = float(fitness_standard_error(fits))
        se = max(se_across, ep_se / math.sqrt(n))
        source = "episodes_only"
        logger.warning(
            "only %d successful seed(s); keeping fitness_se_source=episodes_only "
            "(need >=%d for across_seeds); se=max(across=%.4g, episode/sqrt(K)=%.4g)",
            n,
            MIN_SEEDS_FOR_ACROSS_SEEDS_SE,
            se_across,
            ep_se / math.sqrt(n),
        )
    else:
        se = ep_se
        source = "episodes_only"
    return {
        "fitness": mean_fit,
        "fitness_se": se,
        "fitness_se_source": source,
        "n_ok_seeds": n,
        "seed_fits": list(fits),
    }


def median_seed_checkpoint(
    seed_runs: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[str], Optional[int]]:
    """Pick the checkpoint of the median-fitness successful seed (not the max).

    Returns ``(checkpoint_path, seed_index)``. With an even count of ok seeds,
    the **lower median** is used (index ``(n-1)//2`` after sorting by fitness).
    """
    ok = [
        r
        for r in seed_runs
        if r.get("status") == "ok" and r.get("checkpoint_path") is not None
    ]
    if not ok:
        ok = [r for r in seed_runs if r.get("status") == "ok"]
    if not ok:
        return None, None
    ordered = sorted(
        ok,
        key=lambda r: (
            float(r.get("fitness") or float("-inf")),
            int(r.get("seed_index") if r.get("seed_index") is not None else r.get("seed_slot") or 0),
        ),
    )
    mid = (len(ordered) - 1) // 2  # lower median for even n
    chosen = ordered[mid]
    path = chosen.get("checkpoint_path")
    idx = chosen.get("seed_index")
    if idx is None:
        idx = chosen.get("seed_slot")
    return (str(path) if path is not None else None, int(idx) if idx is not None else None)


def partial_seeds_flag(*, n_ok: int, n_seeds: int) -> bool:
    """True when fewer seeds succeeded than requested."""
    return int(n_ok) < max(1, int(n_seeds))

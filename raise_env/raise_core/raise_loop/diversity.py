"""Population diversity helpers for closed-loop next-generation ranking.

Highway-only: when many elites share an identical Stage-II metric fingerprint
(constant-cruise clones), demote duplicates so evolution explores alternatives.
CrowdNav paths never call this.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if v != v:  # NaN
        return float(default)
    return float(v)


def metrics_fingerprint(candidate: Any) -> Optional[Tuple[Any, ...]]:
    """
    Coarse metric signature for duplicate detection.

    Uses holdout metrics when present (selection profile). Returns None when
    the candidate has no Stage-II labels yet (Score1-only).
    """
    md = getattr(candidate, "metadata", None) or {}
    metrics = md.get("last_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    src = metrics.get("holdout") if isinstance(metrics.get("holdout"), dict) else metrics
    if not isinstance(src, dict):
        return None
    # Require at least one survival field so empty stubs don't collide.
    if "SR" not in src and "sr" not in src and "soft_success" not in src:
        return None
    sr = round(_finite(src.get("SR", src.get("sr", 0.0))), 3)
    cr = round(_finite(src.get("CR", src.get("cr", 0.0))), 3)
    soft = round(_finite(src.get("soft_success", 0.0)), 3)
    speed = round(_finite(src.get("mean_speed", src.get("ITR", 0.0))), 2)
    lc = round(_finite(src.get("lane_change_rate", 0.0)), 3)
    pl = round(_finite(src.get("PL", src.get("mean_progress", 0.0))), 0)
    return (sr, cr, soft, speed, lc, pl)


def diversify_ranking(
    ranked: Sequence[Any],
    *,
    keep_first_n_unique: int = 0,
) -> List[Any]:
    """
    Stable reorder: first occurrence of each fingerprint stays; later clones
    are deferred to the end (preserving relative order among deferred).

    Candidates without fingerprints (unlabeled) keep their relative position
    among the unique front — they are treated as always-unique.
    """
    del keep_first_n_unique  # reserved; uniqueness is per-fingerprint
    if not ranked:
        return []
    seen: set = set()
    unique: List[Any] = []
    dupes: List[Any] = []
    for c in ranked:
        fp = metrics_fingerprint(c)
        if fp is None:
            unique.append(c)
            continue
        if fp in seen:
            dupes.append(c)
        else:
            seen.add(fp)
            unique.append(c)
    return unique + dupes

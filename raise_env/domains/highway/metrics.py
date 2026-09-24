"""Highway evaluation metrics + selection scalar (progress/speed aware)."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


# Reference scales for ~40s highway-fast episodes at reasonable cruise.
_REF_PROGRESS_M = 800.0  # ~20 m/s * 40 s
_REF_SPEED_MPS = 25.0
_CRAWL_SPEED_MPS = 12.0  # below this while "surviving" is discouraged


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _tanh01(x: float) -> float:
    return float(math.tanh(max(0.0, float(x))))


def highway_navigation_scalar(metrics: Optional[Mapping[str, Any]]) -> float:
    """
    Highway elite / R2-R3 scalar (higher better).

    Base safety term matches CrowdNav form, then rewards forward progress and
    cruise speed so "crawl forever to avoid crashes" loses to competent driving::

        (SR - CR - 0.5·TR)
        + 0.35·tanh(PL / 800)
        + 0.25·tanh(mean_speed / 25)
        + 0.15·soft_success
        - crawl_penalty
    """
    if not metrics:
        return float("-inf")
    sr = _finite(metrics.get("SR", metrics.get("sr", 0.0)))
    cr = _finite(metrics.get("CR", metrics.get("cr", 0.0)))
    tr = _finite(metrics.get("TR", metrics.get("tr", 0.0)))
    pl = _finite(metrics.get("PL", metrics.get("mean_progress", 0.0)))
    speed = _finite(
        metrics.get("mean_speed", metrics.get("ITR", metrics.get("avg_speed", 0.0)))
    )
    soft = _finite(metrics.get("soft_success", 0.0))

    base = sr - cr - 0.5 * tr
    progress_term = 0.35 * _tanh01(pl / _REF_PROGRESS_M)
    speed_term = 0.25 * _tanh01(speed / _REF_SPEED_MPS)
    soft_term = 0.15 * soft

    crawl = 0.0
    # If the policy mostly survives but crawls, penalize explicitly.
    if sr >= 0.5 and speed < _CRAWL_SPEED_MPS:
        crawl = 0.40 * (_CRAWL_SPEED_MPS - speed) / _CRAWL_SPEED_MPS

    return float(base + progress_term + speed_term + soft_term - crawl)


def format_highway_metrics_line(metrics: Mapping[str, Any]) -> str:
    """One-line human summary for logs."""
    sr = _finite(metrics.get("SR"))
    cr = _finite(metrics.get("CR"))
    tr = _finite(metrics.get("TR"))
    pl = _finite(metrics.get("PL"))
    spd = _finite(metrics.get("mean_speed", metrics.get("ITR")))
    soft = _finite(metrics.get("soft_success"))
    lc = _finite(metrics.get("lane_change_rate"))
    sc = highway_navigation_scalar(metrics)
    return (
        f"SR={sr:.3f} CR={cr:.3f} TR={tr:.3f} | "
        f"progress={pl:.1f}m speed={spd:.1f}m/s soft={soft:.3f} "
        f"lane_chg={lc:.2f}/s | scalar={sc:.3f}"
    )


def attach_selection_scalar(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate/return metrics dict with domain tag + cached selection_scalar."""
    out = dict(metrics)
    out["domain"] = "highway"
    out["selection_scalar"] = float(highway_navigation_scalar(out))
    return out

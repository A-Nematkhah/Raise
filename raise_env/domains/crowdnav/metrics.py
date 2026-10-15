"""
CrowdNav evaluation metrics and the official selection fitness.

Episode statistics follow upstream ``crowd_nav/utils/explorer.py``
(``Explorer.run_k_episodes``):

* SR / CR / TR — success (``ReachGoal``), collision and timeout rates;
* NT — mean navigation time of successful episodes (``time_limit`` if none);
* total reward — mean over episodes of the upstream task reward discounted
  as ``gamma ** (t * time_step * v_pref)`` with ``gamma = 0.9``;
* ITR — danger frequency: steps whose upstream info is ``Danger`` divided by
  all steps (upstream "frequency of being in danger");
* SD — mean ``Danger.min_dist`` over those steps (upstream "average min
  separate distance in danger").

Fitness (higher is better) is that upstream total reward on the holdout
evaluation: ``+1`` per goal reached (discounted by arrival time), ``-0.25``
per collision and the discomfort penalties, exactly as CrowdNav reports its
RL objective. ``fitness_se`` is its standard error over episodes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "attach_fitness",
    "crowdnav_fitness",
    "discounted_task_return",
    "episode_fields",
    "evidence_line",
    "fitness_components",
    "fitness_standard_error",
    "format_crowdnav_metrics_line",
    "paired_difference",
]

OUTCOMES = ("success", "collision", "timeout")


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def discounted_task_return(
    native_rewards: Sequence[float], *, gamma: float, time_step: float, v_pref: float
) -> float:
    """``sum_t gamma ** (t * time_step * v_pref) * r_t`` (Explorer.run_k_episodes)."""
    return float(
        sum(
            math.pow(gamma, t * float(time_step) * float(v_pref)) * float(r)
            for t, r in enumerate(native_rewards)
        )
    )


def _metrics_for_fitness(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        return holdout
    return metrics


def crowdnav_fitness(metrics: Optional[Mapping[str, Any]]) -> float:
    """Mean discounted upstream task return of the (holdout) evaluation."""
    if not metrics:
        return float("-inf")
    src = _metrics_for_fitness(metrics)
    value = src.get("task_return")
    if value is None:
        return float("-inf")
    return _finite(value, float("-inf"))


def fitness_standard_error(task_returns: Sequence[float]) -> float:
    """Standard error of the mean task return (0.0 for fewer than two episodes)."""
    n = len(task_returns)
    if n < 2:
        return 0.0
    return float(np.std(np.asarray(task_returns, dtype=float), ddof=1) / math.sqrt(n))


def paired_difference(
    eps_a: Sequence[Mapping[str, Any]],
    eps_b: Sequence[Mapping[str, Any]],
    *,
    key: str = "task_return",
) -> Optional[Dict[str, float]]:
    """Mean per-case difference and its SE when both lists share the same cases.

    Episodes are matched by ``case`` (must be the same multiset). Returns
    ``None`` when cases differ or fewer than two pairs are available.
    """
    def _by_case(eps: Sequence[Mapping[str, Any]]) -> Optional[Dict[Any, float]]:
        out: Dict[Any, float] = {}
        for row in eps:
            if not isinstance(row, Mapping) or "case" not in row or key not in row:
                return None
            case = row["case"]
            if case in out:
                return None
            try:
                out[case] = float(row[key])
            except (TypeError, ValueError):
                return None
        return out

    map_a, map_b = _by_case(eps_a), _by_case(eps_b)
    if map_a is None or map_b is None:
        return None
    if set(map_a) != set(map_b) or len(map_a) < 2:
        return None
    diffs = [map_b[c] - map_a[c] for c in sorted(map_a, key=lambda x: (str(type(x)), x))]
    arr = np.asarray(diffs, dtype=float)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return {"mean": mean, "se": se, "n": float(len(arr))}


def episode_fields(
    outcomes: Sequence[str],
    nav_times: Sequence[float],
    task_returns: Sequence[float],
    danger_steps: Sequence[int],
    steps: Sequence[int],
    danger_min_dists: Sequence[float],
    *,
    time_limit: float,
) -> Dict[str, float]:
    """Aggregate per-episode records into the Explorer statistics."""
    n = max(1, len(outcomes))
    success_times = [t for o, t in zip(outcomes, nav_times) if o == "success"]
    total_steps = sum(int(s) for s in steps)
    return {
        "SR": sum(1 for o in outcomes if o == "success") / n,
        "CR": sum(1 for o in outcomes if o == "collision") / n,
        "TR": sum(1 for o in outcomes if o == "timeout") / n,
        "NT": float(np.mean(success_times)) if success_times else float(time_limit),
        "task_return": float(np.mean(task_returns)) if len(task_returns) else 0.0,
        "ITR": float(sum(int(d) for d in danger_steps)) / float(max(1, total_steps)),
        "SD": float(np.mean(danger_min_dists)) if len(danger_min_dists) else 0.0,
    }


def fitness_components(metrics: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Diagnostics for logs / feedback (not used by the selection hot path)."""
    if not metrics:
        return {"fitness": float("-inf")}
    src = _metrics_for_fitness(metrics)
    return {
        "fitness": crowdnav_fitness(metrics),
        "SR": _finite(src.get("SR")),
        "CR": _finite(src.get("CR")),
        "TR": _finite(src.get("TR")),
        "NT": _finite(src.get("NT")),
    }


def format_crowdnav_metrics_line(metrics: Mapping[str, Any]) -> str:
    """One-line human summary for logs (fitness is the official objective)."""
    src: Mapping[str, Any] = metrics
    tag = ""
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        src = holdout
        tag = "holdout "
    se = src.get("fitness_se")
    se_txt = f"±{_finite(se):.3f}" if se is not None else ""
    counts = ""
    if src.get("n_eval_episodes") is not None:
        counts = (
            f" ok={int(_finite(src.get('n_success')))}/{int(_finite(src.get('n_eval_episodes')))}"
            f" col={int(_finite(src.get('n_collision')))} to={int(_finite(src.get('n_timeout')))}"
        )
    return (
        f"{tag}SR={_finite(src.get('SR')):.2f} CR={_finite(src.get('CR')):.2f} "
        f"TR={_finite(src.get('TR')):.2f} nav_time={_finite(src.get('NT')):.2f}s "
        f"path={_finite(src.get('PL')):.2f}m danger_freq={_finite(src.get('ITR')):.3f} "
        f"min_dist_in_danger={_finite(src.get('SD')):.3f}m "
        f"fitness={crowdnav_fitness(metrics):.3f}{se_txt}{counts}"
    )


def evidence_line(metrics: Mapping[str, Any]) -> str:
    """Holdout metrics for the D.2 mutation evidence block (no fitness; the caller adds it)."""
    src: Mapping[str, Any] = metrics
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        src = holdout
    n_ep = src.get("n_eval_episodes", metrics.get("n_eval_episodes", "?"))
    try:
        n_ep_s = str(int(float(n_ep)))
    except (TypeError, ValueError):
        n_ep_s = str(n_ep)
    return (
        f"Holdout eval ({n_ep_s} episodes): "
        f"SR={_finite(src.get('SR')):.2f} CR={_finite(src.get('CR')):.2f} "
        f"TR={_finite(src.get('TR')):.2f} nav_time={_finite(src.get('NT')):.2f}s "
        f"path_length={_finite(src.get('PL')):.2f}m danger_freq={_finite(src.get('ITR')):.3f} "
        f"min_dist_in_danger={_finite(src.get('SD')):.3f}m"
    )


def attach_fitness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Cache the official ``fitness`` (alias ``selection_scalar``) on a metrics dict."""
    out = dict(metrics)
    out["domain"] = "crowdnav"
    value = float(crowdnav_fitness(out))
    out["fitness"] = value
    out["selection_scalar"] = value
    return out

"""CrowdNav++ evaluation metrics and the official outcome-score fitness.

Episode statistics follow CrowdNav++ upstream ``rl/evaluation.py``
(``evaluate``):

* SR / CR / TR — success (``ReachGoal``), collision and timeout rates;
* NT — mean navigation time of successful episodes (``time_limit`` if none);
* task return — mean per-episode sum of native rewards (diagnostic only);
* ITR — danger frequency: steps whose upstream info is ``Danger`` divided by
  all steps (upstream "frequency of being in danger");
* SD — mean ``Danger.min_dist`` over those steps (upstream "average min
  separate distance in danger").

Fitness (higher is better) is the mean outcome score on held-out validation
episodes: +1 for success, -1 for collision, and -0.5 for timeout.
``fitness_se`` is its standard error over per-episode outcome scores.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

# Domain-agnostic implementations live in raise_core; re-export for existing imports.
from raise_core.stats import fitness_standard_error, paired_difference

__all__ = [
    "attach_fitness",
    "crowdnav_fitness",
    "discounted_task_return",
    "episode_task_return",
    "episode_fields",
    "evidence_line",
    "fitness_components",
    "fitness_standard_error",
    "format_crowdnav_metrics_line",
    "paired_difference",
]

OUTCOME_SCORES = {"success": 1.0, "collision": -1.0, "timeout": -0.5}


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def discounted_task_return(
    native_rewards: Sequence[float], *, gamma: float, time_step: float, v_pref: float
) -> float:
    """Generic time-scaled discounted sum; not the CrowdNav++ fitness."""
    return float(
        sum(
            math.pow(gamma, t * float(time_step) * float(v_pref)) * float(r)
            for t, r in enumerate(native_rewards)
        )
    )


def episode_task_return(native_rewards: Sequence[float]) -> float:
    """CrowdNav++ evaluator's per-episode task return: the sum of native rewards."""
    return float(sum(float(reward) for reward in native_rewards))


def episode_selection_score(outcome: str) -> float:
    """Score a terminal outcome according to the CrowdNav selection objective."""
    try:
        return OUTCOME_SCORES[str(outcome)]
    except KeyError as exc:
        raise ValueError(f"Unsupported CrowdNav episode outcome: {outcome!r}") from exc


def _metrics_for_fitness(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        return holdout
    return metrics


def crowdnav_fitness(metrics: Optional[Mapping[str, Any]]) -> float:
    """Mean outcome score on the (holdout) evaluation."""
    if not metrics:
        return float("-inf")
    src = _metrics_for_fitness(metrics)
    value = src.get("selection_score")
    if value is None and all(key in src for key in ("SR", "CR", "TR")):
        value = (
            _finite(src.get("SR"))
            - _finite(src.get("CR"))
            - 0.5 * _finite(src.get("TR"))
        )
    if value is None:
        return float("-inf")
    return _finite(value, float("-inf"))


def episode_fields(
    outcomes: Sequence[str],
    nav_times: Sequence[float],
    task_returns: Sequence[float],
    selection_scores: Sequence[float],
    danger_steps: Sequence[int],
    steps: Sequence[int],
    danger_min_dists: Sequence[float],
    *,
    time_limit: float,
) -> Dict[str, float]:
    """Aggregate per-episode records into CrowdNav++ evaluator statistics."""
    n = max(1, len(outcomes))
    success_times = [t for o, t in zip(outcomes, nav_times) if o == "success"]
    total_steps = sum(int(s) for s in steps)
    return {
        "SR": sum(1 for o in outcomes if o == "success") / n,
        "CR": sum(1 for o in outcomes if o == "collision") / n,
        "TR": sum(1 for o in outcomes if o == "timeout") / n,
        "NT": float(np.mean(success_times)) if success_times else float(time_limit),
        "task_return": float(np.mean(task_returns)) if len(task_returns) else 0.0,
        "selection_score": (
            float(np.mean(selection_scores)) if len(selection_scores) else float("-inf")
        ),
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
        f"TR={_finite(src.get('TR')):.2f} nav_time={_nav_time_text(src)} "
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
        f"TR={_finite(src.get('TR')):.2f} nav_time={_nav_time_text(src)} "
        f"path_length={_finite(src.get('PL')):.2f}m danger_freq={_finite(src.get('ITR')):.3f} "
        f"min_dist_in_danger={_finite(src.get('SD')):.3f}m"
    )


def _nav_time_text(metrics: Mapping[str, Any]) -> str:
    if _finite(metrics.get("SR")) <= 0.0:
        return "n/a"
    return f"{_finite(metrics.get('NT')):.2f}s"


def attach_fitness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Cache the official ``fitness`` (alias ``selection_scalar``) on a metrics dict."""
    out = dict(metrics)
    out["domain"] = "crowdnav"
    value = float(crowdnav_fitness(out))
    out["fitness"] = value
    out["selection_scalar"] = value
    return out

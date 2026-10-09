"""Attach Stage-II evaluation metrics to D.2 mutation prompts inside the RAISE loop.

Holdout numbers plus the selection fitness — the LLM diagnoses from the evidence.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from raise_core.explore import RewardCandidate


def _f(metrics: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                continue
    return float(default)


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and abs(v) != float("inf")


def _fmt_opt(src: Mapping[str, Any], metrics: Mapping[str, Any], key: str, spec: str) -> str:
    raw = src.get(key, metrics.get(key, "n/a"))
    if raw is None or raw == "n/a":
        return "n/a"
    try:
        return format(float(raw), spec)
    except (TypeError, ValueError):
        return str(raw)


def evidence_block(
    metrics: Mapping[str, Any],
    *,
    score1: Optional[float] = None,
) -> str:
    """Holdout evaluation metrics (plus fitness / Score1 when finite) for the LLM."""
    src = metrics
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        src = holdout
    n_ep = src.get("n_eval_episodes", metrics.get("n_eval_episodes", "?"))
    try:
        n_ep_s = str(int(float(n_ep)))
    except (TypeError, ValueError):
        n_ep_s = str(n_ep)
    progress = _f(src, "mean_progress", "PL", default=_f(metrics, "mean_progress", "PL"))
    lines = [
        (
            f"Holdout eval ({n_ep_s} episodes): "
            f"SR={_f(src, 'SR', 'sr'):.2f} CR={_f(src, 'CR', 'cr'):.2f} "
            f"TR={_f(src, 'TR', 'tr'):.2f} mean_speed={_f(src, 'mean_speed', 'ITR'):.1f}m/s "
            f"speed_p10={_fmt_opt(src, metrics, 'speed_p10', '.1f')} "
            f"progress={progress:.0f}m "
            f"lane_change_rate={_fmt_opt(src, metrics, 'lane_change_rate', '.3f')} "
            f"overtakes_per_km={_fmt_opt(src, metrics, 'overtakes_per_km', '.2f')} "
            f"overtake_episode_frac={_fmt_opt(src, metrics, 'overtake_episode_frac', '.2f')}"
        ),
    ]
    if _finite(metrics.get("fitness")):
        lines.append(f"fitness={float(metrics['fitness']):.3f}")
    if score1 is not None and _finite(score1):
        lines.append(f"Score1={float(score1):.3f}")
    return "\n".join(lines)


def attach_proxy_feedback(candidate: RewardCandidate, metrics: Mapping[str, Any]) -> None:
    """Write ``last_metrics`` and the ``proxy_feedback`` evidence onto the candidate."""
    md = dict(candidate.metadata or {})
    md["last_metrics"] = dict(metrics)
    score1 = candidate.score
    if score1 is None and md.get("score1_train") is not None:
        try:
            score1 = float(md["score1_train"])
        except (TypeError, ValueError):
            score1 = None
    score1_f = float(score1) if score1 is not None and _finite(score1) else None
    md["proxy_feedback"] = evidence_block(metrics, score1=score1_f)
    candidate.metadata = md

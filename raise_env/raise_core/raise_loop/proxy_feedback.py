"""Attach Stage-II evaluation metrics to D.2 mutation prompts inside the RAISE loop.

Holdout numbers plus the selection fitness; the LLM diagnoses from the evidence.
The domain pack formats its own metrics (``DomainPack.evidence_line``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

from raise_core.candidate import RewardCandidate

EvidenceLine = Callable[[Mapping[str, Any]], str]


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


def generic_evidence_line(metrics: Mapping[str, Any]) -> str:
    """Outcome rates of the holdout eval (domains without their own formatter)."""
    src = metrics
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
        f"SR={_f(src, 'SR', 'sr'):.2f} CR={_f(src, 'CR', 'cr'):.2f} "
        f"TR={_f(src, 'TR', 'tr'):.2f}"
    )


def evidence_block(
    metrics: Mapping[str, Any], *, evidence_line: Optional[EvidenceLine] = None
) -> str:
    """Holdout evaluation metrics (plus fitness when finite) for the LLM.

    Score1 is deliberately absent: it is a sanity filter with no shown relation
    to fitness, so the LLM must not optimize it.
    """
    line_fn = evidence_line or generic_evidence_line
    lines = [line_fn(metrics)]
    if _finite(metrics.get("fitness")):
        lines.append(f"fitness={float(metrics['fitness']):.3f}")
    return "\n".join(lines)


def attach_proxy_feedback(
    candidate: RewardCandidate,
    metrics: Mapping[str, Any],
    *,
    evidence_line: Optional[EvidenceLine] = None,
) -> None:
    """Write ``last_metrics`` and the ``proxy_feedback`` evidence onto the candidate."""
    md = dict(candidate.metadata or {})
    md["last_metrics"] = dict(metrics)
    md["proxy_feedback"] = evidence_block(metrics, evidence_line=evidence_line)
    candidate.metadata = md

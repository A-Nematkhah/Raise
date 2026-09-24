"""Attach Stage-II proxy metrics to LLM prompts inside the RAISE loop.

Score1 remains the cheap evolutionary signal; SR/CR/TR feedback is selective
so short-horizon noise does not dominate every mutation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from crowd_nav.reward_search.explore import RewardCandidate


def nav_scalar(metrics: Mapping[str, Any]) -> float:
    """SR - CR - 0.5 * TR (same spirit as Refine ProxyMetrics.scalar_score)."""
    sr = float(metrics.get("SR") or 0.0)
    cr = float(metrics.get("CR") or 0.0)
    tr = float(metrics.get("TR") or 0.0)
    return float(sr - cr - 0.5 * tr)


def focus_note_from_metrics(metrics: Mapping[str, Any]) -> str:
    """One-line editing hint for mutation / D.3."""
    sr = float(metrics.get("SR") or 0.0)
    cr = float(metrics.get("CR") or 0.0)
    tr = float(metrics.get("TR") or 0.0)
    if tr >= 0.5 and tr >= cr:
        return (
            "high timeout — strengthen dense goal progress / time pressure; "
            "keep collision penalties"
        )
    if cr >= 0.5:
        return (
            "high collision — strengthen proximity/collision penalties and "
            "clearance shaping; keep goal progress"
        )
    if sr < 0.10:
        return (
            "near-zero success — increase goal-seeking reward density; "
            "avoid near-constant returns"
        )
    return (
        "improve success vs collision/timeout trade-off; keep dense finite shaping"
    )


def format_proxy_feedback_block(
    metrics: Mapping[str, Any],
    *,
    score1: Optional[float] = None,
) -> str:
    """Compact block embedded in mutation weakness / D.3 feedback."""
    sr = float(metrics.get("SR") or 0.0)
    cr = float(metrics.get("CR") or 0.0)
    tr = float(metrics.get("TR") or 0.0)
    sc = nav_scalar(metrics)
    s1_bit = ""
    if score1 is not None and _finite(score1):
        s1_bit = f" | Score1={float(score1):.3f}"
    focus = focus_note_from_metrics(metrics)
    return (
        f"ProxyRefine: SR={sr:.2f} CR={cr:.2f} TR={tr:.2f} "
        f"scalar={sc:.2f}{s1_bit}\nFocus: {focus}"
    )


def should_attach_proxy_feedback(
    metrics: Mapping[str, Any],
    *,
    score1: Optional[float] = None,
    population_score1: Optional[Sequence[float]] = None,
    population_scalars: Optional[Sequence[float]] = None,
) -> bool:
    """
    Attach when proxy looks bad, or Score1 looks strong while proxy scalar is weak.
    """
    sr = float(metrics.get("SR") or 0.0)
    cr = float(metrics.get("CR") or 0.0)
    tr = float(metrics.get("TR") or 0.0)
    if sr < 0.10 or tr >= 0.50 or cr >= 0.50:
        return True
    if (
        score1 is not None
        and _finite(score1)
        and population_score1
        and population_scalars
        and len(population_score1) >= 4
        and len(population_scalars) >= 4
    ):
        sc = nav_scalar(metrics)
        s1_q75 = _quantile(list(population_score1), 0.75)
        sc_q25 = _quantile(list(population_scalars), 0.25)
        if float(score1) >= s1_q75 and sc <= sc_q25:
            return True
    return False


def attach_proxy_feedback(
    candidate: RewardCandidate,
    metrics: Mapping[str, Any],
    *,
    enabled: bool,
    n_labeled_dataset: int,
    min_labels: int,
    epoch: int,
    population_score1: Optional[Sequence[float]] = None,
    population_scalars: Optional[Sequence[float]] = None,
) -> bool:
    """
    Write ``proxy_feedback`` (+ related keys) onto ``candidate.metadata``.

    Always stores ``last_metrics`` if missing. Returns True when a prompt-facing
    ``proxy_feedback`` block was attached.
    """
    md = dict(candidate.metadata or {})
    md["last_metrics"] = dict(metrics)
    score1 = candidate.score
    if score1 is None and md.get("score1_train") is not None:
        try:
            score1 = float(md["score1_train"])
        except (TypeError, ValueError):
            score1 = None

    attach = bool(enabled) and int(n_labeled_dataset) >= int(min_labels) and int(epoch) >= 1
    if attach and should_attach_proxy_feedback(
        metrics,
        score1=float(score1) if score1 is not None and _finite(score1) else None,
        population_score1=population_score1,
        population_scalars=population_scalars,
    ):
        block = format_proxy_feedback_block(
            metrics,
            score1=float(score1) if score1 is not None and _finite(score1) else None,
        )
        md["proxy_feedback"] = block
        md["proxy_nav_scalar"] = nav_scalar(metrics)
        md["proxy_feedback_focus"] = focus_note_from_metrics(metrics)
        candidate.metadata = md
        return True

    # Keep stale prompt block from confusing a later good label.
    md.pop("proxy_feedback", None)
    md.pop("proxy_feedback_focus", None)
    if "SR" in metrics:
        md["proxy_nav_scalar"] = nav_scalar(metrics)
    candidate.metadata = md
    return False


def proxy_summary_for_reflection(candidates: Sequence[RewardCandidate]) -> Optional[str]:
    """One-line population summary for the global reflection string."""
    rows: List[Tuple[str, float, float, float, float]] = []
    for c in candidates:
        md = c.metadata or {}
        m = md.get("last_metrics")
        if not isinstance(m, dict):
            continue
        rows.append(
            (
                str(c.candidate_id),
                float(m.get("SR") or 0.0),
                float(m.get("CR") or 0.0),
                float(m.get("TR") or 0.0),
                nav_scalar(m),
            )
        )
    if not rows:
        return None
    best = max(rows, key=lambda t: t[4])
    mean_sr = sum(t[1] for t in rows) / len(rows)
    mean_sc = sum(t[4] for t in rows) / len(rows)
    return (
        f"ProxyRefine summary (n={len(rows)}): mean_SR={mean_sr:.2f} "
        f"mean_scalar={mean_sc:.2f}; best={best[0]} "
        f"SR={best[1]:.2f} CR={best[2]:.2f} TR={best[3]:.2f} scalar={best[4]:.2f}"
    )


def select_for_in_loop_d3(
    candidates: Sequence[RewardCandidate],
    *,
    max_n: int,
) -> List[RewardCandidate]:
    """Pick worst proxy scalars that already have metrics (and preferably feedback)."""
    if max_n <= 0:
        return []
    scored: List[Tuple[float, RewardCandidate]] = []
    for c in candidates:
        md = c.metadata or {}
        m = md.get("last_metrics")
        if not isinstance(m, dict):
            continue
        scored.append((nav_scalar(m), c))
    scored.sort(key=lambda t: t[0])  # worst first
    out: List[RewardCandidate] = []
    seen = set()
    for _, c in scored:
        cid = str(c.candidate_id)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
        if len(out) >= int(max_n):
            break
    return out


def apply_in_loop_d3(
    candidate: RewardCandidate,
    *,
    llm: Any,
    validator: Any,
) -> RewardCandidate:
    """
    One D.3 rewrite using proxy metrics on the candidate.

    On failure, returns the original candidate unchanged.
    """
    from dataclasses import replace

    from crowd_nav.reward_search.llm import extract_python_code, normalize_to_compute_reward
    from crowd_nav.reward_search.prompts import D3_SYSTEM_PROMPT, format_d3_refinement

    md = dict(candidate.metadata or {})
    metrics = md.get("last_metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return candidate
    feedback = md.get("proxy_feedback") or format_proxy_feedback_block(
        metrics,
        score1=float(candidate.score) if candidate.score is not None else None,
    )
    last_score = float(md.get("proxy_nav_scalar") or nav_scalar(metrics))
    user_prompt = format_d3_refinement(
        candidate.code,
        last_score=last_score,
        feedback=str(feedback),
        extra_context_if_any="In-loop proxy refine (short Stage II metrics).",
    )
    full_prompt = f"{D3_SYSTEM_PROMPT}\n\n{user_prompt}"
    try:
        raw = llm.complete(full_prompt)
        new_code = normalize_to_compute_reward(extract_python_code(raw))
    except Exception:  # noqa: BLE001
        return candidate
    reward_fn, err = validator.try_validate(new_code)
    if reward_fn is None:
        return candidate
    new_md = dict(md)
    new_md["in_loop_d3"] = True
    new_md["in_loop_d3_parent"] = candidate.candidate_id
    return replace(
        candidate,
        candidate_id=f"{candidate.candidate_id}_d3",
        code=new_code,
        reward_fn=reward_fn,
        valid=True,
        origin="in_loop_d3",
        parent_ids=(candidate.candidate_id,),
        metadata=new_md,
        score=candidate.score,
    )


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v and abs(v) != float("inf")


def _quantile(vals: List[float], q: float) -> float:
    xs = sorted(float(v) for v in vals if _finite(v))
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    q = min(1.0, max(0.0, float(q)))
    idx = int(round(q * (len(xs) - 1)))
    return xs[idx]

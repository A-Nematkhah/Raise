"""Attach Stage-II proxy metrics to LLM prompts inside the RAISE loop.

Highway: always attach raw holdout numbers + Pareto metadata when stamped —
never the legacy ``highway_fitness`` scalar. CrowdNav: selective attach + focus notes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from raise_core.explore import RewardCandidate
from raise_core.selection import is_highway_metrics


def nav_scalar(metrics: Mapping[str, Any]) -> float:
    """Selection scalar: highway-aware when continuous fields are present."""
    from raise_core.selection import navigation_scalar_from_dict

    return float(navigation_scalar_from_dict(metrics))


def _f(metrics: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                continue
    return float(default)


def _is_highway_metrics(metrics: Mapping[str, Any]) -> bool:
    """Private alias — prefer ``raise_core.selection.is_highway_metrics``."""
    return is_highway_metrics(metrics)


def evidence_block(
    metrics: Mapping[str, Any],
    *,
    score1: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Holdout metrics + Pareto selection evidence (no legacy fitness/gate)."""
    src = metrics
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        src = holdout
    md = metadata or {}
    p10 = src.get("speed_p10", metrics.get("speed_p10", "n/a"))
    if p10 is not None and p10 != "n/a":
        try:
            p10 = f"{float(p10):.1f}"
        except (TypeError, ValueError):
            p10 = str(p10)
    lc = src.get("lane_change_rate", metrics.get("lane_change_rate", "n/a"))
    if lc is not None and lc != "n/a":
        try:
            lc = f"{float(lc):.3f}"
        except (TypeError, ValueError):
            lc = str(lc)
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
            f"speed_p10={p10} "
            f"progress={progress:.0f}m "
            f"soft_success={_f(src, 'soft_success'):.2f} "
            f"lane_change_rate={lc}"
        ),
    ]
    if md.get("pareto_rank") is not None:
        n = int(md.get("pareto_n") or 0) or "?"
        feasible = md.get("pareto_feasible")
        feas_s = (
            f"feasible={bool(feasible)}"
            if feasible is not None
            else "feasible=unknown"
        )
        front = md.get("pareto_front")
        front_s = (
            f"front={int(front)}"
            if front is not None
            else "front=n/a"
        )
        lines.append(
            f"Pareto rank: {int(md['pareto_rank'])}/{n} "
            f"(front-relative; lower is better; {feas_s}; {front_s})"
        )
        if md.get("pareto_v_floor") is not None:
            lines.append(
                "Auto-calibrated feasibility this run: "
                f"min_speed≈{float(md['pareto_v_floor']):.1f}m/s "
                "(measured from the environment's own traffic, not a fixed target), "
                f"max_collision_rate≈{float(md.get('pareto_cr_ceiling', float('nan'))):.2f}, "
                f"max_offroad_rate≈{float(md.get('pareto_tr_ceiling', float('nan'))):.2f}"
            )
    if score1 is not None and _finite(score1):
        lines.append(f"Score1={float(score1):.3f}")
    return "\n".join(lines)


def refresh_highway_evidence_after_pareto(
    candidates: Sequence[RewardCandidate],
) -> None:
    """Re-render ``proxy_feedback`` once ``pareto_*`` keys are stamped."""
    for c in candidates:
        md = dict(getattr(c, "metadata", None) or {})
        metrics = md.get("last_metrics")
        if not isinstance(metrics, dict) or not metrics:
            continue
        if not is_highway_metrics(metrics):
            continue
        score1 = getattr(c, "score", None)
        if score1 is None and md.get("score1_train") is not None:
            try:
                score1 = float(md["score1_train"])
            except (TypeError, ValueError):
                score1 = None
        score1_f = float(score1) if score1 is not None and _finite(score1) else None
        md["proxy_feedback"] = evidence_block(
            metrics, score1=score1_f, metadata=md
        )
        c.metadata = md


def focus_note_from_metrics(metrics: Mapping[str, Any]) -> str:
    """CrowdNav-only editing hint (highway uses evidence_block instead)."""
    if is_highway_metrics(metrics):
        # Highway path never interprets; keep a neutral stub for legacy callers.
        return evidence_block(metrics)

    sr = _f(metrics, "SR", "sr")
    cr = _f(metrics, "CR", "cr")
    tr = _f(metrics, "TR", "tr")

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
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Compact block embedded in mutation weakness / D.3 feedback."""
    if is_highway_metrics(metrics):
        return evidence_block(metrics, score1=score1, metadata=metadata)

    sr = _f(metrics, "SR", "sr")
    cr = _f(metrics, "CR", "cr")
    tr = _f(metrics, "TR", "tr")
    sc = nav_scalar(metrics)
    s1_bit = ""
    if score1 is not None and _finite(score1):
        s1_bit = f" | Score1={float(score1):.3f}"
    focus = focus_note_from_metrics(metrics)
    return (
        f"ProxyRefine: SR={sr:.2f} CR={cr:.2f} TR={tr:.2f} "
        f"scalar={sc:.2f}{s1_bit}\nFocus: {focus}"
    )


def should_attach_crowdnav_proxy_feedback(
    metrics: Mapping[str, Any],
    *,
    score1: Optional[float] = None,
    population_score1: Optional[Sequence[float]] = None,
    population_scalars: Optional[Sequence[float]] = None,
) -> bool:
    """
    CrowdNav-only gate for attaching proxy focus notes.

    Highway never calls this: ``attach_proxy_feedback`` always attaches
    ``evidence_block`` when enabled for highway metrics.
    """
    if is_highway_metrics(metrics):
        return True  # unused on highway attach path; keep legacy True

    sr = _f(metrics, "SR", "sr")
    cr = _f(metrics, "CR", "cr")
    tr = _f(metrics, "TR", "tr")
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


# Legacy name — CrowdNav callers / tests.
should_attach_proxy_feedback = should_attach_crowdnav_proxy_feedback


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

    Highway: when enabled, always attach ``evidence_block`` (no threshold gate).
    CrowdNav: selective attach after epoch/label gates (unchanged).
    """
    md = dict(candidate.metadata or {})
    md["last_metrics"] = dict(metrics)
    score1 = candidate.score
    if score1 is None and md.get("score1_train") is not None:
        try:
            score1 = float(md["score1_train"])
        except (TypeError, ValueError):
            score1 = None
    score1_f = float(score1) if score1 is not None and _finite(score1) else None

    if is_highway_metrics(metrics):
        if not bool(enabled):
            md.pop("proxy_feedback", None)
            md.pop("proxy_feedback_focus", None)
            md.pop("proxy_hack_mode", None)
            candidate.metadata = md
            return False
        block = evidence_block(metrics, score1=score1_f, metadata=md)
        md["proxy_feedback"] = block
        md["proxy_nav_scalar"] = nav_scalar(metrics)
        md.pop("proxy_feedback_focus", None)
        md.pop("proxy_hack_mode", None)
        candidate.metadata = md
        return True

    attach = (
        bool(enabled)
        and int(n_labeled_dataset) >= int(min_labels)
        and int(epoch) >= 1
    )
    if attach and should_attach_proxy_feedback(
        metrics,
        score1=score1_f,
        population_score1=population_score1,
        population_scalars=population_scalars,
    ):
        block = format_proxy_feedback_block(
            metrics, score1=score1_f, metadata=md
        )
        md["proxy_feedback"] = block
        md["proxy_nav_scalar"] = nav_scalar(metrics)
        md["proxy_feedback_focus"] = focus_note_from_metrics(metrics)
        md.pop("proxy_hack_mode", None)
        candidate.metadata = md
        return True

    # Keep stale prompt block from confusing a later good label.
    md.pop("proxy_feedback", None)
    md.pop("proxy_feedback_focus", None)
    md.pop("proxy_hack_mode", None)
    if "SR" in metrics:
        md["proxy_nav_scalar"] = nav_scalar(metrics)
    candidate.metadata = md
    return False


def proxy_summary_for_reflection(candidates: Sequence[RewardCandidate]) -> Optional[str]:
    """One-line population summary for the global reflection string (CrowdNav)."""
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
    prompts: Any = None,
) -> RewardCandidate:
    """
    One D.3 rewrite using proxy metrics on the candidate.

    On failure, returns the original candidate unchanged.
    """
    from dataclasses import replace

    from raise_core.explore import extract_llm_diagnosis
    from raise_core.llm import extract_python_code, normalize_to_compute_reward

    if prompts is not None:
        D3_SYSTEM_PROMPT = getattr(prompts, "D3_SYSTEM_PROMPT", None)
        format_d3_refinement = getattr(prompts, "format_d3_refinement", None)
    else:
        D3_SYSTEM_PROMPT = None
        format_d3_refinement = None
    if D3_SYSTEM_PROMPT is None or format_d3_refinement is None:
        from domains.crowdnav.prompts import (
            D3_SYSTEM_PROMPT as _CN_D3,
            format_d3_refinement as _cn_fmt,
        )

        D3_SYSTEM_PROMPT = D3_SYSTEM_PROMPT or _CN_D3
        format_d3_refinement = format_d3_refinement or _cn_fmt

    md = dict(candidate.metadata or {})
    metrics = md.get("last_metrics") or {}
    if not isinstance(metrics, dict) or not metrics:
        return candidate
    feedback = md.get("proxy_feedback") or format_proxy_feedback_block(
        metrics,
        score1=float(candidate.score) if candidate.score is not None else None,
        metadata=md,
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
        diagnosis = extract_llm_diagnosis(raw)
    except Exception:  # noqa: BLE001
        return candidate
    reward_fn, err = validator.try_validate(new_code)
    if reward_fn is None:
        return candidate
    new_md = dict(md)
    new_md["in_loop_d3"] = True
    new_md["in_loop_d3_parent"] = candidate.candidate_id
    new_md["llm_diagnosis"] = diagnosis
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

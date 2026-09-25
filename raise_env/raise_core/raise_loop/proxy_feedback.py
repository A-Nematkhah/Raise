"""Attach Stage-II proxy metrics to LLM prompts inside the RAISE loop.

Score1 remains the cheap evolutionary signal; SR/CR/TR (+ highway continuous
fields) feedback is selective so short-horizon noise does not dominate every
mutation — but known *reward-hacking* fingerprints always surface so the LLM
sees what the policy actually did.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from raise_core.explore import RewardCandidate

# Align with domains.highway.metrics.V_TARGET (avoid hard import cycles in CrowdNav).
_HIGHWAY_TRAFFIC_MPS = 25.0
_SOFT_FLOOR_LEGACY_MPS = 20.0


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


def classify_highway_hack(metrics: Mapping[str, Any]) -> Optional[str]:
    """
    Detect named reward-hacking / fail modes from holdout Stage-II metrics.

    Returns a short mode id or None when behavior looks non-degenerate.
    """
    is_highway = str(metrics.get("domain", "")).lower() == "highway"
    has_speed = "mean_speed" in metrics or "ITR" in metrics
    if not (is_highway or has_speed):
        return None

    sr = _f(metrics, "SR", "sr")
    cr = _f(metrics, "CR", "cr")
    tr = _f(metrics, "TR", "tr")
    speed = _f(metrics, "mean_speed", "ITR")
    soft = _f(metrics, "soft_success") if "soft_success" in metrics else None
    lc = metrics.get("lane_change_rate")
    p10 = metrics.get("speed_p10")
    p90 = metrics.get("speed_p90")
    pstd = metrics.get("progress_std")
    spread = None
    if p10 is not None and p90 is not None:
        try:
            spread = abs(float(p90) - float(p10))
        except (TypeError, ValueError):
            spread = None

    # Crash-loving / unsafe speed shaping.
    if cr >= 0.5:
        return "crash_attract"

    # Constant cruise / soft-threshold lock (the soft@20 exploit fingerprint).
    flat_speed = spread is not None and spread < 0.5
    flat_progress = pstd is not None and float(pstd) < 1.0
    no_lane = lc is not None and float(lc) < 0.02
    near_soft_floor = (
        _SOFT_FLOOR_LEGACY_MPS - 1.0 <= speed <= _SOFT_FLOOR_LEGACY_MPS + 1.5
    )
    below_traffic = speed + 0.5 < _HIGHWAY_TRAFFIC_MPS
    if (
        sr >= 0.85
        and cr < 0.15
        and below_traffic
        and (flat_speed or near_soft_floor)
        and (no_lane or flat_progress or soft is not None and soft < 0.35)
    ):
        return "constant_cruise_hack"

    if soft is not None and sr >= 0.5 and soft < 0.25 and below_traffic:
        return "below_traffic_soft"

    if sr >= 0.5 and speed < 18.0:
        return "lag_traffic"

    if sr >= 0.5 and _f(metrics, "mean_progress", "PL") < 300.0:
        return "low_progress"

    if tr >= 0.5:
        return "offroad_timeout"

    return None


def focus_note_from_metrics(metrics: Mapping[str, Any]) -> str:
    """One-line editing hint for mutation / D.3 — name the hack when present."""
    mode = classify_highway_hack(metrics)
    sr = _f(metrics, "SR", "sr")
    cr = _f(metrics, "CR", "cr")
    tr = _f(metrics, "TR", "tr")
    speed = _f(metrics, "mean_speed", "ITR")
    progress = _f(metrics, "mean_progress", "PL")
    soft = _f(metrics, "soft_success") if "soft_success" in metrics else None

    if mode == "crash_attract":
        return (
            "REWARD HACKING / FAIL: crash attractor — PPO learned high speed then "
            "collides (CR high). Strengthen collision/off-road penalties relative "
            "to speed/progress; keep clearance shaping; still target ~25 m/s traffic "
            "when safe — do NOT only maximize speed"
        )
    if mode == "constant_cruise_hack":
        return (
            "REWARD HACKING DETECTED: constant-cruise exploit — policy locks near "
            f"~{speed:.1f} m/s with almost zero speed variance / lane changes / "
            "progress variance (soft-threshold gaming). Break the plateau: reward "
            f"matching traffic near {_HIGHWAY_TRAFFIC_MPS:.0f} m/s, penalize flat "
            "constant speed just above a soft floor, and do not treat SR=1 alone "
            "as success without throughput diversity"
        )
    if mode == "below_traffic_soft":
        return (
            "REWARD HACKING RISK: soft_success low while surviving — ego stays "
            f"below traffic (~{speed:.1f} m/s vs target ~{_HIGHWAY_TRAFFIC_MPS:.0f}). "
            "Raise shaping toward traffic cruise; soft_success needs survive AND "
            f">={_HIGHWAY_TRAFFIC_MPS:.0f} m/s AND meaningful progress"
        )
    if mode == "lag_traffic":
        return (
            "lags traffic — surrounding flow is ~20–30 m/s (target ~25); increase "
            "reward for matching that band and forward progress; penalize ego "
            "much slower while on-road"
        )
    if mode == "low_progress":
        return (
            "low forward progress — strengthen state.progress / speed terms; "
            "avoid rewarding idle/lane-hold forever"
        )
    if mode == "offroad_timeout":
        return (
            "high off-road/timeout share — keep on_road shaping; still reward "
            "forward progress at traffic speed (~25 m/s)"
        )

    if str(metrics.get("domain", "")).lower() == "highway" or "mean_speed" in metrics:
        if soft is not None and soft < 0.3 and sr >= 0.4:
            return (
                "soft_success low — survive AND match traffic (~>=25 m/s) with "
                "meaningful progress; balance safety with throughput"
            )
        return (
            "improve safe throughput: higher soft_success (survive+~25 m/s+"
            "progress) without raising CR; avoid constant-speed lane-keep hacks"
        )

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
    sr = _f(metrics, "SR", "sr")
    cr = _f(metrics, "CR", "cr")
    tr = _f(metrics, "TR", "tr")
    sc = nav_scalar(metrics)
    s1_bit = ""
    if score1 is not None and _finite(score1):
        s1_bit = f" | Score1={float(score1):.3f}"
    focus = focus_note_from_metrics(metrics)
    mode = classify_highway_hack(metrics)
    mode_bit = f" | mode={mode}" if mode else ""
    extra = ""
    if "mean_speed" in metrics or str(metrics.get("domain", "")).lower() == "highway":
        spd = _f(metrics, "mean_speed", "ITR")
        pl = _f(metrics, "mean_progress", "PL")
        soft = _f(metrics, "soft_success") if "soft_success" in metrics else 0.0
        lc = _f(metrics, "lane_change_rate") if "lane_change_rate" in metrics else None
        p10 = metrics.get("speed_p10")
        p90 = metrics.get("speed_p90")
        pstd = metrics.get("progress_std")
        extra = f" | speed={spd:.1f}m/s progress={pl:.0f}m soft={soft:.2f}"
        if lc is not None:
            extra += f" laneΔ={lc:.3f}"
        if p10 is not None and p90 is not None:
            try:
                extra += f" spd_band=[{float(p10):.1f},{float(p90):.1f}]"
            except (TypeError, ValueError):
                pass
        if pstd is not None:
            try:
                extra += f" prog_std={float(pstd):.2f}"
            except (TypeError, ValueError):
                pass
        fit = metrics.get("fitness", metrics.get("selection_scalar"))
        if fit is not None:
            try:
                extra += f" fitness={float(fit):.3f}"
            except (TypeError, ValueError):
                pass
    return (
        f"ProxyRefine: SR={sr:.2f} CR={cr:.2f} TR={tr:.2f} "
        f"scalar={sc:.2f}{extra}{s1_bit}{mode_bit}\nFocus: {focus}"
    )


def should_attach_proxy_feedback(
    metrics: Mapping[str, Any],
    *,
    score1: Optional[float] = None,
    population_score1: Optional[Sequence[float]] = None,
    population_scalars: Optional[Sequence[float]] = None,
) -> bool:
    """
    Attach when proxy looks bad, reward-hacking fingerprint is present, or
    Score1 looks strong while proxy scalar is weak.
    """
    sr = _f(metrics, "SR", "sr")
    cr = _f(metrics, "CR", "cr")
    tr = _f(metrics, "TR", "tr")
    if sr < 0.10 or tr >= 0.50 or cr >= 0.50:
        return True
    # Named highway hacks (including high-SR constant cruise) always surface.
    if classify_highway_hack(metrics) is not None:
        return True
    is_highway = str(metrics.get("domain", "")).lower() == "highway"
    has_speed = "mean_speed" in metrics or "ITR" in metrics
    if is_highway or has_speed:
        soft = (
            float(metrics["soft_success"])
            if "soft_success" in metrics and metrics.get("soft_success") is not None
            else None
        )
        # Clear lag only; mild shortfall vs 25 m/s is covered by classify_* /
        # soft_success — do not flag every 24 m/s survivor as a hack.
        speed = _f(metrics, "mean_speed", "ITR")
        lag_floor = 18.0 if is_highway else 12.0
        if sr >= 0.5 and speed < lag_floor:
            return True
        if soft is not None and sr >= 0.5 and soft < 0.25:
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
        mode = classify_highway_hack(metrics)
        if mode:
            md["proxy_hack_mode"] = mode
        else:
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
    prompts: Any = None,
) -> RewardCandidate:
    """
    One D.3 rewrite using proxy metrics on the candidate.

    On failure, returns the original candidate unchanged.
    """
    from dataclasses import replace

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

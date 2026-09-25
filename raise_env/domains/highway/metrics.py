"""Highway fitness (official selection objective) + anti-hacking terms.

Official objective for highway RAISE selection / evolution / Stage III elites:

    fitness = highway_fitness(holdout_metrics)

Score1 is only a cheap Stage-I proxy — not this fitness.
PPO optimizes the LLM ``compute_reward``; we keep genomes that raise fitness.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional


# Reference scales for ~40s highway-fast episodes at traffic cruise.
_REF_PROGRESS_M = 800.0  # ~20 m/s * 40 s
_REF_SPEED_MPS = 25.0

# Speed gate (relative to target cruise).
V_TARGET = 25.0  # m/s — nominal traffic / cruise target
V_MIN = 0.4 * V_TARGET  # 10 m/s — below this, survival terms are gated down
V_FLOOR = 0.15 * V_TARGET  # 3.75 m/s — hard disqualify (crawl / parked hack)
K_GATE = 8.0  # sigmoid steepness around v_min
HACK_PENALTY = 10.0  # magnitude when v_eff < v_floor (F = −HACK_PENALTY)
LOW_SPEED_WEIGHT = 0.50  # weight on (v_min − v_eff)+ / v_target

# Lag vs surrounding traffic (~≥20 m/s historically; soft@25 uses V_TARGET).
_LAG_SPEED_MPS = 18.0
# Constant-cruise detector extras (progress_std ≈ 0 + flat speed band).
_CRUISE_PROGRESS_STD_MAX = 1.0
_CRUISE_SPEED_SPREAD_MAX = 0.35
_CONSTANT_CRUISE_PENALTY = 0.35


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


def _sigmoid(x: float) -> float:
    # Numerically stable logistic.
    z = float(x)
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _degeneracy_penalty(metrics: Mapping[str, Any]) -> float:
    """
    Penalize constant-cruise / single-outcome exploits that inflate SR.

    Uses fields logged by ``adapter._eval_metrics`` when present.
    """
    sr = _finite(metrics.get("SR", metrics.get("sr", 0.0)))
    if sr < 0.5:
        return 0.0
    pen = 0.0
    p10 = metrics.get("speed_p10")
    p90 = metrics.get("speed_p90")
    spread = None
    if p10 is not None and p90 is not None:
        spread = abs(_finite(p90) - _finite(p10))
        # Near-constant speed band while surviving → likely idle cruise hack.
        if spread < _CRUISE_SPEED_SPREAD_MAX:
            pen += 0.25 * (1.0 - spread / _CRUISE_SPEED_SPREAD_MAX)
    lc = metrics.get("lane_change_rate")
    if lc is not None and _finite(lc) < 0.02 and sr >= 0.9:
        # Perfect survival with almost no lane changes on dense holdout is suspicious.
        pen += 0.15
    ou = metrics.get("outcome_unique")
    if ou is not None and int(_finite(ou)) <= 1 and sr >= 0.99:
        pen += 0.10
    # Flat progress across episodes + flat speed + no lane Δ → soft@threshold lock.
    pstd = metrics.get("progress_std")
    if (
        pstd is not None
        and _finite(pstd) <= _CRUISE_PROGRESS_STD_MAX
        and sr >= 0.95
        and (spread is None or spread < _CRUISE_SPEED_SPREAD_MAX)
        and (lc is None or _finite(lc) < 0.02)
    ):
        pen += float(_CONSTANT_CRUISE_PENALTY)
    return float(min(0.85, pen))


def _metrics_for_fitness(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Prefer denser holdout eval when the trainer nested it."""
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        return holdout
    if str(metrics.get("eval_profile", "")).lower() == "holdout":
        return metrics
    return metrics


def _v_eff(src: Mapping[str, Any]) -> float:
    """Effective speed: prefer p10 (anti-spike), else mean_speed."""
    if src.get("speed_p10") is not None:
        return _finite(src.get("speed_p10"))
    return _finite(src.get("mean_speed", src.get("ITR", src.get("avg_speed", 0.0))))


def _speed_gate(v_eff: float) -> float:
    """sigmoid(k · (v_eff − v_min) / v_target) ∈ (0, 1)."""
    return _sigmoid(K_GATE * (float(v_eff) - V_MIN) / V_TARGET)


def _low_speed_penalty(v_eff: float) -> float:
    """Extra cost when v_eff is below the 'real driving' floor v_min."""
    gap = max(0.0, V_MIN - float(v_eff))
    if gap <= 0.0:
        return 0.0
    return float(LOW_SPEED_WEIGHT * gap / V_TARGET)


def _lag_penalty(sr: float, mean_speed: float) -> float:
    if sr >= 0.5 and mean_speed < _LAG_SPEED_MPS:
        return 0.40 * (_LAG_SPEED_MPS - mean_speed) / _LAG_SPEED_MPS
    return 0.0


def highway_fitness(metrics: Optional[Mapping[str, Any]]) -> float:
    """
    Official highway fitness (higher better).

    Prefer holdout metrics when present.

        v_eff = speed_p10 if present else mean_speed
        if v_eff < v_floor:  return −hack_penalty   # hard disqualify

        gate = sigmoid(k · (v_eff − v_min) / v_target)

        F = gate · (SR − CR − 0.5·TR)
          + 0.35 · tanh(progress / 800)
          + 0.25 · tanh(mean_speed / 25)
          + 0.15 · soft_success · gate
          − lag_penalty − degeneracy_penalty − low_speed_penalty
    """
    if not metrics:
        return float("-inf")

    src = _metrics_for_fitness(metrics)
    v_eff = _v_eff(src)
    if v_eff < V_FLOOR:
        return float(-HACK_PENALTY)

    sr = _finite(src.get("SR", src.get("sr", 0.0)))
    cr = _finite(src.get("CR", src.get("cr", 0.0)))
    tr = _finite(src.get("TR", src.get("tr", 0.0)))
    pl = _finite(src.get("PL", src.get("mean_progress", 0.0)))
    speed = _finite(
        src.get("mean_speed", src.get("ITR", src.get("avg_speed", 0.0)))
    )
    soft = _finite(src.get("soft_success", 0.0))

    gate = _speed_gate(v_eff)
    survival = sr - cr - 0.5 * tr
    progress_term = 0.35 * _tanh01(pl / _REF_PROGRESS_M)
    speed_term = 0.25 * _tanh01(speed / _REF_SPEED_MPS)
    soft_term = 0.15 * soft * gate

    lag = _lag_penalty(sr, speed)
    degen = _degeneracy_penalty(src)
    low = _low_speed_penalty(v_eff)

    return float(
        gate * survival
        + progress_term
        + speed_term
        + soft_term
        - lag
        - degen
        - low
    )


def fitness_components(metrics: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Diagnostics for logs / tests (not used by selection hot path)."""
    if not metrics:
        return {"fitness": float("-inf"), "disqualified": 1.0}
    src = _metrics_for_fitness(metrics)
    v_eff = _v_eff(src)
    if v_eff < V_FLOOR:
        return {
            "fitness": float(-HACK_PENALTY),
            "v_eff": float(v_eff),
            "gate": 0.0,
            "disqualified": 1.0,
            "hack_penalty": float(HACK_PENALTY),
        }
    gate = _speed_gate(v_eff)
    return {
        "fitness": float(highway_fitness(metrics)),
        "v_eff": float(v_eff),
        "gate": float(gate),
        "v_min": float(V_MIN),
        "v_floor": float(V_FLOOR),
        "v_target": float(V_TARGET),
        "disqualified": 0.0,
        "low_speed_penalty": float(_low_speed_penalty(v_eff)),
    }


# Backward-compatible alias (same function object).
highway_navigation_scalar = highway_fitness


def format_highway_metrics_line(metrics: Mapping[str, Any]) -> str:
    """One-line human summary for logs (fitness is the official objective)."""
    src: Mapping[str, Any] = metrics
    holdout = metrics.get("holdout")
    tag = ""
    if isinstance(holdout, Mapping) and holdout:
        src = holdout
        tag = "holdout "
    sr = _finite(src.get("SR"))
    cr = _finite(src.get("CR"))
    tr = _finite(src.get("TR"))
    pl = _finite(src.get("PL"))
    spd = _finite(src.get("mean_speed", src.get("ITR")))
    soft = _finite(src.get("soft_success"))
    lc = _finite(src.get("lane_change_rate"))
    fit = highway_fitness(metrics)
    v_eff = _v_eff(src)
    gate = 0.0 if v_eff < V_FLOOR else _speed_gate(v_eff)
    n_ok = src.get("n_success")
    n_cr = src.get("n_collision")
    n_ep = src.get("n_eval_episodes")
    counts = ""
    if n_ok is not None and n_cr is not None and n_ep is not None:
        counts = f" ok={int(n_ok)}/{int(n_ep)} col={int(n_cr)}"
    dq = " DQ" if v_eff < V_FLOOR else ""
    return (
        f"{tag}SR={sr:.2f} CR={cr:.2f} TR={tr:.2f} "
        f"progress={pl:.1f}m speed={spd:.1f}m/s v_eff={v_eff:.1f} "
        f"gate={gate:.2f} soft={soft:.3f} laneΔ={lc:.3f} "
        f"fitness={fit:.3f}{dq}{counts}"
    )


def attach_fitness(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cache official ``fitness`` (and alias ``selection_scalar``) on metrics dict.
    """
    out = dict(metrics)
    out["domain"] = "highway"
    value = float(highway_fitness(out))
    out["fitness"] = value
    # Alias kept so CrowdNav-era readers / old JSONL still work.
    out["selection_scalar"] = value
    comps = fitness_components(out)
    out["fitness_gate"] = float(comps.get("gate", 0.0))
    out["fitness_v_eff"] = float(comps.get("v_eff", 0.0))
    out["fitness_disqualified"] = float(comps.get("disqualified", 0.0))
    return out


def attach_selection_scalar(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Deprecated name — use ``attach_fitness``."""
    return attach_fitness(metrics)

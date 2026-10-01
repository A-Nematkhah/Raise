"""
Automatic, formula-free ranking for the highway RAISE / EUREKA loop.

Replaces hand-weighted ``highway_fitness`` for *population ordering*:

1. Auto-calibrated feasibility thresholds (reference rollout or generation
   percentiles) — no hand-picked ``v_min`` / ``v_floor`` as the breeding bar.
2. Pareto non-dominated sorting (NSGA-II) on raw objectives + crowding
   distance — no hand-picked linear weights.

``rank_population()`` returns candidates best → worst. Highway breeding
defaults to ``evolve_rank=pareto`` (see ``raise_core.raise_loop.evolve_rank``).

Scalar ``highway_fitness`` remains a human-facing diagnostic only and must
not be shown to the LLM as the selection objective.

Calibration modes (``calibration_mode``):

* ``population``      — legacy: SLOWER/FASTER fixed-action env reference for
  v_floor / CR·TR ceilings (IDLE is useless here — CR≈1). Falls back to
  generation percentiles only if the gear-endpoint rollout fails. SE slack
  on CR/TR so eval noise does not hard-kill.
* ``env_measured``    — survival-only feasibility; ambient non-ego traffic
  speed stats measured on the holdout config are attached as information.
* ``no_speed_floor``  — survival-only feasibility (SR > 0), no speed floor,
  no CR/TR ceilings; safety is handled by −CR/−TR Pareto objectives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from domains.highway.env_wrapper import (
    HOLDOUT_SEED_OFFSET,
    HOLDOUT_VEHICLES,
    default_env_config,
    make_base_env,
)

CALIBRATION_MODES = ("population", "env_measured", "no_speed_floor")
DEFAULT_CALIBRATION_MODE = "no_speed_floor"
LEGACY_CALIBRATION_MODE = "population"


def parse_calibration_mode(value: object) -> str:
    key = str(value or "").strip().lower()
    if not key:
        return DEFAULT_CALIBRATION_MODE
    if key not in CALIBRATION_MODES:
        raise ValueError(
            f"calibration_mode must be one of {CALIBRATION_MODES}, got {value!r}"
        )
    return key


@dataclass
class Metrics:
    """Raw evaluation metrics for one reward-function candidate."""

    candidate_id: str
    sr: float
    cr: float
    tr: float
    progress: float
    mean_speed: float
    soft_success: float
    speed_samples: Optional[np.ndarray] = field(default=None, repr=False)
    speed_p10: Optional[float] = None
    lane_change_rate: float = 0.0
    overtakes_per_km: float = 0.0
    n_eval_episodes: float = 0.0


@dataclass(frozen=True)
class ParetoObjectives:
    """
    Optional extras beside −CR, −TR, mean_speed.

    ``include_sr`` defaults False: with exclusive outcomes SR+CR+TR≈1, so SR
    is redundant with −CR/−TR and double-weights safety in crowding.
    """

    progress: bool = True
    lane_change: bool = False
    overtake: bool = False
    include_sr: bool = False

    def names(self) -> Tuple[str, ...]:
        out: List[str] = []
        if self.include_sr:
            out.append("SR")
        out.extend(["-CR", "-TR"])
        if self.progress:
            out.append("progress")
        out.append("mean_speed")
        if self.lane_change:
            out.append("lane_change_rate")
        if self.overtake:
            out.append("overtakes_per_km")
        return tuple(out)


LEGACY_OBJECTIVES = ParetoObjectives()


def bernoulli_se(p: float, n: float) -> float:
    """Standard error of a Bernoulli rate estimate; 0 if ``n < 2``."""
    nn = float(n)
    if nn < 2.0:
        return 0.0
    pp = min(1.0, max(0.0, float(p)))
    return float(np.sqrt(pp * (1.0 - pp) / nn))


@dataclass
class ReferenceStats:
    """
    Feasibility thresholds. ``None`` disables a gate.

    ``require_survival`` makes a candidate feasible only if SR > 0.
    ``se_margin`` adds 2×Bernoulli SE slack on CR/TR ceilings (eval noise).
    ``source`` records how thresholds were obtained (shown verbatim to the LLM
    so it is never told a population statistic was measured from traffic).
    """

    v_floor: Optional[float]
    cr_ceiling: Optional[float]
    tr_ceiling: Optional[float]
    require_survival: bool = False
    source: str = "population_percentile"
    ambient: Optional[Dict[str, float]] = None
    se_margin: bool = False


def survival_only_reference(
    *,
    source: str = "no_speed_floor",
    ambient: Optional[Mapping[str, float]] = None,
) -> ReferenceStats:
    return ReferenceStats(
        v_floor=None,
        cr_ceiling=None,
        tr_ceiling=None,
        require_survival=True,
        source=str(source),
        ambient=dict(ambient) if ambient else None,
    )


def calibrate_from_reference_rollout(
    reference_speed_samples: np.ndarray,
    reference_cr: float = 0.0,
    reference_tr: float = 0.0,
    floor_percentile: float = 10.0,
    safety_margin: float = 1.5,
    *,
    max_usable_cr: float = 0.5,
    default_cr_ceiling: float = 0.25,
    default_tr_ceiling: float = 0.05,
) -> Optional[ReferenceStats]:
    """
    Derive thresholds from a short non-RL rollout.

    Returns ``None`` when the reference ego is too crashy (e.g. IDLE with
    CR≈1): that calibration made high-SR cruise *infeasible* while crashy
    high-speed policies looked feasible. Callers should fall back to
    ``calibrate_from_population``.

    When usable, ``v_floor`` is clamped to ``[V_MIN, V_TARGET]`` so the
    feasibility floor stays inside the fitness diagnostic speed band
    (``V_TARGET`` here is *not* an LLM/selection target — only a clamp for
    auto-calibrated feasibility).
    """
    cr = float(reference_cr)
    if cr > float(max_usable_cr):
        return None

    samples = np.asarray(reference_speed_samples, dtype=np.float64).ravel()
    if samples.size == 0:
        samples = np.asarray([0.0], dtype=np.float64)
    v_raw = float(np.percentile(samples, floor_percentile))
    try:
        from domains.highway.metrics import V_MIN as _VMIN
        from domains.highway.metrics import V_TARGET as _VT

        v_lo, v_hi = float(_VMIN), float(_VT)
    except Exception:  # noqa: BLE001
        v_lo, v_hi = 10.0, 25.0
    # Keep auto-calibrated floor inside the diagnostic fitness speed band.
    v_floor = float(min(v_hi, max(v_lo, v_raw)))
    tr = float(reference_tr)
    cr_ceiling = min(1.0, cr * float(safety_margin) + 0.05)
    # Never open the CR gate fully from a weak reference.
    cr_ceiling = min(cr_ceiling, float(default_cr_ceiling) * 2.0)
    if cr_ceiling < 0.05:
        cr_ceiling = float(default_cr_ceiling)
    tr_ceiling = min(1.0, max(float(default_tr_ceiling), tr * float(safety_margin) + 0.05))
    return ReferenceStats(
        v_floor=v_floor,
        cr_ceiling=float(cr_ceiling),
        tr_ceiling=float(tr_ceiling),
        source="idle_ego_reference_rollout",
    )


def calibrate_from_population(
    population: Sequence[Metrics],
    floor_percentile: float = 15.0,
    safety_percentile: float = 85.0,
    absolute_v_floor: Optional[float] = None,
) -> ReferenceStats:
    """
    Fallback: thresholds from the current generation's own spread.
    Lenient early (everyone weak), stricter as the pop improves.

    ``absolute_v_floor`` (default: metrics.V_FLOOR) is a hard minimum so an
    all-crawler generation cannot calibrate itself into mutual feasibility.
    """
    if absolute_v_floor is None:
        try:
            from domains.highway.metrics import V_FLOOR as _VF

            absolute_v_floor = float(_VF)
        except Exception:  # noqa: BLE001
            absolute_v_floor = 0.0
    if not population:
        return ReferenceStats(
            v_floor=float(absolute_v_floor),
            cr_ceiling=1.0,
            tr_ceiling=1.0,
            source="population_percentile",
        )
    speeds = np.asarray([effective_speed(m) for m in population], dtype=np.float64)
    crs = np.asarray([m.cr for m in population], dtype=np.float64)
    trs = np.asarray([m.tr for m in population], dtype=np.float64)
    v_raw = float(np.percentile(speeds, floor_percentile))
    try:
        from domains.highway.metrics import V_MIN as _VMIN
        from domains.highway.metrics import V_TARGET as _VT

        v_lo, v_hi = float(_VMIN), float(_VT)
    except Exception:  # noqa: BLE001
        v_lo, v_hi = 10.0, 25.0
    v_floor = float(
        min(v_hi, max(float(absolute_v_floor), max(v_lo, v_raw)))
    )
    # Prefer CR from survivors so all-crash gens don't open the gate.
    survivor_crs = [m.cr for m in population if m.sr >= 0.3]
    if survivor_crs:
        cr_ceiling = float(np.percentile(np.asarray(survivor_crs), safety_percentile))
    else:
        cr_ceiling = min(0.25, float(np.percentile(crs, safety_percentile)))
    return ReferenceStats(
        v_floor=v_floor,
        cr_ceiling=float(cr_ceiling),
        tr_ceiling=float(np.percentile(trs, safety_percentile)),
        source="population_percentile",
        se_margin=True,
    )


def calibrate_from_gear_endpoints(
    slower_speeds: np.ndarray,
    slower_cr: float,
    slower_tr: float,
    faster_speeds: np.ndarray,
    faster_cr: float,
    faster_tr: float,
    *,
    floor_percentile: float = 10.0,
) -> ReferenceStats:
    """
    Feasibility from fixed DiscreteMetaAction endpoints (SLOWER / FASTER).

    ``v_floor`` = low percentile of SLOWER ego speed (env-measured, not a
    hand-picked cruise target). CR/TR ceilings use the worse of the two
    endpoint policies; ``is_feasible`` adds 2×SE slack when ``se_margin``.
    """
    slow = np.asarray(slower_speeds, dtype=np.float64).ravel()
    if slow.size == 0:
        slow = np.asarray([0.0], dtype=np.float64)
    v_floor = float(np.percentile(slow, floor_percentile))
    cr_ceiling = float(max(float(slower_cr), float(faster_cr)))
    tr_ceiling = float(max(float(slower_tr), float(faster_tr)))
    # Keep a usable gate even if both endpoints were perfect.
    if cr_ceiling < 0.05:
        cr_ceiling = 0.05
    if tr_ceiling < 0.05:
        tr_ceiling = 0.05
    return ReferenceStats(
        v_floor=v_floor,
        cr_ceiling=cr_ceiling,
        tr_ceiling=tr_ceiling,
        source="gear_endpoints_slower_faster",
        se_margin=True,
    )


def reference_for_mode(
    mode: str,
    population: Sequence[Metrics],
    *,
    ambient: Optional[Mapping[str, float]] = None,
    gear_ref: Optional[ReferenceStats] = None,
) -> ReferenceStats:
    """Per-generation thresholds for ``mode`` (see module docstring)."""
    key = parse_calibration_mode(mode)
    if key == "population":
        # Prefer fixed SLOWER/FASTER env reference over self-referential percentiles.
        if gear_ref is not None:
            return gear_ref
        ref = calibrate_from_population(population)
        # Still apply SE slack so noisy CR does not hard-kill.
        ref.se_margin = True
        return ref
    if key == "env_measured":
        return survival_only_reference(source="env_measured", ambient=ambient)
    return survival_only_reference(source="no_speed_floor")


def effective_speed(m: Metrics, percentile: float = 10.0) -> float:
    """
    Robust speed for feasibility.

    Prefer explicit ``speed_p10`` (anti-spike, matches highway eval extras),
    then per-step samples percentile, else mean_speed.
    """
    if m.speed_p10 is not None:
        return float(m.speed_p10)
    if m.speed_samples is not None and len(m.speed_samples) > 0:
        return float(np.percentile(np.asarray(m.speed_samples, dtype=np.float64), percentile))
    return float(m.mean_speed)


def is_feasible(m: Metrics, ref: ReferenceStats) -> bool:
    if ref.require_survival and not m.sr > 0.0:
        return False
    if ref.v_floor is not None and effective_speed(m) < ref.v_floor:
        return False
    n = float(m.n_eval_episodes or 0.0)
    if ref.cr_ceiling is not None:
        slack = 2.0 * bernoulli_se(m.cr, n) if ref.se_margin else 0.0
        if m.cr > float(ref.cr_ceiling) + slack:
            return False
    if ref.tr_ceiling is not None:
        slack = 2.0 * bernoulli_se(m.tr, n) if ref.se_margin else 0.0
        if m.tr > float(ref.tr_ceiling) + slack:
            return False
    return True


def _objectives(m: Metrics, objectives: Optional[ParetoObjectives] = None) -> np.ndarray:
    """Raw objectives to maximize (soft_success is diagnostic-only)."""
    spec = objectives or LEGACY_OBJECTIVES
    vals: List[float] = []
    if spec.include_sr:
        vals.append(m.sr)
    vals.extend([-m.cr, -m.tr])
    if spec.progress:
        vals.append(m.progress)
    vals.append(m.mean_speed)
    if spec.lane_change:
        vals.append(m.lane_change_rate)
    if spec.overtake:
        vals.append(m.overtakes_per_km)
    return np.asarray(vals, dtype=np.float64)


def _objective_margins(
    a: Metrics, b: Metrics, objectives: Optional[ParetoObjectives] = None
) -> np.ndarray:
    """
    Per-objective noise margins (2×SE on rate dims; tiny absolute elsewhere).

    Used so dominance ignores differences smaller than sampling noise.
    """
    spec = objectives or LEGACY_OBJECTIVES
    na = float(a.n_eval_episodes or 0.0)
    nb = float(b.n_eval_episodes or 0.0)
    # Conservative: use the larger SE of the two candidates (0 if n unknown).
    margins: List[float] = []
    if spec.include_sr:
        margins.append(2.0 * max(bernoulli_se(a.sr, na), bernoulli_se(b.sr, nb)))
    margins.append(2.0 * max(bernoulli_se(a.cr, na), bernoulli_se(b.cr, nb)))
    margins.append(2.0 * max(bernoulli_se(a.tr, na), bernoulli_se(b.tr, nb)))
    if spec.progress:
        margins.append(1e-6)
    margins.append(1e-6)  # mean_speed
    if spec.lane_change:
        margins.append(1e-6)
    if spec.overtake:
        margins.append(1e-6)
    return np.asarray(margins, dtype=np.float64)


def dominates(a: Metrics, b: Metrics, objectives: Optional[ParetoObjectives] = None) -> bool:
    """
    a dominates b iff a is ≥ b on every objective within noise and strictly
    better on at least one objective by more than the noise margin.
    """
    oa, ob = _objectives(a, objectives), _objectives(b, objectives)
    m = _objective_margins(a, b, objectives)
    not_worse = bool(np.all(oa >= ob - m))
    strictly_better = bool(np.any(oa > ob + m))
    return not_worse and strictly_better


def pareto_fronts(
    feasible: Sequence[Metrics], objectives: Optional[ParetoObjectives] = None
) -> List[List[Metrics]]:
    """Non-dominated sorting (NSGA-II front assignment)."""
    remaining = list(feasible)
    fronts: List[List[Metrics]] = []
    while remaining:
        front = [
            a
            for a in remaining
            if not any(
                dominates(b, a, objectives) for b in remaining if b is not a
            )
        ]
        if not front:
            # Degenerate guard — should not happen; dump rest as last front.
            fronts.append(remaining)
            break
        fronts.append(front)
        front_ids = {id(x) for x in front}
        remaining = [m for m in remaining if id(m) not in front_ids]
    return fronts


def crowding_distance(
    front: Sequence[Metrics], objectives: Optional[ParetoObjectives] = None
) -> Dict[str, float]:
    """
    Diversity tie-breaker within a front.

    Important: work on a *copy* when sorting per objective so we do not
    scramble the caller's front order mid-pass (sort-direction bug fix).
    """
    n = len(front)
    dist = {m.candidate_id: 0.0 for m in front}
    if n <= 2:
        for m in front:
            dist[m.candidate_id] = float("inf")
        return dist
    n_obj = len(_objectives(front[0], objectives))
    for k in range(n_obj):
        ordered = sorted(
            front, key=lambda m: float(_objectives(m, objectives)[k])
        )
        vmin = float(_objectives(ordered[0], objectives)[k])
        vmax = float(_objectives(ordered[-1], objectives)[k])
        dist[ordered[0].candidate_id] = float("inf")
        dist[ordered[-1].candidate_id] = float("inf")
        if vmax == vmin:
            continue
        span = vmax - vmin
        for i in range(1, n - 1):
            prev_v = float(_objectives(ordered[i - 1], objectives)[k])
            next_v = float(_objectives(ordered[i + 1], objectives)[k])
            dist[ordered[i].candidate_id] += (next_v - prev_v) / span
    return dist


def rank_population(
    population: Sequence[Metrics],
    reference_speed_samples: Optional[np.ndarray] = None,
    reference_cr: Optional[float] = None,
    reference_tr: Optional[float] = None,
    ref: Optional[ReferenceStats] = None,
    objectives: Optional[ParetoObjectives] = None,
) -> List[Metrics]:
    """
    Drop-in replacement for scalar fitness sort: best → worst.

    Infeasible candidates always rank after all feasible ones; among
    infeasible, higher effective speed ranks better (crawlers last).
    """
    pop = list(population)
    if not pop:
        return []

    if ref is None:
        if reference_speed_samples is not None:
            ref = calibrate_from_reference_rollout(
                reference_speed_samples,
                float(reference_cr or 0.0),
                float(reference_tr or 0.0),
            )
        if ref is None:
            ref = calibrate_from_population(pop)

    feasible = [m for m in pop if is_feasible(m, ref)]
    # Preserve input order for stable membership (avoid dataclass value traps).
    feas_ids = {id(m) for m in feasible}
    infeasible = [m for m in pop if id(m) not in feas_ids]

    ordered: List[Metrics] = []
    for front in pareto_fronts(feasible, objectives):
        cd = crowding_distance(front, objectives)
        # Higher crowding distance first (more diverse = preferred tie-break).
        front_sorted = sorted(
            front, key=lambda m: cd[m.candidate_id], reverse=True
        )
        ordered.extend(front_sorted)

    if ref.require_survival:
        # Survival-only mode carries no speed preference: order the non-
        # surviving candidates by the same dominance + crowding.
        for front in pareto_fronts(infeasible, objectives):
            cd = crowding_distance(front, objectives)
            ordered.extend(
                sorted(front, key=lambda m: cd[m.candidate_id], reverse=True)
            )
        return ordered

    # Among infeasible: prefer closer-to-driving over pure crawlers (desc speed).
    infeasible_sorted = sorted(
        infeasible, key=lambda m: effective_speed(m), reverse=True
    )
    ordered.extend(infeasible_sorted)
    return ordered


def metrics_from_mapping(
    candidate_id: str,
    metrics: Mapping[str, Any],
) -> Metrics:
    """Build ``Metrics`` from highway ``last_metrics`` (prefer holdout)."""
    src: Mapping[str, Any] = metrics
    holdout = metrics.get("holdout")
    if isinstance(holdout, Mapping) and holdout:
        src = holdout
    elif str(metrics.get("eval_profile", "")).lower() == "holdout":
        src = metrics

    def _f(key: str, *alts: str, default: float = 0.0) -> float:
        for k in (key,) + alts:
            if k in src and src[k] is not None:
                try:
                    return float(src[k])
                except (TypeError, ValueError):
                    continue
        return float(default)

    samples = src.get("speed_samples")
    arr = None
    if samples is not None:
        try:
            arr = np.asarray(samples, dtype=np.float64).ravel()
            if arr.size == 0:
                arr = None
        except Exception:  # noqa: BLE001
            arr = None

    p10 = None
    if src.get("speed_p10") is not None:
        try:
            p10 = float(src["speed_p10"])
        except (TypeError, ValueError):
            p10 = None

    return Metrics(
        candidate_id=str(candidate_id),
        sr=_f("SR", "sr"),
        cr=_f("CR", "cr"),
        tr=_f("TR", "tr"),
        progress=_f("PL", "mean_progress", "progress"),
        mean_speed=_f("mean_speed", "ITR", "avg_speed"),
        soft_success=_f("soft_success"),
        speed_samples=arr,
        speed_p10=p10,
        lane_change_rate=_f("lane_change_rate"),
        overtakes_per_km=_f("overtakes_per_km"),
        n_eval_episodes=_f("n_eval_episodes", default=0.0),
    )


def stamp_pareto_ranks(
    candidates: Sequence[Any],
    ordered_metrics: Sequence[Metrics],
    *,
    ref: Optional[ReferenceStats] = None,
    objectives: Optional[ParetoObjectives] = None,
) -> None:
    """
    Write Pareto rank metadata onto candidates (best rank = 0).

    Also stamps ``pareto_front`` (0 = first non-dominated front among
    feasible; infeasible get ``pareto_front = None`` / not front-0).

    Does **not** overwrite ``fitness`` / ``selection_scalar`` (those stay as
    legacy ``highway_fitness`` from Stage II). Uses ``pareto_score = n - rank``
    as a separate higher-is-better ordinal for diagnostics only.
    """
    by_id = {str(m.candidate_id): (i, m) for i, m in enumerate(ordered_metrics)}
    n = max(1, len(ordered_metrics))

    # Front index among feasible only (NSGA-II).
    if ref is not None:
        feasible = [m for m in ordered_metrics if is_feasible(m, ref)]
    else:
        feasible = list(ordered_metrics)
    front_of: Dict[str, int] = {}
    for fi, front in enumerate(pareto_fronts(feasible, objectives)):
        for m in front:
            front_of[str(m.candidate_id)] = int(fi)

    for c in candidates:
        cid = str(getattr(c, "candidate_id", ""))
        if cid not in by_id:
            continue
        rank_i, m = by_id[cid]
        md = dict(getattr(c, "metadata", None) or {})
        md["pareto_rank"] = int(rank_i)
        md["pareto_n"] = int(n)
        md["pareto_score"] = float(n - rank_i)
        if cid in front_of:
            md["pareto_front"] = int(front_of[cid])
            md["pareto_front0"] = bool(front_of[cid] == 0)
        else:
            # Infeasible (or unknown): not on any feasible front.
            md["pareto_front"] = None
            md["pareto_front0"] = False
        md.pop("pareto_use_progress", None)
        md["pareto_objectives"] = list((objectives or LEGACY_OBJECTIVES).names())
        if ref is not None:
            md["pareto_feasible"] = bool(is_feasible(m, ref))
            md["pareto_calibration_source"] = str(ref.source)
            md["pareto_require_survival"] = bool(ref.require_survival)
            for key, val in (
                ("pareto_v_floor", ref.v_floor),
                ("pareto_cr_ceiling", ref.cr_ceiling),
                ("pareto_tr_ceiling", ref.tr_ceiling),
            ):
                if val is None:
                    md.pop(key, None)
                else:
                    md[key] = float(val)
            if ref.ambient:
                md["pareto_ambient_traffic"] = dict(ref.ambient)
            else:
                md.pop("pareto_ambient_traffic", None)
        c.metadata = md


def _rollout_constant_action(
    *,
    action: int,
    n_episodes: int,
    seed: int,
    seed_offset: int,
) -> Tuple[np.ndarray, float, float]:
    speeds: List[float] = []
    n_crash = 0
    n_off = 0
    n_eps = max(1, int(n_episodes))
    for ep in range(n_eps):
        env = make_base_env(seed=int(seed) + ep)
        env.reset(seed=int(seed) + int(seed_offset) + ep)
        done = False
        crashed = False
        off = False
        while not done:
            _obs, _r, terminated, truncated, info = env.step(int(action))
            info = dict(info or {})
            try:
                veh = getattr(env.unwrapped, "vehicle", None)
                if veh is not None:
                    speeds.append(float(veh.speed))
                    if hasattr(veh, "on_road") and not bool(veh.on_road):
                        off = True
            except Exception:  # noqa: BLE001
                pass
            if info.get("crashed"):
                crashed = True
            if info.get("off_road"):
                off = True
            done = bool(terminated or truncated)
        env.close()
        if crashed:
            n_crash += 1
        elif off:
            n_off += 1
    cr = float(n_crash) / float(n_eps)
    tr = float(n_off) / float(n_eps)
    return np.asarray(speeds, dtype=np.float64), cr, tr


def collect_gear_endpoint_reference_stats(
    *,
    n_episodes: int = 4,
    seed: int = 425,
) -> Tuple[np.ndarray, float, float, np.ndarray, float, float]:
    """
    Fixed DiscreteMetaAction endpoints: SLOWER=4 and FASTER=3.

    Returns
    -------
    slower_speeds, slower_cr, slower_tr, faster_speeds, faster_cr, faster_tr
    """
    # DiscreteMetaAction: LANE_LEFT=0 IDLE=1 LANE_RIGHT=2 FASTER=3 SLOWER=4
    slow_s, slow_cr, slow_tr = _rollout_constant_action(
        action=4, n_episodes=n_episodes, seed=seed, seed_offset=20_003
    )
    fast_s, fast_cr, fast_tr = _rollout_constant_action(
        action=3, n_episodes=n_episodes, seed=seed + 17, seed_offset=30_003
    )
    return slow_s, slow_cr, slow_tr, fast_s, fast_cr, fast_tr


def collect_ambient_traffic_stats(
    *,
    n_episodes: int = 4,
    seed: int = 425,
) -> Dict[str, float]:
    """
    Measure the speeds of the **non-ego** vehicles on the holdout traffic
    config (ego held at its spawn gear with IDLE). Informational only — the
    result never gates feasibility.
    """
    cfg = default_env_config()
    cfg["vehicles_count"] = int(HOLDOUT_VEHICLES)
    speeds: List[float] = []
    n_eps = max(1, int(n_episodes))
    for ep in range(n_eps):
        env = make_base_env(seed=int(seed) + ep, config=cfg)
        env.reset(seed=int(seed) + int(HOLDOUT_SEED_OFFSET) + ep)
        done = False
        while not done:
            _obs, _r, terminated, truncated, _info = env.step(1)
            ego = env.unwrapped.vehicle
            for v in env.unwrapped.road.vehicles:
                if v is not ego:
                    speeds.append(float(v.speed))
            done = bool(terminated or truncated)
        env.close()
    arr = np.asarray(speeds, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "traffic_speed_mean": float(np.mean(arr)),
        "traffic_speed_p10": float(np.percentile(arr, 10)),
        "traffic_speed_p50": float(np.percentile(arr, 50)),
        "traffic_speed_p90": float(np.percentile(arr, 90)),
        "n_samples": float(arr.size),
        "n_episodes": float(n_eps),
        "vehicles_count": float(HOLDOUT_VEHICLES),
    }

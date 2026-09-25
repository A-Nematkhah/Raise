"""
Automatic, formula-free ranking for the highway RAISE / EUREKA loop.

Replaces hand-weighted ``highway_fitness`` for *population ordering*:

1. Auto-calibrated feasibility thresholds (reference rollout or generation
   percentiles) — no hand-picked ``v_min`` / ``v_floor``.
2. Pareto non-dominated sorting (NSGA-II) on raw objectives + crowding
   distance — no hand-picked linear weights.

``rank_population()`` returns candidates best → worst. Use it instead of
sorting by a scalar fitness when selecting parents for the next generation.

Scalar ``highway_fitness`` may still exist for surrogate / legacy logs; it is
not used by ``evolve_rank=pareto``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


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


@dataclass
class ReferenceStats:
    """Auto-derived feasibility thresholds — no hand-picked constants."""

    v_floor: float
    cr_ceiling: float
    tr_ceiling: float


def calibrate_from_reference_rollout(
    reference_speed_samples: np.ndarray,
    reference_cr: float = 0.0,
    reference_tr: float = 0.0,
    floor_percentile: float = 10.0,
    safety_margin: float = 1.5,
) -> ReferenceStats:
    """
    Derive thresholds from a short rollout of the env's built-in traffic
    / non-RL ego policy instead of writing them by hand.
    """
    samples = np.asarray(reference_speed_samples, dtype=np.float64).ravel()
    if samples.size == 0:
        samples = np.asarray([0.0], dtype=np.float64)
    v_floor = float(np.percentile(samples, floor_percentile))
    cr = float(reference_cr)
    tr = float(reference_tr)
    return ReferenceStats(
        v_floor=v_floor,
        cr_ceiling=min(1.0, cr * safety_margin + 0.05),
        tr_ceiling=min(1.0, tr * safety_margin + 0.05),
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
            v_floor=float(absolute_v_floor), cr_ceiling=1.0, tr_ceiling=1.0
        )
    speeds = np.asarray([effective_speed(m) for m in population], dtype=np.float64)
    crs = np.asarray([m.cr for m in population], dtype=np.float64)
    trs = np.asarray([m.tr for m in population], dtype=np.float64)
    v_floor = max(float(absolute_v_floor), float(np.percentile(speeds, floor_percentile)))
    return ReferenceStats(
        v_floor=v_floor,
        cr_ceiling=float(np.percentile(crs, safety_percentile)),
        tr_ceiling=float(np.percentile(trs, safety_percentile)),
    )


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
    return (
        effective_speed(m) >= ref.v_floor
        and m.cr <= ref.cr_ceiling
        and m.tr <= ref.tr_ceiling
    )


def _objectives(m: Metrics) -> np.ndarray:
    """Objectives to maximize."""
    return np.array(
        [m.sr, -m.cr, -m.tr, m.progress, m.mean_speed, m.soft_success],
        dtype=np.float64,
    )


def dominates(a: Metrics, b: Metrics) -> bool:
    oa, ob = _objectives(a), _objectives(b)
    return bool(np.all(oa >= ob) and np.any(oa > ob))


def pareto_fronts(feasible: Sequence[Metrics]) -> List[List[Metrics]]:
    """Non-dominated sorting (NSGA-II front assignment)."""
    remaining = list(feasible)
    fronts: List[List[Metrics]] = []
    while remaining:
        front = [
            a
            for a in remaining
            if not any(dominates(b, a) for b in remaining if b is not a)
        ]
        if not front:
            # Degenerate guard — should not happen; dump rest as last front.
            fronts.append(remaining)
            break
        fronts.append(front)
        front_ids = {id(x) for x in front}
        remaining = [m for m in remaining if id(m) not in front_ids]
    return fronts


def crowding_distance(front: Sequence[Metrics]) -> Dict[str, float]:
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
    n_obj = len(_objectives(front[0]))
    for k in range(n_obj):
        ordered = sorted(front, key=lambda m: float(_objectives(m)[k]))
        vmin = float(_objectives(ordered[0])[k])
        vmax = float(_objectives(ordered[-1])[k])
        dist[ordered[0].candidate_id] = float("inf")
        dist[ordered[-1].candidate_id] = float("inf")
        if vmax == vmin:
            continue
        span = vmax - vmin
        for i in range(1, n - 1):
            prev_v = float(_objectives(ordered[i - 1])[k])
            next_v = float(_objectives(ordered[i + 1])[k])
            dist[ordered[i].candidate_id] += (next_v - prev_v) / span
    return dist


def rank_population(
    population: Sequence[Metrics],
    reference_speed_samples: Optional[np.ndarray] = None,
    reference_cr: Optional[float] = None,
    reference_tr: Optional[float] = None,
    ref: Optional[ReferenceStats] = None,
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
        else:
            ref = calibrate_from_population(pop)

    feasible = [m for m in pop if is_feasible(m, ref)]
    # Preserve input order for stable membership (avoid dataclass value traps).
    feas_ids = {id(m) for m in feasible}
    infeasible = [m for m in pop if id(m) not in feas_ids]

    ordered: List[Metrics] = []
    for front in pareto_fronts(feasible):
        cd = crowding_distance(front)
        # Higher crowding distance first (more diverse = preferred tie-break).
        front_sorted = sorted(
            front, key=lambda m: cd[m.candidate_id], reverse=True
        )
        ordered.extend(front_sorted)

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
    )


def stamp_pareto_ranks(
    candidates: Sequence[Any],
    ordered_metrics: Sequence[Metrics],
    *,
    ref: Optional[ReferenceStats] = None,
) -> None:
    """
    Write Pareto rank metadata onto candidates (best rank = 0).

    Does **not** overwrite ``fitness`` / ``selection_scalar`` (those stay as
    ``highway_fitness`` from Stage II). Uses ``pareto_score = n - rank`` as a
    separate higher-is-better ordinal for diagnostics only.
    """
    by_id = {str(m.candidate_id): (i, m) for i, m in enumerate(ordered_metrics)}
    n = max(1, len(ordered_metrics))
    for c in candidates:
        cid = str(getattr(c, "candidate_id", ""))
        if cid not in by_id:
            continue
        rank_i, m = by_id[cid]
        md = dict(getattr(c, "metadata", None) or {})
        md["pareto_rank"] = int(rank_i)
        md["pareto_n"] = int(n)
        md["pareto_score"] = float(n - rank_i)
        if ref is not None:
            md["pareto_feasible"] = bool(is_feasible(m, ref))
            md["pareto_v_floor"] = float(ref.v_floor)
            md["pareto_cr_ceiling"] = float(ref.cr_ceiling)
            md["pareto_tr_ceiling"] = float(ref.tr_ceiling)
        c.metadata = md


def collect_reference_rollout_stats(
    *,
    n_episodes: int = 8,
    seed: int = 425,
) -> Tuple[np.ndarray, float, float]:
    """
    Short non-RL rollout on highway-fast-v0 (IDLE ego; IDM traffic).

    Returns (speed_samples, cr, tr) for ``calibrate_from_reference_rollout``.
    """
    from domains.highway.env_wrapper import make_base_env

    speeds: List[float] = []
    n_crash = 0
    n_off = 0
    n_eps = max(1, int(n_episodes))
    for ep in range(n_eps):
        env = make_base_env(seed=int(seed) + ep)
        obs, _ = env.reset(seed=int(seed) + 10_003 + ep)
        done = False
        crashed = False
        off = False
        while not done:
            # DiscreteMetaAction IDLE = 1 — ego coasts; traffic is IDM.
            action = 1
            obs, _r, terminated, truncated, info = env.step(action)
            info = dict(info or {})
            arr = np.asarray(obs, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 5:
                vx, vy = float(arr[0][3]), float(arr[0][4])
                speeds.append(float((vx * vx + vy * vy) ** 0.5))
            if info.get("crashed"):
                crashed = True
            if info.get("off_road"):
                off = True
            try:
                veh = getattr(env.unwrapped, "vehicle", None)
                if veh is not None and hasattr(veh, "on_road") and not bool(veh.on_road):
                    off = True
            except Exception:  # noqa: BLE001
                pass
            done = bool(terminated or truncated)
        env.close()
        if crashed:
            n_crash += 1
        elif off:
            n_off += 1
    cr = float(n_crash) / float(n_eps)
    tr = float(n_off) / float(n_eps)
    return np.asarray(speeds, dtype=np.float64), cr, tr


if __name__ == "__main__":
    pop = [
        Metrics(
            "crawler",
            sr=1.0,
            cr=0.0,
            tr=0.0,
            progress=15.0,
            mean_speed=1.5,
            soft_success=1.0,
        ),
        Metrics(
            "fast_risky",
            sr=0.7,
            cr=0.15,
            tr=0.10,
            progress=700.0,
            mean_speed=23.0,
            soft_success=0.8,
        ),
        Metrics(
            "balanced",
            sr=0.9,
            cr=0.04,
            tr=0.05,
            progress=600.0,
            mean_speed=20.0,
            soft_success=0.85,
        ),
    ]
    print("ranking (best -> worst):")
    for i, m in enumerate(rank_population(pop), 1):
        print(f"  {i}. {m.candidate_id}")

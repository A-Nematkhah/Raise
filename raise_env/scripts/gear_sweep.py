"""Sweep DiscreteMetaAction gears under IDLE (± optional rule-based lane change).

Reports SR, CR, mean_speed, progress, and mean TTC (when available) for each
gear of the active action config on the holdout traffic profile.

Usage
-----
python scripts/gear_sweep.py
python scripts/gear_sweep.py --episodes 30 --seeds 0,1,2 --out results/gear_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import raise_paths  # noqa: F401

from domains.highway.action_config import (
    get_action_settings,
    meta_target_speeds,
    set_action_settings,
    ActionModeSettings,
)
from domains.highway.env_wrapper import holdout_env_config, make_base_env

_LANE_LEFT = 0
_IDLE = 1
_LANE_RIGHT = 2
_FASTER = 3
_SLOWER = 4


def _ego_speed(env: Any) -> float:
    veh = env.unwrapped.vehicle
    return float(getattr(veh, "speed", 0.0) or 0.0)


def _ego_target(env: Any) -> float:
    veh = env.unwrapped.vehicle
    return float(getattr(veh, "target_speed", 0.0) or 0.0)


def _front_gap(env: Any) -> Optional[float]:
    """Longitudinal gap to the nearest vehicle ahead in the same lane (m)."""
    road = env.unwrapped.road
    ego = env.unwrapped.vehicle
    if ego is None or road is None:
        return None
    best: Optional[float] = None
    ego_lane = getattr(ego, "lane_index", None)
    for other in road.vehicles:
        if other is ego:
            continue
        other_lane = getattr(other, "lane_index", None)
        if ego_lane is not None and other_lane is not None and other_lane != ego_lane:
            continue
        dx = float(other.position[0] - ego.position[0])
        if dx <= 0.0:
            continue
        if best is None or dx < best:
            best = dx
    return best


def _ttc_to_front(env: Any) -> Optional[float]:
    """Simple same-lane TTC using relative longitudinal speed."""
    road = env.unwrapped.road
    ego = env.unwrapped.vehicle
    if ego is None or road is None:
        return None
    ego_lane = getattr(ego, "lane_index", None)
    best_ttc: Optional[float] = None
    for other in road.vehicles:
        if other is ego:
            continue
        other_lane = getattr(other, "lane_index", None)
        if ego_lane is not None and other_lane is not None and other_lane != ego_lane:
            continue
        dx = float(other.position[0] - ego.position[0])
        if dx <= 0.0:
            continue
        rel = float(ego.speed - other.speed)
        if rel <= 1e-3:
            continue
        ttc = dx / rel
        if best_ttc is None or ttc < best_ttc:
            best_ttc = ttc
    return best_ttc


def _reach_gear(env: Any, target: float, *, max_steps: int = 40) -> None:
    """Apply FASTER/SLOWER until ego target_speed matches ``target``, then IDLE."""
    for _ in range(max_steps):
        cur = _ego_target(env)
        if abs(cur - target) < 0.51:
            env.step(_IDLE)
            return
        action = _FASTER if cur < target - 0.1 else _SLOWER
        _obs, _r, term, trunc, _info = env.step(action)
        if term or trunc:
            return
    env.step(_IDLE)


def _rule_lane_change(env: Any, headway_m: float) -> int:
    gap = _front_gap(env)
    if gap is not None and gap < headway_m:
        # Prefer left then right; fall back to IDLE if neither helps.
        return _LANE_LEFT
    return _IDLE


def run_episode(
    *,
    gear: float,
    seed: int,
    variant: str,
    headway_m: float,
    duration_steps: int = 200,
) -> Dict[str, float]:
    env = make_base_env(seed=seed, config=holdout_env_config())
    env.reset(seed=seed)
    _reach_gear(env, gear)

    crashed = False
    off = False
    speed_sum = 0.0
    progress = 0.0
    steps = 0
    ttc_vals: List[float] = []
    prev_x = float(env.unwrapped.vehicle.position[0])

    for _ in range(duration_steps):
        action = _IDLE if variant == "idle" else _rule_lane_change(env, headway_m)
        _obs, _r, term, trunc, info = env.step(action)
        info = dict(info or {})
        spd = _ego_speed(env)
        speed_sum += spd
        x = float(env.unwrapped.vehicle.position[0])
        progress += max(0.0, x - prev_x)
        prev_x = x
        steps += 1
        ttc = _ttc_to_front(env)
        if ttc is not None and ttc < 1e6:
            ttc_vals.append(float(ttc))
        if info.get("crashed"):
            crashed = True
        if info.get("off_road"):
            off = True
        try:
            veh = env.unwrapped.vehicle
            if veh is not None and hasattr(veh, "on_road") and not bool(veh.on_road):
                off = True
        except Exception:  # noqa: BLE001
            pass
        if term or trunc:
            break
    env.close()

    mean_speed = speed_sum / float(max(1, steps))
    return {
        "sr": 0.0 if crashed or off else 1.0,
        "cr": 1.0 if crashed else 0.0,
        "tr": 1.0 if (off and not crashed) else 0.0,
        "mean_speed": float(mean_speed),
        "progress": float(progress),
        "mean_ttc": float(np.mean(ttc_vals)) if ttc_vals else float("nan"),
        "steps": float(steps),
    }


def sweep(
    *,
    gears: Sequence[float],
    episodes: int,
    seeds: Sequence[int],
    variant: str,
    headway_m: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gear in gears:
        ep_stats: List[Dict[str, float]] = []
        for seed in seeds:
            for ep in range(episodes):
                ep_stats.append(
                    run_episode(
                        gear=float(gear),
                        seed=int(seed) + 10_000 * ep,
                        variant=variant,
                        headway_m=headway_m,
                    )
                )
        n = float(len(ep_stats))
        row = {
            "gear": float(gear),
            "variant": variant,
            "n_episodes": int(n),
            "SR": float(np.mean([e["sr"] for e in ep_stats])),
            "CR": float(np.mean([e["cr"] for e in ep_stats])),
            "mean_speed": float(np.mean([e["mean_speed"] for e in ep_stats])),
            "progress": float(np.mean([e["progress"] for e in ep_stats])),
            "mean_ttc": float(
                np.nanmean([e["mean_ttc"] for e in ep_stats])
            ),
        }
        rows.append(row)
    return rows


def _print_table(rows: Sequence[Mapping[str, Any]]) -> None:
    header = (
        f"{'gear':>6} {'variant':>10} {'n':>4} {'SR':>5} {'CR':>5} "
        f"{'speed':>7} {'prog':>8} {'ttc':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ttc = r["mean_ttc"]
        ttc_s = f"{ttc:7.1f}" if ttc == ttc else f"{'n/a':>7}"
        print(
            f"{float(r['gear']):6.1f} {str(r['variant']):>10} {int(r['n_episodes']):4d} "
            f"{float(r['SR']):5.2f} {float(r['CR']):5.2f} "
            f"{float(r['mean_speed']):7.2f} {float(r['progress']):8.1f} {ttc_s}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per seed")
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds (default: 0,1,2)",
    )
    parser.add_argument(
        "--headway",
        type=float,
        default=25.0,
        help="Lane-change variant: change lane when front gap < this (m)",
    )
    parser.add_argument(
        "--variants",
        default="idle,lane_change",
        help="Comma-separated: idle and/or lane_change",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional CSV path (default: results/gear_sweep.csv)",
    )
    parser.add_argument(
        "--action-mode",
        choices=("meta_default", "meta_fine"),
        default="meta_default",
        help="Which DiscreteMetaAction gear set to sweep",
    )
    args = parser.parse_args()

    set_action_settings(ActionModeSettings(mode=str(args.action_mode)))
    gears = list(meta_target_speeds(get_action_settings()))
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]

    all_rows: List[Dict[str, Any]] = []
    for variant in variants:
        if variant not in ("idle", "lane_change"):
            raise SystemExit(f"Unknown variant {variant!r}")
        print(f"\n=== variant={variant} gears={gears} episodes/seed={args.episodes} ===")
        rows = sweep(
            gears=gears,
            episodes=int(args.episodes),
            seeds=seeds,
            variant=variant,
            headway_m=float(args.headway),
        )
        _print_table(rows)
        all_rows.extend(rows)

    out = str(args.out).strip() or os.path.join("results", "gear_sweep.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "gear",
                "variant",
                "n_episodes",
                "SR",
                "CR",
                "mean_speed",
                "progress",
                "mean_ttc",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

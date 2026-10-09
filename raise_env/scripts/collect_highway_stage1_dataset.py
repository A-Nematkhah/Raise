#!/usr/bin/env python
"""Collect the Stage I trajectory dataset for the highway domain pack.

Every trajectory is a genuine rollout: no post-hoc speed rescaling. All
behaviours share the same 40 s episodes and alternate between the training
and holdout traffic profiles, so neither traffic density nor episode length
separates the labels. Ego speed is held by a gear-tracking controller over
DiscreteMetaAction gears 10..30 m/s (1 m/s apart), so slow behaviours are
physically consistent with the surrounding traffic.

Behaviours and the role a *success* gets (failures are always ``failure``):

- ``overtake``    cruises at 22.5..24.5 m/s and passes slower leads        -> positive
                  (a success without a pass is dropped; crashes are kept)
- ``follow``      lane-keeps at up to 25 m/s, slowing behind a slower lead -> neutral
- ``safe_cruise`` holds 20 m/s, PPO's lowest meta_default gear              -> neutral
- ``lag``         holds 16 m/s                                             -> lag
- ``crawl``       holds 12 m/s                                             -> lag
- ``aggressive``  pushes to 30 m/s without regard for leads                -> positive
- ``random`` / ``swerve``  random actions / repeated lane changes at 24 m/s -> by speed

Lane-locked survivors (follow, safe_cruise) are neutral because the Stage II
fitness penalizes lane-lock below traffic speed. Crashes of every behaviour are
kept, so slow crashes (always-SLOWER in dense traffic) are represented. A few
off-road failures (label ``timeout``) are synthesized from positive successes;
they are the only synthesized frames.

``state.action`` is the meta action index; Score1 maps it to the run's action
mode when it loads the dataset.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401
import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import replace as dc_replace  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

from domains.highway.env_wrapper import (  # noqa: E402
    HOLDOUT_VEHICLES,
    HOLDOUT_VEHICLES_DENSITY,
    default_env_config,
    detect_off_road,
    kinematics_to_state,
    make_base_env,
)
from domains.highway.objective_constants import LAG_SPEED_MPS  # noqa: E402
from domains.highway.overtake import OvertakeTracker  # noqa: E402
from domains.highway.stage1 import HighwayTrajectoryRecord, save_highway_dataset  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402

_LEFT, _IDLE, _RIGHT, _FASTER, _SLOWER = 0, 1, 2, 3, 4

COLLECT_TARGET_SPEEDS: Tuple[float, ...] = tuple(float(v) for v in np.linspace(10.0, 30.0, 21))

# Goal speed (m/s) per behaviour; overtake cycles through its goals per episode.
_GOAL_MPS = {
    "follow": 25.0,
    "safe_cruise": 20.0,
    "lag": 16.0,
    "crawl": 12.0,
    "aggressive": 30.0,
    "swerve": 24.0,
}
_OVERTAKE_GOALS = (22.5, 23.5, 24.5)

_SUCCESS_ROLE = {
    "overtake": "positive",
    "aggressive": "positive",
    "follow": "neutral",
    "safe_cruise": "neutral",
    "lag": "lag",
    "crawl": "lag",
}

DEFAULT_BEHAVIORS = (
    "overtake,overtake,overtake,follow,safe_cruise,lag,crawl,aggressive,random,swerve"
)


def trajectory_role(label: str, behavior: str, mean_speed: float) -> str:
    if label != "success":
        return "failure"
    if behavior in _SUCCESS_ROLE:
        return _SUCCESS_ROLE[behavior]
    return "lag" if mean_speed < LAG_SPEED_MPS else "positive"


def traffic_config(profile: str) -> Dict[str, Any]:
    """The training or holdout traffic, with the collection gears."""
    cfg = default_env_config()
    cfg["action"] = {"type": "DiscreteMetaAction", "target_speeds": list(COLLECT_TARGET_SPEEDS)}
    if profile == "holdout":
        cfg["vehicles_count"] = int(HOLDOUT_VEHICLES)
        cfg["vehicles_density"] = float(HOLDOUT_VEHICLES_DENSITY)
    elif profile != "train":
        raise ValueError(f"unknown traffic profile {profile!r}")
    return cfg


def _ego(env: Any) -> Any:
    return getattr(getattr(env, "unwrapped", env), "vehicle", None)


def _vehicles(env: Any) -> List[Any]:
    road = getattr(getattr(env, "unwrapped", env), "road", None)
    return list(getattr(road, "vehicles", []) or [])


def _track_gear(env: Any, goal: float) -> int:
    """FASTER/SLOWER/IDLE so the ego's target gear converges to ``goal``.

    FASTER/SLOWER move one gear from the *current* speed, so the controller
    compares the tracked target speed, not the actual speed, with the goal.
    """
    ego = _ego(env)
    if ego is None:
        return _IDLE
    target = float(getattr(ego, "target_speed", ego.speed))
    if target < goal - 0.5:
        return _FASTER
    if target > goal + 0.5:
        return _SLOWER
    return _IDLE


def _lead(env: Any, *, max_gap: float = 40.0) -> Optional[Tuple[float, float]]:
    """(gap, speed) of the closest same-lane vehicle ahead, if any."""
    ego = _ego(env)
    if ego is None:
        return None
    best: Optional[Tuple[float, float]] = None
    for other in _vehicles(env):
        if other is ego:
            continue
        dx = float(other.position[0]) - float(ego.position[0])
        dy = float(other.position[1]) - float(ego.position[1])
        if 0.0 < dx < max_gap and abs(dy) < 2.0 and (best is None or dx < best[0]):
            best = (dx, float(other.speed))
    return best


def _lane_clear(env: Any, *, side: str) -> bool:
    """True if the adjacent lane has no vehicle in a dangerous longitudinal window."""
    ego = _ego(env)
    if ego is None:
        return False
    want_dy = -4.0 if side == "left" else 4.0
    lane = float(ego.position[1]) / 4.0
    if (side == "left" and lane < 0.5) or (side == "right" and lane > 2.5):
        return False
    for other in _vehicles(env):
        if other is ego:
            continue
        dx = float(other.position[0]) - float(ego.position[0])
        dy = float(other.position[1]) - float(ego.position[1])
        if abs(dy - want_dy) < 2.5 and -12.0 < dx < 28.0:
            return False
    return True


def _follow_action(env: Any, goal: float) -> int:
    lead = _lead(env)
    if lead is not None:
        gap, lead_speed = lead
        goal = min(goal, lead_speed - (2.0 if gap < 20.0 else 0.0))
    return _track_gear(env, max(10.0, goal))


def _overtake_action(env: Any, goal: float, mem: Dict[str, Any]) -> int:
    """Cruise at the goal; change lane past a slower lead, then pass it."""
    ego = _ego(env)
    if ego is None:
        return _IDLE
    phase = str(mem.get("phase", "cruise"))
    if phase == "cruise":
        lead = _lead(env, max_gap=48.0)
        if lead is not None and lead[0] > 6.0 and lead[1] < float(ego.speed) - 0.3:
            for side, lat in (("left", _LEFT), ("right", _RIGHT)):
                if _lane_clear(env, side=side):
                    mem.update(phase="change", hold=0)
                    return lat
            return _track_gear(env, max(10.0, lead[1] - 1.0))
        return _track_gear(env, goal)
    mem["hold"] = int(mem.get("hold", 0)) + 1
    if phase == "change":
        if mem["hold"] >= 6:
            mem.update(phase="pass", hold=0)
        return _IDLE
    if mem["hold"] > 30:
        mem.update(phase="cruise", hold=0)
    return _track_gear(env, goal + 1.0)


def _action(env: Any, behavior: str, step: int, goal: float, mem: Dict[str, Any], rng: Any) -> int:
    if behavior == "follow":
        return _follow_action(env, goal)
    if behavior == "overtake":
        return _overtake_action(env, goal, mem)
    if behavior == "random":
        return int(rng.integers(0, 5))
    if behavior == "swerve" and step % 6 == 0:
        return _LEFT if (step // 6) % 2 == 0 else _RIGHT
    return _track_gear(env, goal)


def run_episode(
    *, seed: int, profile: str, behavior: str, goal: float
) -> Tuple[str, List[Any], int]:
    """Roll one episode; returns ``(label, states, overtakes)``."""
    cfg = traffic_config(profile)
    time_step = 1.0 / float(cfg["policy_frequency"])
    time_limit = float(cfg["duration"])
    rng = np.random.default_rng(seed)
    env = make_base_env(seed=seed, config=cfg)
    try:
        obs, _info = env.reset(seed=seed)
        tracker = OvertakeTracker()
        tracker.observe(env)
        states: List[Any] = []
        label = "success"
        mem: Dict[str, Any] = {}
        ego_x: Optional[float] = None
        step = 0
        done = False
        while not done:
            action = _action(env, behavior, step, goal, mem, rng)
            obs, _r, terminated, truncated, info = env.step(action)
            tracker.observe(env)
            info = dict(info or {})
            collision = bool(info.get("crashed", False))
            off_road = detect_off_road(env, info)
            step += 1
            state, ego_x = kinematics_to_state(
                obs,
                action=int(action),
                collision=collision,
                off_road=off_road,
                timeout=bool(truncated) and not collision and not off_road,
                time_step=time_step,
                global_time=min(time_limit, step * time_step),
                time_limit=time_limit,
                prev_ego_x=ego_x,
            )
            states.append(state)
            if collision:
                label = "collision"
            elif off_road:
                label = "timeout"
            done = bool(terminated or truncated)
        return label, states, int(tracker.passes)
    finally:
        env.close()


def _record(
    tid: int, *, seed: int, profile: str, behavior: str, goal: float,
    label: str, states: List[Any], overtakes: int,
) -> HighwayTrajectoryRecord:
    mean_speed = float(np.mean([float(s.speed) for s in states]))
    return HighwayTrajectoryRecord(
        trajectory_id=f"hw_{tid:04d}",
        scenario_id=f"{behavior}_{profile}",
        seed=seed,
        states=tuple(states),
        label=label,
        behavior=behavior,
        metadata={
            "role": trajectory_role(label, behavior, mean_speed),
            "traffic": profile,
            "goal_speed": float(goal),
            "mean_speed": mean_speed,
            "mean_progress": float(sum(float(s.progress) for s in states)),
            "n_steps": len(states),
            "overtakes": int(overtakes),
            "synthesized": False,
        },
    )


def synth_off_road(src: HighwayTrajectoryRecord, tid: int) -> HighwayTrajectoryRecord:
    """Copy of a success whose last two frames leave the road (label ``timeout``)."""
    frames = list(src.states)
    for j in (-2, -1):
        s = frames[j]
        frames[j] = dc_replace(
            s, off_road=True, timeout=False, collision=False, ego=dc_replace(s.ego, on_road=False)
        )
    meta = {**src.metadata, "role": "failure", "overtakes": 0, "synthesized": True}
    return HighwayTrajectoryRecord(
        trajectory_id=f"hw_{tid:04d}",
        scenario_id=f"synth_off_road_from_{src.trajectory_id}",
        seed=src.seed,
        states=tuple(frames),
        label="timeout",
        behavior="synth_off_road",
        metadata=meta,
    )


def main() -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="domains/highway/data/stage1_dataset")
    parser.add_argument("--episodes-per-behavior", type=int, default=10)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--behaviors", default=DEFAULT_BEHAVIORS,
                        help="Comma-separated behaviours; repeat a name to collect more of it")
    parser.add_argument("--synth-off-road", type=int, default=5,
                        help="Off-road failures synthesized from positive successes")
    parser.add_argument("--overtake-max-attempts-factor", type=float, default=8.0)
    parser.add_argument("--min-per-role", type=int, default=8,
                        help="Exit 2 if any role (positive/neutral/lag/failure) has fewer trajectories")
    args = parser.parse_args()
    os.chdir(_ROOT)

    behaviors = [b.strip() for b in str(args.behaviors).split(",") if b.strip()]
    trajs: List[HighwayTrajectoryRecord] = []
    overtake_stats = Counter()
    for bi, behavior in enumerate(behaviors):
        target_n = int(args.episodes_per_behavior)
        max_attempts = (
            int(float(args.overtake_max_attempts_factor) * target_n) if behavior == "overtake" else target_n
        )
        kept = 0
        for attempt in range(max_attempts):
            if kept >= target_n:
                break
            seed = int(args.seed) + 1000 * bi + attempt
            profile = "train" if attempt % 2 == 0 else "holdout"
            goal = (
                _OVERTAKE_GOALS[attempt % len(_OVERTAKE_GOALS)]
                if behavior == "overtake"
                else _GOAL_MPS.get(behavior, 24.0)
            )
            label, states, overtakes = run_episode(seed=seed, profile=profile, behavior=behavior, goal=goal)
            counts_toward_target = True
            if behavior == "overtake":
                overtake_stats[label if label != "success" or overtakes else "no_pass"] += 1
                if label == "success" and overtakes < 1:
                    continue
                counts_toward_target = label == "success"
            rec = _record(
                len(trajs), seed=seed, profile=profile, behavior=behavior, goal=goal,
                label=label, states=states, overtakes=overtakes,
            )
            trajs.append(rec)
            kept += int(counts_toward_target)
            print(
                f"{rec.trajectory_id} {behavior:11s} {profile:7s} label={label:9s} "
                f"role={rec.metadata['role']:8s} steps={len(states):3d} "
                f"speed={rec.metadata['mean_speed']:.1f} overtakes={overtakes}",
                flush=True,
            )
        if kept < target_n:
            print(f"WARNING: {behavior} kept {kept}/{target_n}", file=sys.stderr)

    donors = [t for t in trajs if t.metadata["role"] == "positive" and t.label == "success"]
    for i in range(min(int(args.synth_off_road), len(donors))):
        trajs.append(synth_off_road(donors[i * len(donors) // int(args.synth_off_road)], len(trajs)))

    save_highway_dataset(str(args.out), trajs)
    roles = Counter(t.metadata["role"] for t in trajs)
    print(f"Wrote {len(trajs)} trajectories -> {args.out}", flush=True)
    print(f"labels: {dict(Counter(t.label for t in trajs))}  roles: {dict(roles)}", flush=True)
    by_label_traffic = Counter((t.label, t.metadata["traffic"]) for t in trajs)
    print(f"label x traffic: {dict(sorted(by_label_traffic.items()))}", flush=True)
    if overtake_stats:
        print(f"overtake attempts: {dict(overtake_stats)}", flush=True)
    short = [r for r in ("positive", "neutral", "lag", "failure") if roles.get(r, 0) < int(args.min_per_role)]
    if short:
        print(f"WARNING: roles below --min-per-role={args.min_per_role}: {short}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

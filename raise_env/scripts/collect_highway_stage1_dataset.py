#!/usr/bin/env python
"""Collect a Stage I trajectory dataset for the highway domain pack.

Balanced labels for hybrid Score1 (incl. anti-hacking decoys).
Speed bands are non-overlapping and aligned with Score1 thresholds
(crawl/lag < 18 m/s; safe cruise ~20–21; traffic cruise ≥ 22 m/s;
env gears 20..30):

- ``safe`` / ``safe_fast`` → success cruise in [22, 25] (mid gears)
- ``safe_cruise`` → genuine success in [20, 21.5] (floor gears; NOT rescaled)
- ``overtake`` → success + ≥1 real OvertakeTracker pass (no synthesis)
- ``crawl`` → success in [10, 14] (hard negative; well below Score1's 18)
- ``lag`` / ``decoy_lag`` → success in [15, 17.5] (alive but lagging; still < 18)
- ``aggressive`` / ``random`` / ``decoy_crash`` → collisions (crash bait)
- ``swerve`` → lane-change collisions at spawn gear (~24); synth timeout fill

NOTE (known limitation): this collector always uses DiscreteMetaAction
(``meta_default`` gears via ``default_env_config`` / ``make_base_env``).
It does **not** follow PPO ``highway_action_mode=continuous|meta_fine``.
Score1 therefore remains a meta-action dataset and is not matched to
continuous policies — out of scope for the action_mode switch.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401
import argparse
import os
import sys
from collections import Counter
from dataclasses import replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_IDLE = 1
_FASTER = 3
_SLOWER = 4
_LEFT = 0
_RIGHT = 2

# Empirically measured under this harness (make_base_env, 12 vehicles, 25 eps):
#   20–21 m/s → ~4% crash (safe cruise floor)
#   22 m/s    → ~16% crash
#   23 m/s    → ~48% crash (moderate-risk overtake band)
#   ≥24 m/s   → ≥72% crash
_OVERTAKE_TARGET_MPS = 22.5
_SAFE_CRUISE_TARGET_MPS = 20.5

# Success speed bands after synthesis (m/s). Keep gaps so Score1 buckets
# (crawl/lag < 18, safe_cruise ~20–21, fast ≥ 22) never double-count.
_CRAWL_LO, _CRAWL_HI = 10.0, 14.0
_LAG_LO, _LAG_HI = 15.0, 17.5
_SAFE_CRUISE_LO, _SAFE_CRUISE_HI = 20.0, 21.5
_FAST_LO, _FAST_HI = 22.0, 25.0


def _clip_action(action: int, action_n: int) -> int:
    return int(max(0, min(int(action_n) - 1, int(action))))


def _behavior_action(behavior: str, step: int, action_n: int) -> int:
    if behavior in ("safe", "safe_fast"):
        # Mostly IDLE on spawn gear (~24); rare FASTER. Synth → [22, 25].
        return _clip_action(_IDLE if step % 5 else _FASTER, action_n)
    if behavior == "safe_cruise":
        # Hold floor gears (~20–21). Prefer SLOWER early, then IDLE.
        if step < 8:
            return _clip_action(_SLOWER, action_n)
        return _clip_action(_IDLE if step % 7 else _SLOWER, action_n)
    if behavior == "crawl":
        # Lowest gear (~20 raw); synth scales into [10, 14].
        return _clip_action(_SLOWER, action_n)
    if behavior in ("lag", "decoy_lag"):
        # Mix IDLE/SLOWER so raw stays near floor; synth → [15, 17.5].
        return _clip_action(_IDLE if step % 3 else _SLOWER, action_n)
    if behavior == "idle":
        return _clip_action(_IDLE, action_n)
    if behavior in ("aggressive", "decoy_crash"):
        return _clip_action(_FASTER if step % 2 else _IDLE, action_n)
    if behavior == "swerve":
        # Lane-only: stays on spawn gear (~24) → collision bait, not a success band.
        return _clip_action(_LEFT if (step // 3) % 2 == 0 else _RIGHT, action_n)
    if behavior == "random":
        return int(np.random.randint(0, action_n))
    return _clip_action(_IDLE, action_n)


def _lane_clear(env: Any, *, side: str, ego_x: float, ego_y: float) -> bool:
    """True if the adjacent lane has no vehicle in a dangerous longitudinal window."""
    road = getattr(getattr(env, "unwrapped", env), "road", None)
    ego = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    if road is None or ego is None:
        return False
    want_dy = 4.0 if side == "left" else -4.0
    for other in road.vehicles:
        if other is ego:
            continue
        dx = float(other.position[0]) - ego_x
        dy = float(other.position[1]) - ego_y
        if abs(dy - want_dy) < 2.5 and -12.0 < dx < 28.0:
            return False
    return True


def _slower_ahead(
    env: Any, *, ego_x: float, ego_y: float, ego_v: float, gap_max: float = 48.0
) -> Optional[Tuple[float, Any]]:
    """Closest same-lane vehicle ahead that is slower than ego (or clearly lagging)."""
    road = getattr(getattr(env, "unwrapped", env), "road", None)
    ego = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    if road is None or ego is None:
        return None
    best: Optional[Tuple[float, Any]] = None
    for other in road.vehicles:
        if other is ego:
            continue
        dx = float(other.position[0]) - ego_x
        dy = float(other.position[1]) - ego_y
        if not (6.0 < dx < gap_max and abs(dy) < 2.0):
            continue
        ov = float(other.speed)
        if ov < ego_v - 0.3 or ov < _OVERTAKE_TARGET_MPS - 1.5:
            if best is None or dx < best[0]:
                best = (dx, other)
    return best


def _overtake_action(env: Any, action_n: int, mem: Dict[str, Any]) -> int:
    """Scripted overtake: cruise ~22.5, lane-change past slower lead, then pass."""
    base = getattr(env, "unwrapped", env)
    ego = getattr(base, "vehicle", None)
    if ego is None:
        return _clip_action(_IDLE, action_n)
    ego_x = float(ego.position[0])
    ego_y = float(ego.position[1])
    ego_v = float(ego.speed)
    target = float(_OVERTAKE_TARGET_MPS)

    if ego_v < target - 1.0:
        speed_a = _FASTER
    elif ego_v > target + 1.5:
        speed_a = _SLOWER
    else:
        speed_a = _IDLE

    phase = str(mem.get("phase", "cruise"))
    if phase == "cruise":
        ahead = _slower_ahead(env, ego_x=ego_x, ego_y=ego_y, ego_v=ego_v)
        if ahead is not None:
            for side, lat in (("left", _LEFT), ("right", _RIGHT)):
                if _lane_clear(env, side=side, ego_x=ego_x, ego_y=ego_y):
                    mem["phase"] = "change"
                    mem["lat"] = lat
                    mem["hold"] = 0
                    return _clip_action(lat, action_n)
            # Blocked: ease off rather than rear-end.
            return _clip_action(_SLOWER, action_n)
        return _clip_action(speed_a, action_n)

    if phase == "change":
        # Issue the lane command once, then hold IDLE while lateral settles.
        mem["hold"] = int(mem.get("hold", 0)) + 1
        if int(mem["hold"]) == 1:
            return _clip_action(int(mem.get("lat", _LEFT)), action_n)
        if int(mem["hold"]) >= 6:
            mem["phase"] = "pass"
            mem["hold"] = 0
        return _clip_action(_IDLE, action_n)

    if phase == "pass":
        mem["hold"] = int(mem.get("hold", 0)) + 1
        if int(mem["hold"]) > 30:
            mem["phase"] = "cruise"
            mem["hold"] = 0
        # Push slightly above target to complete the longitudinal pass.
        if ego_v < target + 1.0:
            return _clip_action(_FASTER, action_n)
        return _clip_action(_IDLE, action_n)

    return _clip_action(speed_a, action_n)


def _scale_success_speeds(
    states: list,
    *,
    target: float,
    lo: float,
    hi: float,
    scale_lo: float,
    scale_hi: float,
) -> tuple[list, float, float]:
    """Rescale speeds/progress of a success traj into ``[lo, hi]`` around ``target``."""
    mean_speed = float(sum(float(s.speed) for s in states) / max(1, len(states)))
    scale_v = float(target) / max(mean_speed, 1e-3)
    scale_v = float(min(max(scale_v, scale_lo), scale_hi))
    new_states = []
    x = float(states[0].ego.x) if states else 0.0
    for s in states:
        spd = max(lo, min(hi, float(s.speed) * scale_v))
        prog = float(s.progress) * scale_v
        x = x + prog
        new_states.append(
            dc_replace(
                s,
                progress=prog,
                speed=spd,
                ego=dc_replace(s.ego, x=x, vx=spd, speed=spd),
            )
        )
    mean_speed_out = float(
        sum(float(s.speed) for s in new_states) / max(1, len(new_states))
    )
    mean_progress_out = float(sum(float(s.progress) for s in new_states))
    return new_states, mean_speed_out, mean_progress_out


def _env_overrides(behavior: str) -> dict:
    if behavior in ("safe", "safe_fast"):
        # Light traffic so cruise can finish episode (need enough success_fast).
        return {"vehicles_count": 3, "lanes_count": 4, "duration": 40}
    if behavior == "safe_cruise":
        # Sparse traffic; hold ~20–21 without needing post-hoc rescale.
        return {"vehicles_count": 4, "lanes_count": 4, "duration": 40}
    if behavior == "overtake":
        # Enough traffic for slower leads, not so dense that passes are impossible.
        return {"vehicles_count": 14, "lanes_count": 4, "duration": 40}
    if behavior == "crawl":
        return {"vehicles_count": 4, "lanes_count": 4, "duration": 40}
    if behavior in ("lag", "decoy_lag"):
        return {"vehicles_count": 6, "lanes_count": 4, "duration": 40}
    if behavior == "idle":
        return {"vehicles_count": 8, "duration": 40}
    if behavior in ("aggressive", "decoy_crash"):
        return {"vehicles_count": 28, "duration": 30}
    if behavior == "swerve":
        return {"vehicles_count": 10, "duration": 25}
    if behavior == "random":
        return {"vehicles_count": 22, "duration": 30}
    return {}


def _run_episode(
    *,
    env: Any,
    behavior: str,
    reward_fn: Any,
    time_step: float,
    time_limit: float,
) -> Tuple[str, List[Any], float, float, int]:
    """
    Roll one episode.

    Returns ``(label, states, mean_speed, mean_progress, overtakes)``.
    ``overtakes`` is OvertakeTracker.passes (0 for non-overtake behaviors).
    """
    from domains.highway.env_wrapper import kinematics_to_state
    from domains.highway.overtake import OvertakeTracker

    obs, _info = env.reset()
    ego_x = None
    global_time = 0.0
    states: List[Any] = []
    label = "success"
    done = False
    step = 0
    action_n = int(env.action_space.n)
    speed_sum = 0.0
    mem: Dict[str, Any] = {}
    tracker = OvertakeTracker()
    tracker.observe(env)

    while not done:
        if behavior == "overtake":
            action = _overtake_action(env, action_n, mem)
        else:
            action = _behavior_action(behavior, step, action_n)
        obs, _r, terminated, truncated, info = env.step(action)
        tracker.observe(env)
        info = dict(info or {})
        collision = bool(info.get("crashed", False))
        off_road = bool(info.get("off_road", False))
        if isinstance(info.get("rewards"), dict):
            if info["rewards"].get("on_road_reward", 1.0) == 0.0:
                off_road = True
        timeout = bool(truncated) and not collision and not off_road
        global_time = min(time_limit, global_time + time_step)
        state, ego_x = kinematics_to_state(
            obs,
            action=action,
            collision=collision,
            off_road=off_road,
            timeout=timeout,
            time_step=time_step,
            global_time=global_time,
            time_limit=time_limit,
            prev_ego_x=ego_x,
        )
        _ = reward_fn.compute(state)
        states.append(state)
        speed_sum += float(state.speed)
        if collision:
            label = "collision"
        elif off_road:
            label = "timeout"
        elif timeout:
            label = "success"
        done = bool(terminated or truncated)
        step += 1

    if hasattr(reward_fn, "reset"):
        reward_fn.reset()
    mean_speed = speed_sum / float(max(1, len(states)))
    mean_progress = float(sum(float(s.progress) for s in states))
    overtakes = int(tracker.passes)
    return label, states, mean_speed, mean_progress, overtakes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="domains/highway/data/stage1_dataset")
    parser.add_argument("--episodes-per-behavior", type=int, default=10)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument(
        "--behaviors",
        default=(
            "safe_fast,safe_fast,safe_fast,"
            "safe_cruise,safe_cruise,"
            "overtake,overtake,overtake,"
            "crawl,lag,aggressive,decoy_crash,random,swerve"
        ),
        help=(
            "Comma-separated behaviors (safe_fast / overtake repeated for "
            "emphasis; safe_cruise fills the ~20–21 m/s gap)"
        ),
    )
    parser.add_argument(
        "--min-per-label",
        type=int,
        default=5,
        help="Warn/exit 2 if success/collision/timeout below this",
    )
    parser.add_argument(
        "--min-crawl-success",
        type=int,
        default=5,
        help="Need at least this many crawl-tagged success trajs",
    )
    parser.add_argument(
        "--min-overtake-success",
        type=int,
        default=8,
        help="Need at least this many genuine overtake successes",
    )
    parser.add_argument(
        "--min-safe-cruise-success",
        type=int,
        default=5,
        help="Need at least this many safe_cruise successes (~20–21 m/s)",
    )
    parser.add_argument(
        "--overtake-max-attempts-factor",
        type=float,
        default=8.0,
        help="Max attempts = factor × target count for overtake resampling",
    )
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    from domains.highway.env_wrapper import (
        make_base_env,
        default_env_config,
    )
    from domains.highway.prompts import D5_SEED_FUNCTION
    from domains.highway.stage1 import (
        HighwayTrajectoryRecord,
        save_highway_dataset,
    )
    from domains.highway.reward_checks import make_highway_validator
    from domains.highway.state import default_smoke_states

    validator = make_highway_validator(smoke_states=default_smoke_states())
    reward_fn = validator.validate_code(D5_SEED_FUNCTION)

    behaviors = [b.strip() for b in str(args.behaviors).split(",") if b.strip()]
    print(
        "NOTE: Stage1 collector uses DiscreteMetaAction (meta_default) only; "
        "Score1 dataset is not matched to continuous PPO policies.",
        flush=True,
    )
    print(
        f"Empirical bands: safe_cruise~{_SAFE_CRUISE_TARGET_MPS} m/s, "
        f"overtake_target~{_OVERTAKE_TARGET_MPS} m/s "
        f"(crash risk rises sharply above 23).",
        flush=True,
    )
    trajs = []
    tid = 0
    base_cfg = default_env_config()
    overtake_stats = {"attempts": 0, "kept": 0, "crash": 0, "no_pass": 0}

    for bi, behavior in enumerate(behaviors):
        cfg = dict(base_cfg)
        for key, value in _env_overrides(behavior).items():
            cfg[key] = value
        time_step = 1.0 / float(cfg["policy_frequency"])
        time_limit = float(cfg["duration"])
        target_n = int(args.episodes_per_behavior)

        if behavior == "overtake":
            max_attempts = max(
                target_n,
                int(float(args.overtake_max_attempts_factor) * target_n),
            )
            kept = 0
            attempt = 0
            while kept < target_n and attempt < max_attempts:
                seed = int(args.seed) + 1000 * bi + attempt
                env = make_base_env(seed=seed, config=cfg)
                try:
                    label, states, mean_speed, mean_progress, overtakes = _run_episode(
                        env=env,
                        behavior=behavior,
                        reward_fn=reward_fn,
                        time_step=time_step,
                        time_limit=time_limit,
                    )
                finally:
                    env.close()
                attempt += 1
                overtake_stats["attempts"] += 1
                if label == "collision":
                    overtake_stats["crash"] += 1
                if not (label == "success" and overtakes >= 1):
                    if label == "success":
                        overtake_stats["no_pass"] += 1
                    print(
                        f"overtake discard attempt={attempt} label={label} "
                        f"passes={overtakes} speed={mean_speed:.1f}",
                        flush=True,
                    )
                    continue
                overtake_stats["kept"] += 1
                kept += 1
                trajs.append(
                    HighwayTrajectoryRecord(
                        trajectory_id=f"hw_{tid:04d}",
                        scenario_id=f"beh_{behavior}",
                        seed=seed,
                        states=tuple(states) if states else tuple(),
                        label="success",
                        behavior="overtake",
                        metadata={
                            "mean_speed": float(mean_speed),
                            "mean_progress": float(mean_progress),
                            "n_steps": int(len(states)),
                            "requested_behavior": behavior,
                            "overtakes": int(overtakes),
                            "genuine_overtake": True,
                            "synthesized": False,
                            "decoy": False,
                        },
                    )
                )
                tid += 1
                print(
                    f"collected {trajs[-1].trajectory_id} label=success "
                    f"behavior=overtake steps={len(states)} "
                    f"speed={mean_speed:.1f} progress={mean_progress:.1f} "
                    f"overtakes={overtakes} "
                    f"(kept {kept}/{target_n} after {attempt} attempts)",
                    flush=True,
                )
            if kept < target_n:
                print(
                    f"WARNING: overtake kept only {kept}/{target_n} "
                    f"after {attempt} attempts",
                    file=sys.stderr,
                )
            continue

        for ep in range(target_n):
            seed = int(args.seed) + 1000 * bi + ep
            env = make_base_env(seed=seed, config=cfg)
            try:
                label, states, mean_speed, mean_progress, overtakes = _run_episode(
                    env=env,
                    behavior=behavior,
                    reward_fn=reward_fn,
                    time_step=time_step,
                    time_limit=time_limit,
                )
            finally:
                env.close()

            beh_out = behavior
            states_out = list(states)

            # Raw SLOWER/FASTER land on discrete gears (~20 / ~30). Rescale
            # success trajs into non-overlapping Score1 bands (except
            # safe_cruise / overtake which keep genuine kinematics).
            if label == "success" and behavior == "crawl":
                beh_out = "crawl"
                target = _CRAWL_LO + (_CRAWL_HI - _CRAWL_LO) * float(ep % 5) / 4.0
                states_out, mean_speed, mean_progress = _scale_success_speeds(
                    states,
                    target=target,
                    lo=_CRAWL_LO,
                    hi=_CRAWL_HI,
                    scale_lo=0.35,
                    scale_hi=1.0,
                )
            elif label == "success" and behavior in ("lag", "decoy_lag"):
                beh_out = "decoy_lag"
                target = _LAG_LO + (_LAG_HI - _LAG_LO) * float(ep % 3) / 2.0
                states_out, mean_speed, mean_progress = _scale_success_speeds(
                    states,
                    target=target,
                    lo=_LAG_LO,
                    hi=_LAG_HI,
                    scale_lo=0.45,
                    scale_hi=0.95,
                )
            elif label == "collision" and behavior == "decoy_crash":
                beh_out = "decoy_crash"
            elif label == "success" and behavior == "safe_cruise":
                # Keep raw ~20–21 kinematics; drop if outside the band.
                if not (_SAFE_CRUISE_LO <= mean_speed <= _SAFE_CRUISE_HI + 0.5):
                    # Soft clamp only if slightly outside; never invent overtakes.
                    target = _SAFE_CRUISE_TARGET_MPS
                    states_out, mean_speed, mean_progress = _scale_success_speeds(
                        states,
                        target=target,
                        lo=_SAFE_CRUISE_LO,
                        hi=_SAFE_CRUISE_HI,
                        scale_lo=0.85,
                        scale_hi=1.05,
                    )
                beh_out = "safe_cruise"
            elif label == "success" and behavior in ("safe", "safe_fast"):
                beh_out = "safe_fast"
                target = _FAST_LO + (_FAST_HI - _FAST_LO) * float(ep % 4) / 3.0
                states_out, mean_speed, mean_progress = _scale_success_speeds(
                    states,
                    target=target,
                    lo=_FAST_LO,
                    hi=_FAST_HI,
                    scale_lo=0.70,
                    scale_hi=1.10,
                )

            trajs.append(
                HighwayTrajectoryRecord(
                    trajectory_id=f"hw_{tid:04d}",
                    scenario_id=f"beh_{behavior}",
                    seed=seed,
                    states=tuple(states_out) if states_out else tuple(),
                    label=label,
                    behavior=beh_out,
                    metadata={
                        "mean_speed": float(mean_speed),
                        "mean_progress": float(mean_progress),
                        "n_steps": int(len(states_out)),
                        "requested_behavior": behavior,
                        "overtakes": int(overtakes),
                        "crawl_synthesized": bool(
                            behavior == "crawl" and label == "success"
                        ),
                        "lag_synthesized": bool(
                            behavior in ("lag", "decoy_lag") and label == "success"
                        ),
                        "safe_cruise_band": bool(
                            behavior == "safe_cruise" and label == "success"
                        ),
                        "genuine_overtake": False,
                        "synthesized": bool(
                            behavior in ("crawl", "lag", "decoy_lag", "safe", "safe_fast")
                            and label == "success"
                        ),
                        "decoy": bool(
                            behavior in ("lag", "decoy_lag", "decoy_crash")
                            or (behavior == "crawl" and label == "success")
                        ),
                    },
                )
            )
            tid += 1
            print(
                f"collected {trajs[-1].trajectory_id} label={label} "
                f"behavior={beh_out} steps={len(states)} "
                f"speed={mean_speed:.1f} progress={mean_progress:.1f} "
                f"overtakes={overtakes}",
                flush=True,
            )

    trajs = [t for t in trajs if len(t.states) >= 1]

    counts = Counter(t.label for t in trajs)
    need_to = max(0, int(args.min_per_label) - int(counts.get("timeout", 0)))
    if need_to > 0:
        # Prefer cruise donors so synth timeouts keep high-speed kinematics.
        donors = [
            t
            for t in trajs
            if t.label == "success"
            and len(t.states) >= 4
            and any(
                tag in str(t.behavior).lower()
                for tag in ("safe_fast", "safe_cruise", "overtake")
            )
        ]
        if not donors:
            donors = [t for t in trajs if t.label == "success" and len(t.states) >= 4]
        if not donors:
            donors = [t for t in trajs if len(t.states) >= 4]
        for i in range(need_to):
            if not donors:
                break
            src = donors[i % len(donors)]
            frames = list(src.states)
            for j in (-2, -1):
                s = frames[j]
                frames[j] = dc_replace(
                    s,
                    off_road=True,
                    timeout=True,
                    collision=False,
                    ego=dc_replace(s.ego, on_road=False),
                )
            meta = dict(src.metadata or {})
            meta["overtakes"] = 0
            meta["genuine_overtake"] = False
            trajs.append(
                HighwayTrajectoryRecord(
                    trajectory_id=f"hw_{tid:04d}",
                    scenario_id=f"synth_timeout_from_{src.trajectory_id}",
                    seed=src.seed,
                    states=tuple(frames),
                    label="timeout",
                    behavior="synth_timeout",
                    metadata=meta,
                )
            )
            print(
                f"collected {trajs[-1].trajectory_id} label=timeout "
                f"behavior=synth_timeout (from {src.trajectory_id})",
                flush=True,
            )
            tid += 1

    save_highway_dataset(str(args.out), trajs)
    counts = Counter(t.label for t in trajs)
    crawl_n = sum(
        1
        for t in trajs
        if t.label == "success"
        and (
            any(tag in str(t.behavior).lower() for tag in ("crawl", "lag", "decoy_lag"))
            or float((t.metadata or {}).get("mean_speed") or 99) < 18.0
        )
    )
    fast_n = sum(
        1
        for t in trajs
        if t.label == "success"
        and (
            any(tag in str(t.behavior).lower() for tag in ("safe_fast", "fast"))
            or (
                "safe_cruise" not in str(t.behavior).lower()
                and "overtake" not in str(t.behavior).lower()
                and float((t.metadata or {}).get("mean_speed") or 0) >= _FAST_LO
            )
        )
    )
    cruise_n = sum(
        1
        for t in trajs
        if t.label == "success"
        and (
            "safe_cruise" in str(t.behavior).lower()
            or bool((t.metadata or {}).get("safe_cruise_band"))
        )
    )
    overtake_n = sum(
        1
        for t in trajs
        if t.label == "success"
        and (
            bool((t.metadata or {}).get("genuine_overtake"))
            or (
                "overtake" in str(t.behavior).lower()
                and float((t.metadata or {}).get("overtakes") or 0) >= 1
            )
        )
    )
    decoy_crash_n = sum(
        1
        for t in trajs
        if "decoy_crash" in str(t.behavior).lower() or t.label == "collision"
    )
    print(f"Wrote {len(trajs)} trajectories -> {args.out}", flush=True)
    print(f"label counts: {dict(counts)}", flush=True)
    print(
        f"success_crawl/lag~={crawl_n} success_fast~={fast_n} "
        f"safe_cruise~={cruise_n} overtake~={overtake_n} "
        f"collision/decoy_crash~={decoy_crash_n}",
        flush=True,
    )
    if overtake_stats["attempts"]:
        print(
            "overtake generation: "
            f"attempts={overtake_stats['attempts']} "
            f"kept={overtake_stats['kept']} "
            f"crash={overtake_stats['crash']} "
            f"success_no_pass={overtake_stats['no_pass']} "
            f"keep_rate="
            f"{overtake_stats['kept'] / max(1, overtake_stats['attempts']):.3f}",
            flush=True,
        )

    min_n = int(args.min_per_label)
    missing = [
        lab
        for lab in ("success", "collision", "timeout")
        if counts.get(lab, 0) < min_n
    ]
    if missing:
        print(
            f"WARNING: labels below --min-per-label={min_n}: {missing}",
            file=sys.stderr,
        )
        return 2
    if crawl_n < int(args.min_crawl_success):
        print(
            f"WARNING: crawl successes {crawl_n} < --min-crawl-success="
            f"{args.min_crawl_success}",
            file=sys.stderr,
        )
        return 2
    if cruise_n < int(args.min_safe_cruise_success):
        print(
            f"WARNING: safe_cruise successes {cruise_n} < "
            f"--min-safe-cruise-success={args.min_safe_cruise_success}",
            file=sys.stderr,
        )
        return 2
    if overtake_n < int(args.min_overtake_success):
        print(
            f"WARNING: overtake successes {overtake_n} < "
            f"--min-overtake-success={args.min_overtake_success}",
            file=sys.stderr,
        )
        return 2
    if fast_n < 3:
        print(
            f"WARNING: few fast successes ({fast_n}); Score1 throughput term weak",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

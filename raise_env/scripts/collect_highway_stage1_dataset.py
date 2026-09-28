#!/usr/bin/env python
"""Collect a Stage I trajectory dataset for the highway domain pack.

Balanced labels for hybrid Score1 (incl. anti-hacking decoys).
Speed bands are non-overlapping and aligned with Score1 thresholds
(crawl/lag < 18 m/s; traffic cruise ≥ 20 m/s; env gears 20..30):

- ``safe`` / ``safe_fast`` → success cruise in [22, 25] (mid gears)
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

import numpy as np

_IDLE = 1
_FASTER = 3
_SLOWER = 4
_LEFT = 0
_RIGHT = 2


def _clip_action(action: int, action_n: int) -> int:
    return int(max(0, min(int(action_n) - 1, int(action))))


# Success speed bands after synthesis (m/s). Keep gaps so Score1 buckets
# (crawl/lag < 18, fast ≥ 20) never double-count the same traj.
_CRAWL_LO, _CRAWL_HI = 10.0, 14.0
_LAG_LO, _LAG_HI = 15.0, 17.5
_FAST_LO, _FAST_HI = 22.0, 25.0


def _behavior_action(behavior: str, step: int, action_n: int) -> int:
    if behavior in ("safe", "safe_fast"):
        # Mostly IDLE on spawn gear (~24); rare FASTER. Synth → [22, 25].
        return _clip_action(_IDLE if step % 5 else _FASTER, action_n)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="domains/highway/data/stage1_dataset")
    parser.add_argument("--episodes-per-behavior", type=int, default=10)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument(
        "--behaviors",
        default="safe_fast,safe_fast,safe_fast,crawl,lag,aggressive,decoy_crash,random,swerve",
        help="Comma-separated behaviors (safe_fast repeated for more cruise successes)",
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
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    from domains.highway.env_wrapper import (
        kinematics_to_state,
        make_base_env,
        default_env_config,
    )
    from domains.highway.prompts import D5_SEED_FUNCTION
    from domains.highway.stage1 import (
        HighwayTrajectoryRecord,
        save_highway_dataset,
    )
    from raise_core.sandbox import RewardValidator
    from domains.highway.state import default_smoke_states

    validator = RewardValidator(smoke_states=default_smoke_states())
    reward_fn = validator.validate_code(D5_SEED_FUNCTION)

    behaviors = [b.strip() for b in str(args.behaviors).split(",") if b.strip()]
    print(
        "NOTE: Stage1 collector uses DiscreteMetaAction (meta_default) only; "
        "Score1 dataset is not matched to continuous PPO policies.",
        flush=True,
    )
    trajs = []
    tid = 0
    base_cfg = default_env_config()

    for bi, behavior in enumerate(behaviors):
        cfg = dict(base_cfg)
        for key, value in _env_overrides(behavior).items():
            cfg[key] = value
        time_step = 1.0 / float(cfg["policy_frequency"])
        time_limit = float(cfg["duration"])

        for ep in range(int(args.episodes_per_behavior)):
            seed = int(args.seed) + 1000 * bi + ep
            env = make_base_env(seed=seed, config=cfg)
            obs, _info = env.reset(seed=seed)
            ego_x = None
            global_time = 0.0
            states = []
            label = "success"
            done = False
            step = 0
            action_n = int(env.action_space.n)
            speed_sum = 0.0
            while not done:
                action = _behavior_action(behavior, step, action_n)
                obs, _r, terminated, truncated, info = env.step(action)
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
            env.close()
            if hasattr(reward_fn, "reset"):
                reward_fn.reset()

            mean_speed = speed_sum / float(max(1, len(states)))
            mean_progress = float(sum(float(s.progress) for s in states))
            beh_out = behavior
            states_out = list(states)

            # Raw SLOWER/FASTER land on discrete gears (~20 / ~30). Rescale
            # success trajs into non-overlapping Score1 bands.
            if label == "success" and behavior == "crawl":
                # Hard negative: well below Score1 crawl threshold (18).
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
                # Soft decoy: still < 18 so Score1 lag/crawl path fires; above crawl.
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
            elif label == "success" and behavior in ("safe", "safe_fast"):
                # Positive cruise: mid-band [22, 25] (above Score1 fast floor of 20).
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
            # Do not auto-relabel other successes (e.g. rare swerve) as safe_fast.
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
                        "crawl_synthesized": bool(behavior == "crawl" and label == "success"),
                        "lag_synthesized": bool(
                            behavior in ("lag", "decoy_lag") and label == "success"
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
                f"speed={mean_speed:.1f} progress={mean_progress:.1f}"
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
            and "safe" in str(t.behavior).lower()
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
            trajs.append(
                HighwayTrajectoryRecord(
                    trajectory_id=f"hw_{tid:04d}",
                    scenario_id=f"synth_timeout_from_{src.trajectory_id}",
                    seed=src.seed,
                    states=tuple(frames),
                    label="timeout",
                    behavior="synth_timeout",
                    metadata=dict(src.metadata or {}),
                )
            )
            print(
                f"collected {trajs[-1].trajectory_id} label=timeout "
                f"behavior=synth_timeout (from {src.trajectory_id})"
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
            any(tag in str(t.behavior).lower() for tag in ("safe", "fast"))
            or float((t.metadata or {}).get("mean_speed") or 0) >= _FAST_LO
        )
    )
    decoy_crash_n = sum(
        1
        for t in trajs
        if "decoy_crash" in str(t.behavior).lower() or t.label == "collision"
    )
    print(f"Wrote {len(trajs)} trajectories -> {args.out}")
    print(f"label counts: {dict(counts)}")
    print(
        f"success_crawl/lag~={crawl_n} success_fast~={fast_n} "
        f"collision/decoy_crash~={decoy_crash_n}"
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
    if fast_n < 3:
        print(
            f"WARNING: few fast successes ({fast_n}); Score1 throughput term weak",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Collect a Stage I trajectory dataset for the highway domain pack.

Produces a mix of ``success`` / ``collision`` / ``timeout`` labels. Dense
default traffic makes naive random/idle policies crash-only; this collector
uses per-behavior env + action presets so Score1 has preference diversity.
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

import numpy as np

# DiscreteMetaAction indices (highway-env): LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
_IDLE = 1
_FASTER = 3
_SLOWER = 4
_LEFT = 0
_RIGHT = 2


def _clip_action(action: int, action_n: int) -> int:
    return int(max(0, min(int(action_n) - 1, int(action))))


def _behavior_action(behavior: str, step: int, action_n: int) -> int:
    if behavior == "safe":
        # Prefer slow cruise; rare idle — maximize survival to timeout=success.
        return _clip_action(_SLOWER if step % 7 else _IDLE, action_n)
    if behavior == "idle":
        return _clip_action(_IDLE, action_n)
    if behavior == "aggressive":
        return _clip_action(_FASTER if step % 3 else _IDLE, action_n)
    if behavior == "swerve":
        # Alternate lane changes → often off-road / unstable → timeout label.
        return _clip_action(_LEFT if (step // 4) % 2 == 0 else _RIGHT, action_n)
    if behavior == "random":
        return int(np.random.randint(0, action_n))
    return _clip_action(_IDLE, action_n)


def _env_overrides(behavior: str) -> dict:
    """Ease traffic for success; densify for collisions."""
    if behavior == "safe":
        return {"vehicles_count": 6, "lanes_count": 4, "duration": 40}
    if behavior == "idle":
        return {"vehicles_count": 10, "duration": 40}
    if behavior == "aggressive":
        return {"vehicles_count": 25, "duration": 30}
    if behavior == "swerve":
        return {"vehicles_count": 8, "duration": 25}
    if behavior == "random":
        return {"vehicles_count": 20, "duration": 30}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="domains/highway/data/stage1_dataset")
    parser.add_argument("--episodes-per-behavior", type=int, default=10)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument(
        "--behaviors",
        default="safe,aggressive,swerve,random",
        help="Comma-separated behavior names (default targets label diversity)",
    )
    parser.add_argument(
        "--min-per-label",
        type=int,
        default=5,
        help="Warn (and exit 2) if any of success/collision/timeout is below this",
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
                if collision:
                    label = "collision"
                elif off_road:
                    label = "timeout"
                elif timeout:
                    # Survived full horizon without crash/off-road → success for Score1.
                    label = "success"
                done = bool(terminated or truncated)
                step += 1
            env.close()
            if hasattr(reward_fn, "reset"):
                reward_fn.reset()
            trajs.append(
                HighwayTrajectoryRecord(
                    trajectory_id=f"hw_{tid:04d}",
                    scenario_id=f"beh_{behavior}",
                    seed=seed,
                    states=tuple(states) if states else tuple(),
                    label=label,
                    behavior=behavior,
                )
            )
            tid += 1
            print(
                f"collected {trajs[-1].trajectory_id} label={label} "
                f"behavior={behavior} steps={len(states)}"
            )

    trajs = [t for t in trajs if len(t.states) >= 1]

    from collections import Counter
    from dataclasses import replace as dc_replace

    counts = Counter(t.label for t in trajs)
    # DiscreteMetaAction rarely leaves the road; synthesize timeout trajs for Score1
    # preference diversity when rollouts yielded none.
    need_to = max(0, int(args.min_per_label) - int(counts.get("timeout", 0)))
    if need_to > 0:
        donors = [t for t in trajs if t.label == "success" and len(t.states) >= 4]
        if not donors:
            donors = [t for t in trajs if len(t.states) >= 4]
        for i in range(need_to):
            if not donors:
                break
            src = donors[i % len(donors)]
            frames = list(src.states)
            # Mark the last 2 frames as off-road / timeout terminal.
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
                )
            )
            print(
                f"collected {trajs[-1].trajectory_id} label=timeout "
                f"behavior=synth_timeout (from {src.trajectory_id})"
            )
            tid += 1

    save_highway_dataset(str(args.out), trajs)

    counts = Counter(t.label for t in trajs)
    print(f"Wrote {len(trajs)} trajectories -> {args.out}")
    print(f"label counts: {dict(counts)}")
    min_n = int(args.min_per_label)
    missing = [
        lab
        for lab in ("success", "collision", "timeout")
        if counts.get(lab, 0) < min_n
    ]
    if missing:
        print(
            f"WARNING: labels below --min-per-label={min_n}: {missing}. "
            "Re-run with more episodes or different --behaviors.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

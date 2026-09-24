#!/usr/bin/env python
"""Collect a small Stage I trajectory dataset for the highway domain pack."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _behavior_action(behavior: str, step: int, action_n: int) -> int:
    # DiscreteMetaAction: typically LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
    idle = min(1, action_n - 1)
    faster = min(3, action_n - 1) if action_n > 3 else idle
    slower = min(4, action_n - 1) if action_n > 4 else idle
    if behavior == "idle":
        return idle
    if behavior == "aggressive":
        return faster if step % 5 else idle
    if behavior == "random":
        return int(np.random.randint(0, action_n))
    return idle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/highway_stage1_dataset")
    parser.add_argument("--episodes-per-behavior", type=int, default=8)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument(
        "--behaviors",
        default="random,aggressive,idle",
        help="Comma-separated behavior names",
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
    cfg = default_env_config()
    time_step = 1.0 / float(cfg["policy_frequency"])
    time_limit = float(cfg["duration"])

    for bi, behavior in enumerate(behaviors):
        for ep in range(int(args.episodes_per_behavior)):
            seed = int(args.seed) + 1000 * bi + ep
            env = make_base_env(seed=seed, config=cfg)
            obs, _info = env.reset(seed=seed)
            ego_x = 0.0
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
                # Still call seed reward so shaping paths are exercised (ignored).
                _ = reward_fn.compute(state)
                states.append(state)
                if collision:
                    label = "collision"
                elif off_road:
                    label = "timeout"
                elif timeout:
                    label = "timeout"
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
            print(f"collected {trajs[-1].trajectory_id} label={label} behavior={behavior}")

    # Filter empty (should not happen)
    trajs = [t for t in trajs if len(t.states) >= 1]
    save_highway_dataset(str(args.out), trajs)
    print(f"Wrote {len(trajs)} trajectories → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

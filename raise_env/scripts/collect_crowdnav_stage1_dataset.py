#!/usr/bin/env python
"""Collect the Stage I trajectory dataset for the CrowdNav domain pack.

Every trajectory is a genuine rollout of the vendored upstream simulator
(vita-epfl/CrowdNav ``CrowdSim``, unmodified ``env.config``) on upstream
``train``-phase scenarios; labels are the upstream episode outcomes
(``ReachGoal`` -> success, ``Collision``, ``Timeout``). Every behaviour runs
the same scenarios, so the scenario does not separate the labels.

Robot behaviours (holonomic; Score1 re-expresses actions for unicycle runs):

- ``orca``          upstream ORCA policy (sees the humans)
- ``orca_cautious`` upstream ORCA with ``safety_space`` 0.15 m
- ``linear``        upstream Linear policy: straight to the goal at v_pref
- ``linear_slow``   straight to the goal at 0.3 v_pref (freezing-like)
- ``stop``          stands still (the "freezing robot")
- ``random``        uniformly random CADRL action each step (seeded)
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401
import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from typing import Any, Callable, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
from crowd_sim.envs.policy.linear import Linear  # noqa: E402
from crowd_sim.envs.policy.orca import ORCA  # noqa: E402
from crowd_sim.envs.utils.action import ActionXY  # noqa: E402
from crowd_sim.envs.utils.state import JointState  # noqa: E402

from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv, eval_cases  # noqa: E402
from domains.crowdnav.settings import CrowdNavSettings  # noqa: E402
from domains.crowdnav.stage1 import (  # noqa: E402
    DEFAULT_STAGE1_DATASET,
    CrowdNavTrajectoryRecord,
    save_crowdnav_dataset,
)
from raise_core.console import ensure_utf8_stdio  # noqa: E402

UPSTREAM_COMMIT = "20d678085c06831e658a65b9e20c8bb6f6ecdc10"
BEHAVIOURS = ("orca", "orca_cautious", "linear", "linear_slow", "stop", "random")
_SLOW_FRACTION = 0.3
_CAUTIOUS_SAFETY_SPACE = 0.15


def _joint_state(env: RewardInjectedCrowdNavEnv) -> JointState:
    return JointState(
        env.sim.robot.get_full_state(), [h.get_observable_state() for h in env.sim.humans]
    )


def _make_controller(
    behaviour: str, env: RewardInjectedCrowdNavEnv, rng: np.random.Generator
) -> Callable[[], Any]:
    """Per-step upstream action for ``behaviour`` (called after ``env.reset``)."""
    if behaviour in ("orca", "orca_cautious"):
        policy = ORCA()
        policy.time_step = env.time_step
        if behaviour == "orca_cautious":
            policy.safety_space = _CAUTIOUS_SAFETY_SPACE
        return lambda: policy.predict(_joint_state(env))
    if behaviour in ("linear", "linear_slow"):
        policy = Linear()
        scale = 1.0 if behaviour == "linear" else _SLOW_FRACTION

        def _linear() -> Any:
            a = policy.predict(_joint_state(env))
            return ActionXY(a.vx * scale, a.vy * scale)

        return _linear
    if behaviour == "stop":
        return lambda: ActionXY(0.0, 0.0)
    if behaviour == "random":
        table = env.action_table
        return lambda: ActionXY(*table[int(rng.integers(len(table)))])
    raise ValueError(f"unknown behaviour {behaviour!r}")


def collect(*, episodes: int, seed: int, behaviours: List[str]) -> List[CrowdNavTrajectoryRecord]:
    settings = CrowdNavSettings(kinematics="holonomic", action_mode="discrete")
    env = RewardInjectedCrowdNavEnv(None, phase="train", settings=settings)
    cases = eval_cases(int(seed), int(episodes), int(env.sim.case_size["train"]))
    rng = np.random.default_rng(int(seed))
    out: List[CrowdNavTrajectoryRecord] = []
    for behaviour in behaviours:
        for ep, case in enumerate(cases):
            env.reset(options={"case": case})
            act = _make_controller(behaviour, env, rng)
            states = []
            native: List[float] = []
            done = False
            info: Dict[str, Any] = {}
            while not done:
                _obs, reward, terminated, truncated, info = env.step_upstream(act())
                states.append(env.last_state)
                native.append(float(reward))
                done = bool(terminated or truncated)
            label = str(info["raise_outcome"])
            path = sum(
                math.hypot(b.robot.px - a.robot.px, b.robot.py - a.robot.py)
                for a, b in zip(states, states[1:])
            )
            out.append(
                CrowdNavTrajectoryRecord(
                    trajectory_id=f"{behaviour}_{ep:03d}",
                    scenario_id=f"train_case_{case}",
                    seed=int(case),
                    states=tuple(states),
                    label=label,
                    behavior=behaviour,
                    metadata={
                        "phase": "train",
                        "case": int(case),
                        "native_return": float(sum(native)),
                        "nav_time": float(states[-1].global_time),
                        "path_length": float(path),
                        "min_separation": float(min(s.min_separation for s in states)),
                    },
                )
            )
    env.close()
    return out


def main(argv: List[str] | None = None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_STAGE1_DATASET, help="dataset directory")
    parser.add_argument("--episodes", type=int, default=40, help="episodes per behaviour")
    parser.add_argument("--seed", type=int, default=7, help="scenario selection seed")
    parser.add_argument(
        "--behaviours", default=",".join(BEHAVIOURS), help=f"comma list from {BEHAVIOURS}"
    )
    args = parser.parse_args(argv)
    behaviours = [b.strip() for b in str(args.behaviours).split(",") if b.strip()]
    unknown = [b for b in behaviours if b not in BEHAVIOURS]
    if unknown:
        parser.error(f"unknown behaviours {unknown}")

    trajs = collect(episodes=int(args.episodes), seed=int(args.seed), behaviours=behaviours)
    save_crowdnav_dataset(
        args.out,
        trajs,
        extra_manifest={
            "simulator": "vita-epfl/CrowdNav crowd_sim",
            "upstream_commit": UPSTREAM_COMMIT,
            "env_config": "third_party/crowdnav/configs/env.config",
            "phase": "train",
            "kinematics": "holonomic",
            "seed": int(args.seed),
            "episodes_per_behaviour": int(args.episodes),
            "behaviours": behaviours,
        },
    )
    by_behaviour: Dict[str, Counter] = {}
    for t in trajs:
        by_behaviour.setdefault(t.behavior, Counter())[t.label] += 1
    print(f"Wrote {len(trajs)} trajectories ({sum(t.length for t in trajs)} frames) to {args.out}")
    for behaviour, counts in by_behaviour.items():
        print(f"  {behaviour:14s} " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

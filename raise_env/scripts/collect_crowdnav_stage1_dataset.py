#!/usr/bin/env python
"""Collect the Stage I trajectory dataset for the CrowdNav domain pack.

Every trajectory is a genuine rollout of the vendored upstream simulator
(Shuijing725/CrowdNav++ ``CrowdSimVarNum``) on ``train``-phase scenarios; by
default it uses the vendored Python config. An alternative CrowdNav++ Python
config can be supplied with ``--env-config``. Labels are the native wrapper
outcomes (``success``, ``collision``, ``timeout``). Every behaviour runs the
same scenarios, so the scenario does not separate the labels.

Robot behaviours:

- ``orca``            upstream CrowdNav++ ORCA (``safety_space`` 0)
- ``orca_ss005``      ORCA with ``safety_space`` 0.05 m
- ``orca_ss010``      ORCA with ``safety_space`` 0.10 m
- ``orca_cautious``   ORCA with ``safety_space`` 0.15 m
- ``orca_ss025``      ORCA with ``safety_space`` 0.25 m
- ``orca_slow08``     ORCA at 0.8 ``v_pref`` (slower successes)
- ``orca_slow06``     ORCA at 0.6 ``v_pref``
- ``orca_reckless``   ORCA with ``time_horizon`` 1.0 s (reacts late; close successes)
- ``linear``          straight to the goal at ``v_pref``
- ``linear_slow``     straight to the goal at 0.3 ``v_pref``
- ``stop``            stands still (the "freezing robot")
- ``random``          uniformly sampled valid control each step (seeded)
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
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

from domains.crowdnav.env_wrapper import (  # noqa: E402
    RewardInjectedCrowdNavEnv,
    eval_cases,
    load_env_config,
)
from crowd_sim.envs.utils.action import ActionRot, ActionXY  # noqa: E402,F401
from domains.crowdnav.reference_controllers import (  # noqa: E402
    BEHAVIOURS,
    joint_state as _joint_state,  # noqa: F401
    make_controller,
)
from domains.crowdnav.settings import (  # noqa: E402
    DEFAULT_ENV_CONFIG_PATH,
    CrowdNavSettings,
    UNICYCLE_ROTATION_DELTA_MAX,  # noqa: F401
    UNICYCLE_SPEED_DELTA_MAX,  # noqa: F401
    UNICYCLE_SPEED_DELTA_MIN,  # noqa: F401
)

_make_controller = make_controller
from domains.crowdnav.stage1 import (  # noqa: E402
    DEFAULT_STAGE1_DATASET,
    CrowdNavTrajectoryRecord,
    save_crowdnav_dataset,
)
from raise_config import CONFIG  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402

UPSTREAM_COMMIT = "390773137be04ed14e27620dc5fd7c5e1a5b1f62"
_ORCA_SPECS = {
    k: {"safety_space": s, "speed_scale": v, "time_horizon": th}
    for k, (s, v, th) in (
        ("orca", (0.0, 1.0, None)),
        ("orca_ss005", (0.05, 1.0, None)),
        ("orca_ss010", (0.10, 1.0, None)),
        ("orca_cautious", (0.15, 1.0, None)),
        ("orca_ss025", (0.25, 1.0, None)),
        ("orca_slow08", (0.0, 0.8, None)),
        ("orca_slow06", (0.0, 0.6, None)),
        ("orca_reckless", (0.0, 1.0, 1.0)),
    )
}


def collect(
    *,
    episodes: int,
    seed: int,
    behaviours: List[str],
    env_config_path: str | None = None,
    kinematics: str = "holonomic",
) -> List[CrowdNavTrajectoryRecord]:
    settings = CrowdNavSettings(
        kinematics=kinematics,
        action_mode="discrete" if kinematics == "holonomic" else "continuous",
        env_config_path=env_config_path,
    )
    env = RewardInjectedCrowdNavEnv(None, phase="train", settings=settings)
    cases = eval_cases(int(seed), int(episodes), int(env.sim.case_size["train"]))
    rng = np.random.default_rng(int(seed))
    out: List[CrowdNavTrajectoryRecord] = []
    for behaviour in behaviours:
        for ep, case in enumerate(cases):
            env.reset(options={"case": case})
            act = make_controller(behaviour, env, rng)
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
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(CONFIG.budget.stage1.collect_episodes),
        help="episodes per behaviour",
    )
    parser.add_argument("--seed", type=int, default=7, help="scenario selection seed")
    parser.add_argument(
        "--env-config",
        default=None,
        help="CrowdNav++ Python config file (default: vendored config.py)",
    )
    parser.add_argument(
        "--kinematics",
        choices=("holonomic", "unicycle"),
        default=str(CONFIG.budget.stage1.collect_kinematics),
        help="robot kinematics for collected rollouts (default from raise_config)",
    )
    parser.add_argument(
        "--behaviours", default=",".join(BEHAVIOURS), help=f"comma list from {BEHAVIOURS}"
    )
    args = parser.parse_args(argv)
    behaviours = [b.strip() for b in str(args.behaviours).split(",") if b.strip()]
    unknown = [b for b in behaviours if b not in BEHAVIOURS]
    if unknown:
        parser.error(f"unknown behaviours {unknown}")

    trajs = collect(
        episodes=int(args.episodes),
        seed=int(args.seed),
        behaviours=behaviours,
        env_config_path=args.env_config,
        kinematics=args.kinematics,
    )
    env_config = load_env_config(args.env_config)
    human_num = int(env_config.sim.human_num)
    human_num_range = int(env_config.sim.human_num_range)
    save_crowdnav_dataset(
        args.out,
        trajs,
        extra_manifest={
            "simulator": "Shuijing725/CrowdNav_Prediction_AttnGraph CrowdSimVarNum",
            "upstream_commit": UPSTREAM_COMMIT,
            "env_config": args.env_config or _os.path.relpath(DEFAULT_ENV_CONFIG_PATH, _ROOT),
            "phase": "train",
            "kinematics": args.kinematics,
            "environment": {
                "time_limit_s": float(env_config.env.time_limit),
                "time_step_s": float(env_config.env.time_step),
                "human_num": human_num,
                "human_num_range": human_num_range,
                "min_human_num": max(1, human_num - human_num_range),
                "max_human_num": human_num + human_num_range,
                "robot_visible_to_humans": bool(env_config.robot.visible),
                "predict_method": str(getattr(env_config.sim, "predict_method", "none")),
            },
            "seed": int(args.seed),
            "episodes_per_behaviour": int(args.episodes),
            "behaviours": behaviours,
            "orca_specs": _ORCA_SPECS,

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
    from raise_core.script_utils import bootstrap_raise_env

    bootstrap_raise_env()
    sys.exit(main())

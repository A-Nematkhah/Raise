#!/usr/bin/env python
"""
Collect the Stage I analytical dataset once; reuse across all generations.

Default: M=100 scenarios × N_traj=10 **diverse** behaviors on the *same*
scenario layout (paper only specifies N_traj=10 diverse trajectories — not
this mix). Our schedule (not from the paper):
  1× ORCA clean, 1× SF clean,
  2× ORCA + Gaussian noise (two std levels),
  2× SF + Gaussian noise (two std levels),
  2× high-noise ORCA/SF,
  2× random-action
so Success / Other / Fail are represented and duplicate deterministic
rollouts are avoided.

Persists RewardState sequences only (no env reward scalars). Default format is
a compact gzip+pickle archive ``data/stage1_dataset/stage1_dataset.npz``;
``--format jsonl`` writes ``scenario_{j:03d}.jsonl`` instead.

Examples::

    python scripts/collect_stage1_dataset.py
    python scripts/collect_stage1_dataset.py --n-scenarios 3 --n-traj 4 --out data/stage1_smoke
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


class _RecordingReward:
    """Wrap an inner RewardFunction; record RewardState, ignore returned scalars for scoring."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.states: List[Any] = []

    def reset(self) -> None:
        self.states = []
        self.inner.reset()

    def sync_potential(self, potential: float) -> None:
        if hasattr(self.inner, "sync_potential"):
            self.inner.sync_potential(potential)

    def compute(self, state: Any) -> float:
        self.states.append(state)
        return float(self.inner.compute(state))


def _make_env(
    seed: int,
    *,
    human_num: int = 20,
    randomization_regime: str = "without_random",
):
    import gym
    import crowd_sim  # noqa: F401 — register envs before policy imports
    from crowd_nav.configs.config import Config
    from crowd_nav.reward_search.regime import apply_regime_to_config
    from crowd_nav.reward_search.state import LegacyReward

    cfg = Config()
    # Stage I collect: ORCA/SF rollouts → RewardState only; no GST wrapper.
    apply_regime_to_config(
        cfg,
        randomization_regime,
        predict_method="none",
        entry_point="collect_stage1_dataset",
    )
    cfg.sim.human_num = int(human_num)
    cfg.sim.human_num_range = 0
    cfg.robot.policy = "orca"

    env = gym.make("CrowdSimVarNum-v0")
    env.configure(cfg)
    env.thisSeed = seed
    env.nenv = 1
    env.phase = "train"

    # Import policy_factory only after CrowdSim finished loading (avoids circular import).
    from crowd_nav.policy.policy_factory import policy_factory

    policy = policy_factory["orca"](cfg)
    policy.time_step = env.time_step
    env.robot.policy = policy

    legacy = LegacyReward.from_crowd_sim_var_num(env)
    recorder = _RecordingReward(legacy)
    env.set_reward_fn(recorder)
    return env, cfg, recorder


def _set_robot_policy(env, cfg, name: str) -> None:
    from crowd_nav.policy.policy_factory import policy_factory

    policy = policy_factory[name](cfg)
    policy.time_step = env.time_step
    env.robot.policy = policy


def _reset_scenario(env, scenario_id: int, base_seed: int) -> Any:
    env.thisSeed = int(base_seed)
    env.phase = "train"
    env.case_counter["train"] = int(scenario_id)
    return env.reset(phase="train")


def _label_from_info(info: Any) -> str:
    from crowd_sim.envs.utils.info import Collision, ReachGoal, Timeout

    raw = info["info"] if isinstance(info, dict) and "info" in info else info
    if isinstance(raw, ReachGoal):
        return "success"
    if isinstance(raw, Collision):
        return "collision"
    if isinstance(raw, Timeout):
        return "timeout"
    # Fallback from last RewardState flags if needed
    return "timeout"


class _ExternalActionPolicy:
    """
    Lets ``CrowdSimVarNum.step`` accept an externally supplied ActionXY.

    When ``robot.policy.name`` is ORCA/social_force, step ignores the action
    argument and re-queries the planner. Swapping to this policy forces the
    clip_action path used for learned policies.
    """

    name = "external_collect"

    def __init__(self) -> None:
        self.time_step = None

    def clip_action(self, action: Any, v_pref: float) -> Any:
        from crowd_sim.envs.utils.action import ActionXY

        if hasattr(action, "vx") and hasattr(action, "vy"):
            vx, vy = float(action.vx), float(action.vy)
            speed = (vx * vx + vy * vy) ** 0.5
            if speed > float(v_pref) and speed > 1e-8:
                s = float(v_pref) / speed
                return ActionXY(vx * s, vy * s)
            return ActionXY(vx, vy)
        return action


def _step_with_external_action(env, action: Any):
    old = env.robot.policy
    env.robot.policy = _ExternalActionPolicy()
    env.robot.policy.time_step = env.time_step
    try:
        return env.step(action)
    finally:
        env.robot.policy = old


def _rollout(
    env,
    recorder: _RecordingReward,
    *,
    behavior: str,
    noise_std: float = 0.0,
    random_action: bool = False,
    max_steps: int = 400,
) -> Tuple[List[Any], str]:
    from crowd_sim.envs.utils.action import ActionXY

    recorder.reset()
    if hasattr(recorder.inner, "sync_potential") and getattr(env, "potential", None) is not None:
        recorder.sync_potential(env.potential)

    done = False
    steps = 0
    info: Any = {"info": None}
    while not done and steps < max_steps:
        steps += 1
        if random_action:
            v = float(env.robot.v_pref)
            action = ActionXY(
                float(np.random.uniform(-v, v)),
                float(np.random.uniform(-v, v)),
            )
            _ob, _r, done, info = _step_with_external_action(env, action)
        elif noise_std > 0.0:
            human_states = np.copy(env.last_human_states)
            base = env.robot.act(human_states.tolist())
            action = ActionXY(
                float(base.vx + np.random.normal(0.0, noise_std)),
                float(base.vy + np.random.normal(0.0, noise_std)),
            )
            _ob, _r, done, info = _step_with_external_action(env, action)
        else:
            # ORCA / SF path: step ignores the tensor and uses robot.act.
            _ob, _r, done, info = env.step(ActionXY(0.0, 0.0))

    label = _label_from_info(info)
    if not recorder.states and hasattr(env, "humans"):
        raise RuntimeError(f"behavior={behavior} produced zero RewardState frames")
    return list(recorder.states), label


def build_stage1_traj_schedule(
    n_traj: int,
    noise_stds: Tuple[float, float] = (0.25, 0.55),
) -> List[Tuple[str, float, bool]]:
    """
    Behavior schedule for one scenario (policy_name, noise_std, is_random).

    Not from the paper (which only says N_traj=10 diverse). Ensures distinct
    (policy, noise, random) slots so deterministic ORCA/SF are not triplicated.
    """
    n0 = float(noise_stds[0])
    n1 = float(noise_stds[1])
    full_schedule: List[Tuple[str, float, bool]] = [
        ("orca", 0.0, False),
        ("social_force", 0.0, False),
        ("orca", n0, False),
        ("social_force", n0, False),
        ("orca", n1, False),
        ("social_force", n1, False),
        ("orca", max(n1 * 1.5, n1 + 0.1), False),
        ("social_force", max(n1 * 1.5, n1 + 0.1), False),
        ("random", 0.0, True),
        ("random", 0.0, True),
    ]
    if n_traj >= len(full_schedule):
        schedule = list(full_schedule)
        while len(schedule) < n_traj:
            schedule.append(full_schedule[len(schedule) % len(full_schedule)])
        return schedule
    prefer = [0, 1, 8, 9, 2, 3, 4, 5, 6, 7]
    return [full_schedule[i] for i in prefer[:n_traj]]


def collect_dataset(
    *,
    n_scenarios: int = 100,
    n_traj: int = 10,
    base_seed: int = 425,
    out_dir: str = "data/stage1_dataset",
    noise_stds: Tuple[float, float] = (0.25, 0.55),
    human_num: int = 20,
    fmt: str = "npz",
    randomization_regime: str = "without_random",
) -> None:
    # Config.get_args() parses sys.argv at import — isolate from pytest/CLI.
    saved_argv = list(sys.argv)
    sys.argv = [sys.argv[0], "--no-cuda", "--num-processes", "1", "--seed", str(base_seed)]
    try:
        _collect_dataset_impl(
            n_scenarios=n_scenarios,
            n_traj=n_traj,
            base_seed=base_seed,
            out_dir=out_dir,
            noise_stds=noise_stds,
            human_num=human_num,
            fmt=fmt,
            randomization_regime=randomization_regime,
        )
    finally:
        sys.argv = saved_argv


def _collect_dataset_impl(
    *,
    n_scenarios: int,
    n_traj: int,
    base_seed: int,
    out_dir: str,
    noise_stds: Tuple[float, float],
    human_num: int,
    fmt: str,
    randomization_regime: str,
) -> None:
    from crowd_nav.reward_search.dataset import TrajectoryRecord, save_stage1_dataset

    if n_traj != 10:
        logging.warning(
            "Paper Table 4 uses N_traj=10 (diversity unspecified); got %d. "
            "Our default mix is ORCA/SF/noise/random — not claimed as paper text.",
            n_traj,
        )

    env, cfg, recorder = _make_env(
        base_seed,
        human_num=human_num,
        randomization_regime=randomization_regime,
    )
    logging.info(
        "Stage I collect regime=%s randomize_attributes=%s random_goal_changing=%s",
        randomization_regime,
        cfg.env.randomize_attributes,
        cfg.humans.random_goal_changing,
    )
    dataset = {}

    # Ours (not paper): one clean ORCA/SF + graduated noise + random.
    schedule = build_stage1_traj_schedule(n_traj, noise_stds)

    from crowd_nav.reward_search import console
    import time as _time

    t0 = _time.perf_counter()
    console.status(
        f"collecting Stage I dataset: M={n_scenarios} scenarios × "
        f"N_traj={len(schedule)} → {out_dir}",
        stage="Stage I collect",
    )
    scenario_iter = console.progress(
        range(n_scenarios),
        total=n_scenarios,
        desc="[Stage I collect] scenarios",
        unit="sc",
    )
    for j in scenario_iter:
        sid = f"scenario_{j:03d}"
        trajs: List[TrajectoryRecord] = []
        for t_idx, (policy_name, noise, is_random) in enumerate(schedule):
            # Same scenario layout (case_counter=j, thisSeed=base_seed); diversify
            # stochastic behaviors with a per-trajectory RNG stream.
            traj_seed = int(base_seed) + int(j) * 10007 + int(t_idx) * 97
            np.random.seed(traj_seed)
            _reset_scenario(env, scenario_id=j, base_seed=base_seed)
            if not is_random:
                _set_robot_policy(env, cfg, policy_name)
            behavior = (
                f"random"
                if is_random
                else (f"{policy_name}_noise{noise:.2f}" if noise > 0 else policy_name)
            )
            try:
                states, label = _rollout(
                    env,
                    recorder,
                    behavior=behavior,
                    noise_std=noise,
                    random_action=is_random,
                )
            except Exception as exc:  # noqa: BLE001
                console.fail(
                    f"traj {t_idx} failed ({behavior}): {exc}",
                    stage="Stage I collect",
                )
                logging.warning("traj %d failed (%s): %s — skipping", t_idx, behavior, exc)
                continue
            trajs.append(
                TrajectoryRecord(
                    trajectory_id=f"{sid}_t{t_idx:02d}",
                    scenario_id=sid,
                    seed=traj_seed,
                    states=tuple(states),
                    label=label,
                    behavior=behavior,
                    metadata={"policy": policy_name, "noise_std": noise},
                )
            )
        dataset[sid] = trajs
        labels = [t.label for t in trajs]
        logging.debug(
            "  collected %d trajs labels=%s",
            len(trajs),
            {k: labels.count(k) for k in sorted(set(labels))},
        )

    save_stage1_dataset(dataset, out_dir, fmt=fmt)
    from crowd_nav.reward_search import console as _console

    _console.status(
        f"Wrote Stage I dataset → {out_dir} "
        f"({len(dataset)} scenarios, fmt={fmt}) in "
        f"{_console.format_seconds(_time.perf_counter() - t0)}",
        stage="Stage I collect",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Stage I trajectory dataset")
    parser.add_argument("--n-scenarios", type=int, default=100, help="M (paper=100)")
    parser.add_argument("--n-traj", type=int, default=10, help="N_traj (paper=10)")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--out", type=str, default="data/stage1_dataset")
    parser.add_argument("--noise-std-low", type=float, default=0.25)
    parser.add_argument("--noise-std-high", type=float, default=0.55)
    parser.add_argument("--human-num", type=int, default=20)
    parser.add_argument("--format", type=str, default="npz", choices=["npz", "jsonl"])
    parser.add_argument(
        "--regime",
        type=str,
        default="without_random",
        choices=["without_random", "with_random", "both"],
        help="EVOLUTION_RANDOMIZATION_REGIME (default: without_random; AUDIT.md §8.1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-scenario label breakdowns at INFO",
    )
    args = parser.parse_args()
    if args.regime == "both":
        print(
            "regime=both requires two separate collect passes (doubled budget). "
            "Use --regime without_random or --regime with_random.",
            file=sys.stderr,
        )
        return 2
    sys.argv = [sys.argv[0], "--no-cuda", "--num-processes", "1"]

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    from crowd_nav.reward_search import console as _console

    _console.set_verbose(bool(args.verbose))
    collect_dataset(
        n_scenarios=args.n_scenarios,
        n_traj=args.n_traj,
        base_seed=args.seed,
        out_dir=args.out,
        noise_stds=(args.noise_std_low, args.noise_std_high),
        human_num=args.human_num,
        fmt=args.format,
        randomization_regime=args.regime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

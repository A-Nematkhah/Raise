#!/usr/bin/env python
"""Train and evaluate PPO with CrowdNav++'s native reward for calibration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os as _os
import sys as _sys

_RAISE_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _RAISE_ROOT not in _sys.path:
    _sys.path.insert(0, _RAISE_ROOT)
import raise_paths  # noqa: E402,F401

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from domains.crowdnav.adapter import (  # noqa: E402
    _crowdnav_policy_kwargs,
    _eval_metrics,
    _make_train_vec_env,
    metrics_to_crowdnav_dict,
)
from domains.crowdnav.env_wrapper import (  # noqa: E402
    RewardInjectedCrowdNavEnv,
    eval_cases,
)
from domains.crowdnav.settings import CrowdNavSettings  # noqa: E402
from raise_config import build_upstream_config, resolve_config  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.seeds import HOLDOUT_SEED_OFFSET  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override raise_config.env (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/crowdnav_native_calibration_<timestamp>)",
    )
    parser.add_argument("--train-steps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate an existing native_reward_ppo.zip without retraining",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Continue PPO training from an existing native_reward_ppo.zip",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.train_steps < 1:
        raise ValueError("--train-steps must be positive")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be positive")
    if args.n_envs < 1:
        raise ValueError("--n-envs must be positive")
    if args.torch_threads < 0:
        raise ValueError("--torch-threads cannot be negative")
    if args.eval_only and args.resume_from:
        raise ValueError("--eval-only and --resume-from cannot be used together")

    raise_cfg = resolve_config(set_overrides=tuple(args.set or ()))
    upstream = build_upstream_config(raise_cfg.env, config=raise_cfg)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"results/crowdnav_native_calibration_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "native_reward_ppo.zip"
    metrics_path = output_dir / "calibration_metrics.json"
    if metrics_path.exists() or (model_path.exists() and not args.eval_only):
        raise FileExistsError(
            f"Calibration outputs already exist in {output_dir}; choose a new --output-dir"
        )
    if args.eval_only and not model_path.is_file():
        raise FileNotFoundError(
            f"--eval-only requires an existing PPO checkpoint: {model_path}"
        )

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    kinematics = str(upstream.action_space.kinematics)
    settings = CrowdNavSettings(
        kinematics=kinematics,
        action_mode="continuous",
        env_config_obj=upstream,
    )
    n_steps = min(256, max(16, args.train_steps // args.n_envs))
    batch_size = min(64 * args.n_envs, max(8, n_steps * args.n_envs))

    if args.eval_only:
        model = PPO.load(str(model_path), device=args.device)
        train_seconds = None
    else:
        train_env = _make_train_vec_env(
            None, seed=args.seed, n_envs=args.n_envs, settings=settings
        )
        try:
            if args.resume_from:
                resume_path = Path(args.resume_from).resolve()
                if not resume_path.is_file():
                    raise FileNotFoundError(
                        f"PPO checkpoint for continuation not found: {resume_path}"
                    )
                model = PPO.load(
                    str(resume_path),
                    env=train_env,
                    device=args.device,
                )
                starting_timesteps = int(model.num_timesteps)
                print(
                    f"Continuing PPO from {resume_path} "
                    f"({starting_timesteps:,} existing steps)"
                )
            else:
                model = PPO(
                    "MultiInputPolicy",
                    train_env,
                    verbose=1,
                    seed=args.seed,
                    device=args.device,
                    policy_kwargs=_crowdnav_policy_kwargs(),
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=10,
                    gamma=0.99,
                    learning_rate=3e-4,
                )
                starting_timesteps = 0
            train_started = time.perf_counter()
            model.learn(
                total_timesteps=args.train_steps,
                progress_bar=False,
                reset_num_timesteps=not bool(args.resume_from),
            )
        finally:
            train_env.close()
        train_seconds = time.perf_counter() - train_started
        model.save(str(model_path))

    train_steps_added = (
        int(model.num_timesteps - starting_timesteps)
        if train_seconds is not None
        else 0
    )
    eval_env = RewardInjectedCrowdNavEnv(
        None, phase="val", settings=settings
    )
    try:
        model.set_random_seed(args.seed + HOLDOUT_SEED_OFFSET)
        cases = eval_cases(
            args.seed,
            args.eval_episodes,
            int(eval_env.sim.case_size["val"]),
        )
        metrics = _eval_metrics(
            eval_env,
            model,
            cases=cases,
            deterministic=False,
        )
        results = metrics_to_crowdnav_dict(metrics)
        results["holdout_episodes"] = list(
            getattr(metrics, "_crowdnav_episodes", [])
        )
        results["calibration"] = {
            "simulator": "CrowdNav++ CrowdSimVarNum",
            "reward": "native upstream reward",
            "env_config": str(env_config),
            "kinematics": settings.kinematics,
            "action_mode": settings.action_mode,
            "train_steps_requested": int(args.train_steps),
            "train_steps_actual": int(model.num_timesteps),
            "train_steps_added": train_steps_added,
            "starting_steps": int(starting_timesteps)
            if train_seconds is not None
            else int(model.num_timesteps),
            "resumed_from": (
                str(Path(args.resume_from).resolve()) if args.resume_from else None
            ),
            "train_seed": int(args.seed),
            "eval_split": "val",
            "eval_seed": int(args.seed),
            "eval_episodes": int(args.eval_episodes),
            "train_wall_seconds": (
                round(train_seconds, 3) if train_seconds is not None else None
            ),
            "train_steps_per_second": (
                round(float(train_steps_added) / max(train_seconds, 1e-9), 3)
                if train_seconds is not None
                else None
            ),
            "human_num": int(eval_env.human_num),
            "human_num_range": int(eval_env.sim.human_num_range),
            "min_human_num": int(eval_env.min_human_num),
            "max_human_num": int(eval_env.max_human_num),
            "robot_visible_to_humans": bool(eval_env.sim.robot.visible),
            "time_limit_s": float(eval_env.time_limit),
        }
    finally:
        eval_env.close()

    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Native-reward calibration: "
        f"SR={results['SR']:.3f} CR={results['CR']:.3f} "
        f"TR={results['TR']:.3f} fitness={results['fitness']:.3f}"
    )
    if train_seconds is None:
        print(
            f"Training: loaded checkpoint with {model.num_timesteps:,} steps "
            "(evaluation only)"
        )
    else:
        if args.resume_from:
            print(
                f"Training: +{train_steps_added:,} steps in "
                f"{train_seconds / 3600:.2f} h "
                f"({model.num_timesteps:,} total)"
            )
        else:
            print(
                f"Training: {model.num_timesteps:,} steps "
                f"in {train_seconds / 3600:.2f} h"
            )
    if results["SR"] > 0.90:
        print("Calibration result: too easy (success rate above 90%)")
    elif results["SR"] < 0.30:
        print("Calibration result: too difficult (success rate below 30%)")
    elif 0.50 <= results["SR"] <= 0.75:
        print("Calibration result: success rate is within the preferred 50%-75% target")
    else:
        print(
            "Calibration result: success rate is between the 30%-90% thresholds, "
            "but outside the preferred 50%-75% target"
        )
    print(f"Metrics: {metrics_path}")
    print(f"Model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

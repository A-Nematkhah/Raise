#!/usr/bin/env python
"""Phase 0-1: Stage II label reliability across PPO seeds (highway).

Re-labels final-population reward codes of a finished closed-loop run with the
same Stage II recipe (K2 env steps, holdout eval) under one training seed, then
re-evaluates every saved policy on a FIXED holdout episode set so training-seed
noise is separable from eval-episode noise. Also logs the deterministic action
histogram and mean policy entropy on the fixed set (collapse diagnostics).

One seed per process; run seeds in parallel:

  python scripts/label_reliability_highway.py --seed 425
  python scripts/label_reliability_highway.py --seed 426
  python scripts/label_reliability_highway.py --seed 427
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import replace as dc_replace  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from domains.highway.action_config import configure_action_mode_from_config  # noqa: E402
from domains.highway.adapter import (  # noqa: E402
    _eval_metrics,
    _make_train_vec_env,
    metrics_to_highway_dict,
)
from domains.highway.env_wrapper import (  # noqa: E402
    HOLDOUT_SEED_OFFSET,
    RewardInjectedHighwayEnv,
    holdout_env_config,
    training_env_config,
)
from domains.highway.metrics import attach_fitness  # noqa: E402
from domains.highway.reward_checks import make_highway_validator  # noqa: E402
from raise_core.domains import load_domain, make_stage2_trainer_for_domain  # noqa: E402
from raise_core.explore import RewardCandidate  # noqa: E402
from raise_core.refine import Stage2Config  # noqa: E402
from raise_core.sandbox.config import SandboxConfig  # noqa: E402

DEFAULT_RUN = "results/highway_4h_20260930_124808"
KEEP_KEYS = (
    "SR",
    "CR",
    "TR",
    "mean_speed",
    "mean_progress",
    "fitness",
    "speed_p10",
    "speed_p90",
    "lane_change_rate",
    "overtakes_per_km",
)


def _subset(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(metrics[k]) for k in KEEP_KEYS if metrics.get(k) is not None}


def _fixed_eval(model: Any, reward_fn: Any, args: Any, *, deterministic: bool) -> Dict[str, float]:
    env = RewardInjectedHighwayEnv(
        reward_fn, seed=int(args.fixed_eval_seed) + 7, config=holdout_env_config()
    )
    try:
        pm = _eval_metrics(
            env,
            model,
            n_episodes=int(args.eval_episodes),
            seed=int(args.fixed_eval_seed),
            seed_offset=HOLDOUT_SEED_OFFSET,
            deterministic=deterministic,
        )
    finally:
        env.close()
    return _subset(attach_fitness(metrics_to_highway_dict(pm)))


def _action_stats(model: Any, reward_fn: Any, *, seed: int, n_episodes: int) -> Dict[str, Any]:
    env = RewardInjectedHighwayEnv(reward_fn, seed=seed + 7, config=holdout_env_config())
    hist: Counter = Counter()
    entropies: List[float] = []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + HOLDOUT_SEED_OFFSET + ep * 97)
            done = False
            while not done:
                obs_t, _ = model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    ent = model.policy.get_distribution(obs_t).entropy()
                entropies.append(float(ent.mean().item()))
                action, _ = model.predict(obs, deterministic=True)
                hist[str(np.asarray(action).tolist())] += 1
                obs, _r, term, trunc, _info = env.step(action)
                done = bool(term or trunc)
    finally:
        env.close()
    total = max(1, sum(hist.values()))
    return {
        "action_frac": {k: round(v / total, 4) for k, v in hist.most_common()},
        "mean_entropy": float(np.mean(entropies)) if entropies else None,
        "n_steps": int(total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", default="results/_label_reliability_20260930")
    parser.add_argument("--train-steps", type=int, default=12_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--fixed-eval-seed", type=int, default=10_000)
    parser.add_argument("--action-episodes", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--n-steps", type=int, default=256, help="PPO rollout length cap per env")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument(
        "--zero-train",
        action="store_true",
        help="Control: evaluate the seed-initialised policy without any PPO update.",
    )
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Skip training; re-evaluate existing r00_<cid>/model.zip (writes labels_reeval.jsonl).",
    )
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    run_cfg = json.load(open(os.path.join(args.run, "config.json"), encoding="utf-8"))
    pop = json.load(open(os.path.join(args.run, "stage2_population.json"), encoding="utf-8"))[
        "population"
    ]
    if args.ids:
        pop = [c for c in pop if c["candidate_id"] in set(args.ids)]

    pack = load_domain("highway")
    # Windows spawn re-imports this module (torch/SB3) in the sandbox child.
    validator = make_highway_validator(
        config=dc_replace(pack.sandbox_config or SandboxConfig(), timeout_seconds=90.0),
        smoke_states=pack.smoke_states_fn() if pack.smoke_states_fn is not None else None,
    )
    trainer = make_stage2_trainer_for_domain(pack, use_stub=False)
    out_root = os.path.join(args.out, f"seed_{args.seed}")
    os.makedirs(out_root, exist_ok=True)
    cfg = Stage2Config(
        train_env_steps=int(args.train_steps),
        k2_unit="env_steps",
        eval_episodes=int(args.eval_episodes),
        horizon_steps=int(run_cfg.get("stage2_horizon", 100)),
        seed=int(args.seed),
        device=str(run_cfg.get("device", "cpu")),
        output_root=out_root,
        highway_n_envs=int(run_cfg.get("highway_n_envs", 1)),
        highway_warm_start=False,
        highway_eval_mode=str(run_cfg.get("highway_eval_mode", "holdout_only")),
        highway_action_mode=str(run_cfg.get("highway_action_mode", "meta_default")),
        highway_action_continuous_lateral=bool(
            run_cfg.get("highway_action_continuous_lateral", True)
        ),
        highway_meta_fine_low=float(run_cfg.get("highway_meta_fine_low", 15.0)),
        highway_meta_fine_high=float(run_cfg.get("highway_meta_fine_high", 35.0)),
        highway_meta_fine_n=int(run_cfg.get("highway_meta_fine_n", 21)),
        highway_ent_coef=float(args.ent_coef),
        highway_n_steps=int(args.n_steps),
        highway_n_epochs=int(args.n_epochs),
    )
    configure_action_mode_from_config(cfg)
    rows_path = os.path.join(
        out_root, "labels_reeval.jsonl" if args.reuse_models else "labels.jsonl"
    )

    for src in pop:
        cid = str(src["candidate_id"])
        cand = RewardCandidate(
            candidate_id=cid,
            code=src["code"],
            reward_fn=validator.validate_code(src["code"]),
            valid=True,
            origin=str(src.get("origin") or "unknown"),
        )
        t0 = time.perf_counter()
        native: Dict[str, float] = {}
        if args.zero_train:
            train_env = _make_train_vec_env(
                cand.reward_fn, seed=int(args.seed), n_envs=1, env_config=training_env_config()
            )
            model = PPO("MlpPolicy", train_env, seed=int(args.seed), device="cpu")
            train_env.close()
        else:
            if not args.reuse_models:
                trainer.train_and_eval(cand, round_index=0, config=cfg)
                native = _subset(cand.metadata.get("last_metrics") or {})
            ckpt = os.path.join(out_root, f"r00_{cid}", "model.zip")
            model = PPO.load(ckpt, device="cpu")
        fixed = _fixed_eval(model, cand.reward_fn, args, deterministic=True)
        torch.manual_seed(int(args.fixed_eval_seed))
        fixed_stoch = _fixed_eval(model, cand.reward_fn, args, deterministic=False)
        actions = _action_stats(
            model,
            cand.reward_fn,
            seed=int(args.fixed_eval_seed),
            n_episodes=int(args.action_episodes),
        )
        row = {
            "candidate_id": cid,
            "seed": int(args.seed),
            "train_steps": int(args.train_steps),
            "ent_coef": float(args.ent_coef),
            "n_steps": int(args.n_steps),
            "n_epochs": int(args.n_epochs),
            "zero_train": bool(args.zero_train),
            "native": native,
            "fixed_eval": fixed,
            "fixed_eval_stochastic": fixed_stoch,
            "actions": actions,
            "wall_seconds": round(time.perf_counter() - t0, 1),
        }
        with open(rows_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"[seed {args.seed}] {cid}: native SR={native.get('SR')} v={native.get('mean_speed', 0):.2f} | "
            f"fixed SR={fixed.get('SR')} v={fixed.get('mean_speed', 0):.2f} | "
            f"stoch SR={fixed_stoch.get('SR')} v={fixed_stoch.get('mean_speed', 0):.2f} | "
            f"H={actions['mean_entropy']:.3f} top={list(actions['action_frac'].items())[:2]} "
            f"({row['wall_seconds']}s)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

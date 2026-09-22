#!/usr/bin/env python
"""
Train DS-RNN Table 1 baselines (Option A).

Uses this repo's ``srnn`` policy + ``--algo ppo`` for the same env-step budget
as EvoNav Stage III (``STAGE3_STEPS`` by default; ``--paper-steps`` → 1e7).

Writes::

    trained_models/ds_rnn_no_rand/   # Sec 5.1 without human randomization
    trained_models/ds_rnn_rand/      # Sec 5.1 with randomization

Examples::

    # Practical K3-equivalent budget (default STAGE3_STEPS)
    python scripts/train_ds_rnn.py

    # Paper-scale PPO budget
    python scripts/train_ds_rnn.py --paper-steps --num-processes 16

    # Smoke (tiny steps; for CI wiring checks)
    python scripts/train_ds_rnn.py --fast --only no_rand
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def _purge_config_modules() -> None:
    """Allow Config / get_args to re-parse a fresh sys.argv."""
    doomed = [
        name
        for name in list(sys.modules)
        if name in ("arguments", "train")
        or name.startswith("crowd_nav.configs")
    ]
    for name in doomed:
        del sys.modules[name]


def _patch_get_args() -> None:
    """
    Force DS-RNN training flags.

    ``arguments.py`` uses ``type=bool``, which treats any non-empty string as
    True — so attention flags cannot be disabled via CLI alone.
    """
    import arguments as arguments_mod

    _orig = arguments_mod.get_args

    def _wrapped():
        args = _orig()
        args.use_self_attn = False
        args.use_hr_attn = False
        args.env_name = "CrowdSimVarNum-v0"
        return args

    arguments_mod.get_args = _wrapped  # type: ignore[assignment]


def _apply_dsrnn_config(*, randomize: bool) -> None:
    from crowd_nav.configs.config import Config

    Config.robot.policy = "srnn"
    Config.sim.predict_method = "none"
    Config.env.use_wrapper = False
    Config.env.randomize_attributes = bool(randomize)
    Config.humans.random_goal_changing = bool(randomize)
    # Continuous human flow stays on in both paper settings (AUDIT §4).
    Config.humans.end_goal_changing = True
    Config.args.use_self_attn = False
    Config.args.use_hr_attn = False
    Config.args.env_name = "CrowdSimVarNum-v0"


def train_ds_rnn_variant(
    *,
    output_dir: str,
    randomize: bool,
    num_env_steps: int,
    seed: int,
    num_processes: int,
    device: str,
    overwrite: bool,
) -> str:
    """
    Train one DS-RNN checkpoint tree; returns ``output_dir``.

    Mutates process-global ``Config`` / argv — call once per variant in-process,
    or use the CLI which trains sequentially.
    """
    argv: List[str] = [
        sys.argv[0],
        "--output_dir",
        output_dir,
        "--algo",
        "ppo",
        "--env-name",
        "CrowdSimVarNum-v0",
        "--num-env-steps",
        str(int(num_env_steps)),
        "--seed",
        str(int(seed)),
        "--num-processes",
        str(int(num_processes)),
    ]
    if device == "cpu":
        argv.append("--no-cuda")
    if overwrite:
        argv.append("--overwrite")

    saved_argv = list(sys.argv)
    sys.argv = argv
    try:
        _purge_config_modules()
        _patch_get_args()
        _apply_dsrnn_config(randomize=randomize)
        import train as train_mod

        logging.info(
            "Training DS-RNN → %s (randomize=%s, steps=%d, processes=%d)",
            output_dir,
            randomize,
            num_env_steps,
            num_processes,
        )
        train_mod.main()
    finally:
        sys.argv = saved_argv

    return output_dir


def main() -> int:
    from crowd_nav.reward_search.dsrnn_baseline import (
        DEFAULT_DSRNN_NO_RAND_DIR,
        DEFAULT_DSRNN_RAND_DIR,
    )
    from crowd_nav.reward_search.stage3 import STAGE3_PAPER_STEPS, STAGE3_STEPS

    parser = argparse.ArgumentParser(description="Train DS-RNN Table 1 baselines")
    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=None,
        help=f"PPO env steps (default=STAGE3_STEPS={STAGE3_STEPS})",
    )
    parser.add_argument(
        "--paper-steps",
        action="store_true",
        help=f"Use paper Stage III budget ({STAGE3_PAPER_STEPS})",
    )
    parser.add_argument("--fast", action="store_true", help="Tiny step budget for smoke")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--only",
        type=str,
        default="both",
        choices=["both", "no_rand", "rand"],
        help="Which Sec 5.1 variant(s) to train",
    )
    parser.add_argument("--no-rand-dir", type=str, default=DEFAULT_DSRNN_NO_RAND_DIR)
    parser.add_argument("--rand-dir", type=str, default=DEFAULT_DSRNN_RAND_DIR)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if output_dir already exists",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.fast:
        steps = 64
    elif args.paper_steps:
        steps = STAGE3_PAPER_STEPS
    elif args.num_env_steps is not None:
        steps = args.num_env_steps
    else:
        steps = STAGE3_STEPS

    variants = []
    if args.only in ("both", "no_rand"):
        variants.append((args.no_rand_dir, False))
    if args.only in ("both", "rand"):
        variants.append((args.rand_dir, True))

    for out_dir, randomize in variants:
        train_ds_rnn_variant(
            output_dir=out_dir,
            randomize=randomize,
            num_env_steps=steps,
            seed=args.seed,
            num_processes=args.num_processes,
            device=args.device,
            overwrite=not args.no_overwrite,
        )
        logging.info("Finished DS-RNN training → %s", out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

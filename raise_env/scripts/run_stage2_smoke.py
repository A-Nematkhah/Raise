#!/usr/bin/env python
"""
Human-triggered Stage II smoke test against the real A2C trainer.

This is intentionally slow (real env + policy). For CI / pytest use the stub
trainer in ``crowd_nav/reward_search/tests/test_stage2.py`` instead.

Example (from ``evonav_env/`` with the project venv active)::

    python scripts/run_stage2_smoke.py
    python scripts/run_stage2_smoke.py --train-env-steps 200 --eval-episodes 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Repo root = parent of scripts/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage II real-trainer smoke")
    parser.add_argument("--train-env-steps", type=int, default=200, help="K2 (smoke default << 8000)")
    parser.add_argument("--eval-episodes", type=int, default=2, help="E2 (smoke default << 50)")
    parser.add_argument("--horizon-steps", type=int, default=20, help="T_short smoke default")
    parser.add_argument("--rounds", type=int, default=1, help="G2 smoke default")
    parser.add_argument("--population-size", type=int, default=1, help="N smoke default")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument("--output-root", type=str, default="trained_models/stage2_smoke")
    parser.add_argument("--no-refine", action="store_true", help="Skip LLM refinement (train+eval only)")
    args = parser.parse_args()
    # Strip smoke-only flags so later Config/get_args imports (class-body
    # side effect in crowd_nav.configs.config) do not see them.
    sys.argv = [sys.argv[0], "--no-cuda", "--num-processes", "1"]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Import crowd_sim registration + Stage II after chdir/path setup.
    import crowd_sim  # noqa: F401
    from crowd_nav.reward_search.evolver import RewardCandidate
    from crowd_nav.reward_search.llm import ScriptedLLMClient
    from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION
    from crowd_nav.reward_search.sandbox import RewardValidator
    from crowd_nav.reward_search.stage2 import (
        RealPolicyTrainer,
        Stage2Config,
        Stage2Runner,
    )

    validator = RewardValidator()
    code = D5_SEED_FUNCTION.strip() + "\n"
    reward_fn, err = validator.try_validate(code)
    if reward_fn is None:
        logging.error("Seed reward failed sandbox: %s", err)
        return 1

    pop = [
        RewardCandidate(
            candidate_id=f"smoke_{i}",
            code=code,
            reward_fn=reward_fn,
            valid=True,
            origin="seed",
        )
        for i in range(args.population_size)
    ]

    cfg = Stage2Config(
        population_size=args.population_size,
        rounds=args.rounds,
        train_env_steps=args.train_env_steps,
        eval_episodes=args.eval_episodes,
        horizon_steps=args.horizon_steps,
        algo="a2c",
        seed=args.seed,
        output_root=args.output_root,
        device="cpu",
        env_name="CrowdSimVarNum-v0",
    )

    if args.no_refine:
        trainer = RealPolicyTrainer()
        metrics = trainer.train_and_eval(pop[0], round_index=0, config=cfg)
        logging.info("Smoke metrics: %s", metrics.feedback_text())
        return 0

    # Scripted refine: return a slightly tweaked valid reward (no API key needed).
    refined = (
        "```python\n"
        "def compute_reward(state):\n"
        "    if state.reaching_goal:\n"
        "        return float(10.0)\n"
        "    if state.collision:\n"
        "        return float(-20.0)\n"
        "    dist = ((state.robot.px - state.robot.gx) ** 2 + "
        "(state.robot.py - state.robot.gy) ** 2) ** 0.5\n"
        "    return float(-0.1 * dist)\n"
        "```\n"
    )
    completions = [refined] * (args.population_size * args.rounds)
    runner = Stage2Runner(
        ScriptedLLMClient(completions),
        RealPolicyTrainer(),
        validator=validator,
        config=cfg,
    )
    out = runner.run(pop)
    for rec in runner.history:
        logging.info(
            "round=%d id=%s refined=%s kept_prev=%s metrics=%s",
            rec.round_index,
            rec.candidate_id,
            rec.refined,
            rec.kept_previous,
            rec.metrics.feedback_text(),
        )
    logging.info("Final population: %s", [c.candidate_id for c in out])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

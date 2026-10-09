#!/usr/bin/env python
"""
Single entry point for RAISE on highway-fast-v0.

  closed loop (G generations): Score1 → Stage II PPO train+eval → fitness
  rank → LLM crossover/mutation with evaluation feedback
  → Stage III long PPO on the final population → best by fitness

Examples (from ``raise_env/`` with system Python)::

    # Seconds-scale dry run (stub trainers + seed LLM)
    python scripts/run_raise.py --fast --output-dir results/raise_fast

    # Real PPO run with a real LLM
    python scripts/run_raise.py --llm groq --output-dir results/highway_run

    # Real PPO run without LLM cost (seed-variant rewards)
    python scripts/run_raise.py --llm seed --allow-seed-llm

    # Resume: rerun with the same --output-dir (resume is on by default)
    python scripts/run_raise.py --llm groq --output-dir results/highway_run
"""

from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402

from raise_core import console  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.pipeline import RaisePipeline, RaiseRunConfig  # noqa: E402

os.chdir(_ROOT)

_DEFAULTS = RaiseRunConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAISE end-to-end (highway)")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the effective argparse namespace and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Run directory (default: results/highway_<timestamp>)",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument(
        "--llm",
        type=str,
        default=_DEFAULTS.llm_provider,
        choices=["seed", "groq", "vllm", "ollama", "scripted"],
        help="LLM provider (seed = local seed variants, no API key; "
        "non-fast runs require a real provider or --allow-seed-llm)",
    )
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument(
        "--allow-seed-llm",
        action="store_true",
        help="Permit --llm seed on a non-fast run (debug / wiring without API cost)",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument(
        "--score1",
        type=str,
        default=_DEFAULTS.score1_mode,
        choices=["dataset", "smoke"],
        help="Stage I scorer: dataset=Score1 on pre-collected trajs; smoke=fast-test only",
    )
    parser.add_argument(
        "--stage1-dataset",
        type=str,
        default=_DEFAULTS.stage1_dataset_path,
        help="Path from scripts/collect_highway_stage1_dataset.py",
    )
    parser.add_argument("--fast", action="store_true", help="Stub trainers + smoke Score1")
    parser.add_argument(
        "--device", type=str, default=_DEFAULTS.device, choices=["cpu", "cuda"]
    )

    parser.add_argument("--population", type=int, default=_DEFAULTS.population)
    parser.add_argument("--generations", type=int, default=_DEFAULTS.generations)
    parser.add_argument(
        "--stage2-train-steps",
        type=int,
        default=_DEFAULTS.stage2_train_steps,
        help="K2: PPO env steps per candidate per generation",
    )
    parser.add_argument(
        "--stage2-eval-episodes", type=int, default=_DEFAULTS.stage2_eval_episodes
    )
    parser.add_argument(
        "--stage3-train-steps",
        type=int,
        default=_DEFAULTS.stage3_train_steps,
        help="K3: PPO env steps per final candidate",
    )
    parser.add_argument(
        "--stage3-eval-episodes", type=int, default=_DEFAULTS.stage3_eval_episodes
    )
    parser.add_argument(
        "--elitism",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.elitism,
        help="Carry each generation's best genome into the next generation",
    )
    parser.add_argument(
        "--evolve-rank",
        default=_DEFAULTS.evolve_rank,
        choices=("scalar", "score1", "hybrid"),
        help="Parent order for the next generation: scalar=fitness (default), "
        "score1, or hybrid (normalized blend)",
    )
    parser.add_argument(
        "--evolve-rank-score1-weight",
        type=float,
        default=_DEFAULTS.evolve_rank_score1_weight,
        help="Weight on Score1 inside hybrid evolve-rank",
    )
    parser.add_argument(
        "--highway-n-envs",
        type=int,
        default=_DEFAULTS.highway_n_envs,
        help="DummyVecEnv count for PPO (same budget)",
    )
    parser.add_argument(
        "--highway-label-workers",
        type=int,
        default=_DEFAULTS.highway_label_workers,
        help="Parallel Stage II training threads",
    )
    parser.add_argument(
        "--highway-eval-mode",
        choices=("both", "holdout_only"),
        default=_DEFAULTS.highway_eval_mode,
        help="Evaluate train+holdout traffic, or holdout only",
    )
    parser.add_argument(
        "--highway-warm-start",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.highway_warm_start,
        help=(
            "Warm-start PPO from the parent's checkpoint when available "
            "(off by default: warm-started children get a larger effective budget)"
        ),
    )
    parser.add_argument(
        "--highway-action-mode",
        choices=("meta_default", "meta_fine", "continuous"),
        default=_DEFAULTS.highway_action_mode,
        help=(
            "Action space: meta_default=DiscreteMetaAction (default), "
            "meta_fine=linspace target speeds, continuous=ContinuousAction Box"
        ),
    )
    parser.add_argument(
        "--highway-action-continuous-lateral",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.highway_action_continuous_lateral,
        help="ContinuousAction: include steering (Box(2,)); False → accel-only Box(1,)",
    )
    parser.add_argument(
        "--highway-meta-fine-n",
        type=int,
        default=_DEFAULTS.highway_meta_fine_n,
        help="meta_fine: number of target_speeds in linspace(low, high)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help=(
            "Resume from checkpoints when present (default: on): "
            "--output-dir/closed_loop/checkpoint.json and stage3/checkpoint.json"
        ),
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and start fresh",
    )
    return parser


def main() -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args()

    if args.print_config:
        for k, v in sorted(vars(args).items()):
            print(f"{k}={v!r}")
        return 0

    if (
        not args.fast
        and str(args.llm).strip().lower() == "seed"
        and not args.allow_seed_llm
    ):
        print(
            "Refusing to run a non-fast pipeline with the seed (no real LLM) "
            "provider - pass --llm groq|ollama|vllm explicitly, or pass "
            "--allow-seed-llm if this is intentional (e.g. debugging Stage II/III "
            "wiring without LLM cost).",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output_dir or (
        "results/highway_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    console.configure_run_logging(verbose=bool(args.verbose))

    cfg = RaiseRunConfig(
        output_dir=output_dir,
        seed=args.seed,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        score1_mode="smoke" if args.fast else args.score1,
        stage1_dataset_path=args.stage1_dataset,
        population=args.population,
        generations=args.generations,
        stage2_train_steps=args.stage2_train_steps,
        stage2_eval_episodes=args.stage2_eval_episodes,
        stage3_train_steps=args.stage3_train_steps,
        stage3_eval_episodes=args.stage3_eval_episodes,
        use_stub=bool(args.fast),
        elitism=bool(args.elitism),
        evolve_rank=args.evolve_rank,
        evolve_rank_score1_weight=float(args.evolve_rank_score1_weight),
        resume=bool(args.resume),
        highway_n_envs=max(1, int(args.highway_n_envs)),
        highway_label_workers=max(1, int(args.highway_label_workers)),
        highway_warm_start=bool(args.highway_warm_start),
        highway_eval_mode=str(args.highway_eval_mode),
        highway_action_mode=str(args.highway_action_mode),
        highway_action_continuous_lateral=bool(args.highway_action_continuous_lateral),
        highway_meta_fine_n=int(args.highway_meta_fine_n),
        device=args.device,
    )
    if args.fast:
        cfg.apply_fast_profile()

    logging.info(
        "RAISE -> %s (fast=%s, N=%d, G=%d, K2=%d, K3=%d)",
        cfg.output_dir,
        cfg.fast,
        cfg.population,
        cfg.generations,
        cfg.stage2_train_steps,
        cfg.stage3_train_steps,
    )
    artifacts = RaisePipeline(cfg).run()
    logging.info("Done. Final candidate: %s", artifacts.best_stage3.candidate_id)
    logging.info("Artifacts: %s", artifacts.output_dir)
    report = os.path.join(cfg.output_dir, "closed_loop", "REPORT.txt")
    if os.path.isfile(report):
        with open(report, encoding="utf-8") as fh:
            print("\n--- closed_loop/REPORT.txt ---\n" + fh.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Single entry point for RAISE on CrowdNav (vita-epfl/CrowdNav crowd_sim).

  closed loop (G generations): Score1 → Stage II PPO train+eval → fitness
  rank → LLM crossover/mutation with evaluation feedback
  → Stage III long PPO on the final population → best by fitness

Examples (from ``raise_env/`` with system Python)::

    # Seconds-scale dry run (stub trainers + seed LLM)
    python scripts/run_raise.py --fast --output-dir results/raise_fast

    # Real PPO run with a real LLM
    python scripts/run_raise.py --llm groq --output-dir results/crowdnav_run

    # Real PPO run without LLM cost (seed-variant rewards)
    python scripts/run_raise.py --llm seed --allow-seed-llm

    # Resume: rerun with the same --output-dir (resume is on by default)
    python scripts/run_raise.py --llm groq --output-dir results/crowdnav_run
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
from raise_core.domains import available_domains  # noqa: E402
from raise_core.explore import MUTATION_PARENTS  # noqa: E402
from raise_core.feedback import FeedbackCardConfig  # noqa: E402
from raise_core.pipeline import RaisePipeline, RaiseRunConfig  # noqa: E402

os.chdir(_ROOT)

_DEFAULTS = RaiseRunConfig()
_FEEDBACK = FeedbackCardConfig()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAISE end-to-end (CrowdNav)")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the effective argparse namespace and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Run directory (default: results/crowdnav_<timestamp>)",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    parser.add_argument(
        "--domain", choices=available_domains(), default=_DEFAULTS.domain
    )
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
        "--llm-max-tokens",
        type=int,
        default=None,
        help="Max completion tokens per LLM call (default: provider default)",
    )
    parser.add_argument(
        "--llm-reasoning-effort",
        type=str,
        default=None,
        choices=["none", "low", "medium", "high"],
        help="Groq gpt-oss reasoning_effort (omit to use the provider default)",
    )
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
        help="Path from scripts/collect_crowdnav_stage1_dataset.py",
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
        "--stage2-eval-split",
        choices=("val", "train"),
        default=_DEFAULTS.stage2_eval_split,
        help=(
            "Stage II holdout pool (default val). 'train' uses train-split cases "
            "for selection; Stage III/final eval stay on val/test."
        ),
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
        "--stage3-warm-start",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.stage3_warm_start,
        help=(
            "Warm-start Stage III PPO from the Stage II policy (off by default: "
            "warm-started candidates would train longer than K3)"
        ),
    )
    parser.add_argument(
        "--stage3-skip-collapsed",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.stage3_skip_collapsed,
        help=(
            "Skip Stage III training for candidates whose Stage II holdout CR "
            "is at/above --collapse-cr (default: on). Timeout-heavy results "
            "are kept. If every candidate is collapsed, all are trained."
        ),
    )
    parser.add_argument(
        "--collapse-cr",
        type=float,
        default=_DEFAULTS.collapse_cr,
        help="Stage II collision-rate threshold for --stage3-skip-collapsed "
        "and --mutation-exclude-collapsed",
    )
    parser.add_argument(
        "--stage3-seeds",
        type=int,
        default=_DEFAULTS.stage3_seeds,
        help=(
            "Independent PPO train seeds per Stage III finalist (default 1). "
            "Selection uses mean fitness across seeds; pay for K>1 with "
            "--stage3-top-k"
        ),
    )
    parser.add_argument(
        "--stage3-top-k",
        type=int,
        default=_DEFAULTS.stage3_top_k,
        help=(
            "After the collapse filter, keep only the top-k Stage II genomes "
            "for Stage III (default: all survivors)"
        ),
    )
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=_DEFAULTS.final_eval_episodes,
        help=(
            "Blind re-evaluation of the Stage III winner on unseen scenarios "
            "(0 = skip); report this fitness, not the selection score"
        ),
    )
    parser.add_argument(
        "--eval-tied-finalists",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.eval_tied_finalists,
        help=(
            "Also blind-eval every Stage III statistical tie on the test split "
            "(report only in tied_final_evals.json; never used for selection)"
        ),
    )
    parser.add_argument(
        "--elitism",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.elitism,
        help="Carry each generation's best genome into the next generation",
    )
    for op, what in (
        ("crossover", "crossover children of the top two genomes"),
        ("mutation", "mutation children of the lower half"),
        ("random", "random restarts"),
    ):
        parser.add_argument(
            f"--n-{op}",
            type=int,
            default=getattr(_DEFAULTS, f"n_{op}"),
            help=f"Next-generation {what} (default: split of N; N=6 with elitism "
            "is 1 elite + 2 crossover + 2 mutation + 1 random). Elite + "
            "crossover + mutation + random must equal --population.",
        )
    parser.add_argument(
        "--mutation-parents",
        choices=MUTATION_PARENTS,
        default=_DEFAULTS.mutation_parents,
        help="Mutation parents: lower = bottom half (default), rank_weighted = "
        "drawn by linear rank, tournament = better of two random genomes",
    )
    parser.add_argument(
        "--mutation-exclude-collapsed",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.mutation_exclude_collapsed,
        help=(
            "Drop Stage II CR>=--collapse-cr genomes from the mutation parent "
            "pool (default: off). If the filtered pool is empty, fall back to "
            "the unfiltered pool"
        ),
    )
    parser.add_argument(
        "--stage2-successive-halving",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.stage2_successive_halving,
        help=(
            "Stage II successive-halving: train all at K2/3, continue the top "
            "half from their own checkpoints for the remaining 2K2/3 (default: "
            "off). Total budget per survivor is still K2"
        ),
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=_DEFAULTS.n_envs,
        help="DummyVecEnv count for PPO (same budget)",
    )
    parser.add_argument(
        "--label-workers",
        type=int,
        default=_DEFAULTS.label_workers,
        help="Parallel Stage II training threads",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=_DEFAULTS.torch_threads,
        help="CPU intra-op threads for torch/OpenMP/MKL (0 = library default)",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("both", "holdout_only"),
        default=_DEFAULTS.eval_mode,
        help="Evaluate upstream train + holdout (val) scenarios, or holdout only",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULTS.warm_start,
        help=(
            "Stage II: warm-start PPO from the parent's checkpoint when available "
            "(off by default: warm-started children get a larger effective budget)"
        ),
    )
    parser.add_argument(
        "--action-mode",
        choices=("discrete", "continuous"),
        default=_DEFAULTS.action_mode,
        help=(
            "Action space: discrete = upstream CADRL velocity set (stop + "
            "speeds x rotations, default), continuous = Box([-1, 1]^2)"
        ),
    )
    parser.add_argument(
        "--robot-kinematics",
        choices=("holonomic", "unicycle"),
        default=_DEFAULTS.robot_kinematics,
        help="Robot kinematics (upstream env.config robot.kinematics)",
    )
    parser.add_argument(
        "--action-speed-samples",
        type=int,
        default=_DEFAULTS.action_speed_samples,
        help="Discrete: exponentially spaced speeds (upstream CADRL: 5)",
    )
    parser.add_argument(
        "--action-rotation-samples",
        type=int,
        default=_DEFAULTS.action_rotation_samples,
        help="Discrete: headings / rotations (upstream CADRL: 16)",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=_DEFAULTS.env_config_path,
        help="CrowdNav env.config (default: vendored upstream config)",
    )
    parser.add_argument(
        "--feedback-card",
        action=argparse.BooleanOptionalAction,
        default=_FEEDBACK.enabled,
        help="Diagnostic feedback card in mutation/crossover/random prompts "
        "(--no-feedback-card restores the plain evidence/reflection text)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=_FEEDBACK.max_input_tokens,
        help="Hard limit on the full LLM request (system + user prompt); never exceeded",
    )
    parser.add_argument(
        "--target-input-tokens",
        type=int,
        default=_FEEDBACK.target_input_tokens,
        help="Card truncation target for the full LLM request",
    )
    parser.add_argument(
        "--feedback-history", type=int, default=_FEEDBACK.history_entries,
        help="Lineage history entries in the card",
    )
    parser.add_argument(
        "--feedback-trajectories", type=int, default=_FEEDBACK.max_trajectories,
        help="Representative holdout episodes in the card",
    )
    parser.add_argument(
        "--feedback-random-calls",
        action=argparse.BooleanOptionalAction,
        default=_FEEDBACK.random_calls,
        help="Show random restarts a reference card of the best evaluated genome",
    )
    parser.add_argument(
        "--feedback-training-dynamics",
        action=argparse.BooleanOptionalAction,
        default=_FEEDBACK.training_dynamics,
        help="Record training-rollout outcome windows during Stage II PPO",
    )
    parser.add_argument(
        "--feedback-components",
        action=argparse.BooleanOptionalAction,
        default=_FEEDBACK.component_instrumentation,
        help="Record per-episode reward-component sums during Stage II holdout eval",
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
    parser.add_argument(
        "--baselines",
        action="store_true",
        default=False,
        help=(
            "After the final blind eval, run scripts/run_baselines.py on this "
            "output dir (native-reward PPO, best Stage-II genome, ORCA/Linear). "
            "Off by default."
        ),
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
        "results/crowdnav_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    cfg = RaiseRunConfig(
        output_dir=output_dir,
        seed=args.seed,
        domain=args.domain,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
        llm_reasoning_effort=args.llm_reasoning_effort,
        score1_mode="smoke" if args.fast else args.score1,
        stage1_dataset_path=args.stage1_dataset,
        population=args.population,
        generations=args.generations,
        stage2_train_steps=args.stage2_train_steps,
        stage2_eval_episodes=args.stage2_eval_episodes,
        stage2_eval_split=str(args.stage2_eval_split),
        stage3_train_steps=args.stage3_train_steps,
        stage3_eval_episodes=args.stage3_eval_episodes,
        stage3_warm_start=bool(args.stage3_warm_start),
        stage3_skip_collapsed=bool(args.stage3_skip_collapsed),
        collapse_cr=float(args.collapse_cr),
        stage3_seeds=max(1, int(args.stage3_seeds)),
        stage3_top_k=(
            None if args.stage3_top_k is None else max(1, int(args.stage3_top_k))
        ),
        final_eval_episodes=max(0, int(args.final_eval_episodes)),
        eval_tied_finalists=bool(args.eval_tied_finalists),
        use_stub=bool(args.fast),
        elitism=bool(args.elitism),
        n_crossover=args.n_crossover,
        n_mutation=args.n_mutation,
        n_random=args.n_random,
        mutation_parents=args.mutation_parents,
        mutation_exclude_collapsed=bool(args.mutation_exclude_collapsed),
        stage2_successive_halving=bool(args.stage2_successive_halving),
        resume=bool(args.resume),
        n_envs=max(1, int(args.n_envs)),
        label_workers=max(1, int(args.label_workers)),
        warm_start=bool(args.warm_start),
        eval_mode=str(args.eval_mode),
        action_mode=str(args.action_mode),
        robot_kinematics=str(args.robot_kinematics),
        action_speed_samples=int(args.action_speed_samples),
        action_rotation_samples=int(args.action_rotation_samples),
        env_config_path=args.env_config,
        feedback=FeedbackCardConfig(
            enabled=bool(args.feedback_card),
            max_input_tokens=int(args.max_input_tokens),
            target_input_tokens=int(args.target_input_tokens),
            history_entries=int(args.feedback_history),
            max_trajectories=int(args.feedback_trajectories),
            random_calls=bool(args.feedback_random_calls),
            training_dynamics=bool(args.feedback_training_dynamics),
            component_instrumentation=bool(args.feedback_components),
        ),
        device=args.device,
        torch_threads=max(0, int(args.torch_threads)),
    )
    if args.fast:
        cfg.apply_fast_profile()
    try:
        operators = cfg.operators()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    console.configure_run_logging(verbose=bool(args.verbose), output_dir=cfg.output_dir)

    logging.info(
        "RAISE -> %s (fast=%s, N=%d, G=%d, K2=%d, K3=%d, operators=%s)",
        cfg.output_dir,
        cfg.fast,
        cfg.population,
        cfg.generations,
        cfg.stage2_train_steps,
        cfg.stage3_train_steps,
        operators.to_dict(),
    )
    artifacts = RaisePipeline(cfg).run()
    logging.info("Done. Final candidate: %s", artifacts.best_stage3.candidate_id)
    logging.info("Artifacts: %s", artifacts.output_dir)
    report = os.path.join(cfg.output_dir, "closed_loop", "REPORT.txt")
    if os.path.isfile(report):
        with open(report, encoding="utf-8") as fh:
            print("\n--- closed_loop/REPORT.txt ---\n" + fh.read())
    if bool(getattr(args, "baselines", False)):
        import importlib.util

        bas_path = os.path.join(_ROOT, "scripts", "run_baselines.py")
        spec = importlib.util.spec_from_file_location("run_baselines", bas_path)
        if spec is None or spec.loader is None:
            logging.error("Could not load %s", bas_path)
            return 1
        bas = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bas)
        logging.info("Running P1-1 baselines on %s", cfg.output_dir)
        bas.run_baselines(cfg.output_dir)
        logging.info("Baselines written to %s", os.path.join(cfg.output_dir, "baselines.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

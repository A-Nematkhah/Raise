#!/usr/bin/env python
"""
Single entry point for EvoNav Algorithm 1 (faithful replication baseline).

  seed generation → Stage I → Stage II → Stage III

No AMFRS mechanisms (novelty archive, Pareto ranking, adaptive controller).

Examples (from ``evonav_env/`` with system Python)::

    # Seconds-scale dry run (stub trainers + seed LLM)
    python scripts/run_evonav.py --fast --output-dir results/evonav_fast

    # Practical local run (real trainers, reduced Stage III K3)
    python scripts/run_evonav.py --llm seed --output-dir results/evonav_local

    # Paper-faithful Stage III budget on a GPU cluster
    python scripts/run_evonav.py --llm vllm --stage3-train-steps 10000000 \\
        --device cuda --output-dir results/evonav_paper
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASELINES_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "baselines_openai"))
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)
os.chdir(_ROOT)


def main() -> int:
    from crowd_nav.reward_search.pipeline import EvoNavPipeline, EvoNavRunConfig
    from crowd_nav.reward_search.stage3 import STAGE3_PAPER_STEPS, STAGE3_STEPS

    parser = argparse.ArgumentParser(description="EvoNav Algorithm 1 end-to-end")
    parser.add_argument("--output-dir", type=str, default="results/evonav_run")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument(
        "--domain",
        type=str,
        default="crowdnav",
        help="Domain pack under crowd_nav.domains (default: crowdnav baseline)",
    )
    parser.add_argument("--llm-model", type=str, default=None)
    parser.add_argument(
        "--llm",
        type=str,
        default="seed",
        choices=["seed", "groq", "vllm", "ollama", "scripted"],
        help="LLM provider (seed = local D.5 variants, no API key; "
        "non-fast runs require a real provider or --allow-seed-llm)",
    )
    parser.add_argument(
        "--allow-seed-llm",
        action="store_true",
        help="Permit --llm seed on a non-fast run (debug / wiring without API cost)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "DEBUG for crowd_nav/crowd_sim only (HTTP/Groq request dumps stay quiet)"
        ),
    )
    parser.add_argument(
        "--score1",
        type=str,
        default="dataset",
        choices=["dataset", "smoke"],
        help="Stage I scorer: dataset=Score1 on pre-collected trajs; smoke=fast-test only",
    )
    parser.add_argument(
        "--stage1-dataset",
        type=str,
        default="data/stage1_dataset",
        help="Path from scripts/collect_stage1_dataset.py",
    )
    parser.add_argument("--fast", action="store_true", help="Stub trainers + smoke Score1")
    parser.add_argument(
        "--easy",
        action="store_true",
        help=(
            "Easier env for pipeline result-getting: predict_method=none, "
            "5 humans, longer Stage II horizon. Not for paper claims."
        ),
    )
    parser.add_argument(
        "--human-num",
        type=int,
        default=None,
        help="Crowd size for Stage II/III (default 20; --easy sets 5)",
    )
    parser.add_argument(
        "--predict-method",
        type=str,
        default=None,
        choices=["inferred", "none", "const_vel", "truth"],
        help="Override sim.predict_method (default inferred; --easy/--fast use none)",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["paper"],
        help=(
            "Named budget preset. 'paper' redirects to scripts/run_evonav_paper_scale.py "
            "(Tables 3–6, multi-seed); not used by pytest."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Parallel envs for Stage II/III (default: auto). Use 2-4 on 4GB GPUs.",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="without_random",
        choices=["without_random", "with_random", "both"],
        help="EVOLUTION_RANDOMIZATION_REGIME (AUDIT.md §8.1; default without_random)",
    )

    parser.add_argument("--stage1-population", type=int, default=8)
    parser.add_argument("--stage1-generations", type=int, default=10)

    parser.add_argument("--stage2-rounds", type=int, default=16)
    parser.add_argument(
        "--stage2-train-steps",
        type=int,
        default=8_000,
        help=(
            "K2 budget per Stage II candidate (paper Table 5: 8000). "
            "Meaning depends on --k2-unit: gradient_steps = A2C updates "
            "(paper §4.3.2 default); env_steps = env interactions."
        ),
    )
    parser.add_argument(
        "--k2-unit",
        type=str,
        default="gradient_steps",
        choices=["env_steps", "gradient_steps"],
        help=(
            "How to interpret --stage2-train-steps "
            "(default gradient_steps = paper §4.3.2)"
        ),
    )
    parser.add_argument("--stage2-eval-episodes", type=int, default=50)
    parser.add_argument("--stage2-stub", action="store_true")

    parser.add_argument(
        "--stage3-train-steps",
        type=int,
        default=STAGE3_STEPS,
        help=f"K3 env steps (default={STAGE3_STEPS}; paper={STAGE3_PAPER_STEPS})",
    )
    parser.add_argument("--stage3-rounds", type=int, default=3)
    parser.add_argument("--stage3-eval-episodes", type=int, default=500)
    parser.add_argument("--stage3-stub", action="store_true")
    parser.add_argument("--no-h-sweep", action="store_true")
    parser.add_argument(
        "--elitism",
        action="store_true",
        help=(
            "Enable non-paper elite inject / protect-refine "
            "(Algorithm 1 has no elitism; default off)"
        ),
    )
    parser.add_argument(
        "--no-elitism",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy no-op; paper default is already off
    )
    parser.add_argument(
        "--final-rank",
        type=str,
        default="llm",
        choices=["scalar", "llm"],
        help=(
            "R2/R3 ranking: llm=Alg.1 multi-objective LLM rank (default; "
            "seed falls back to lex); scalar=SR-CR-0.5TR engineering baseline"
        ),
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default=None,
        help=(
            "Path to fitted surrogate dir (must contain model.joblib). "
            "Enables Stage I predict + optional Stage III gate."
        ),
    )
    parser.add_argument(
        "--no-surrogate-gate",
        action="store_true",
        help="With --surrogate, still predict but do not drop candidates before Stage III",
    )
    parser.add_argument(
        "--surrogate-drop-fraction",
        type=float,
        default=0.25,
        help="Fraction of Stage II pop to drop when confident-weak (default 0.25)",
    )
    parser.add_argument(
        "--active-learning",
        action="store_true",
        help="After Stage I, run one AL step (requires --surrogate with a fitted model)",
    )
    parser.add_argument(
        "--al-max-queries",
        type=int,
        default=3,
        help="Max AL acquire queries after Stage I (default 3)",
    )
    parser.add_argument(
        "--surrogate-dataset",
        type=str,
        default="data/surrogate_dataset",
        help="Surrogate label jsonl root for AL append/refit",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help=(
            "Innovation path: Gen epochs with Score1 → Surrogate gate → "
            "in-loop AL → Stage II short labels → refit (not paper Alg.1 linear)"
        ),
    )
    parser.add_argument(
        "--closed-loop-no-al",
        action="store_true",
        help="With --closed-loop, disable in-loop active learning picks",
    )
    parser.add_argument(
        "--closed-loop-al-max",
        type=int,
        default=4,
        help="Max AL stage2_label picks per closed-loop epoch (default 4)",
    )
    parser.add_argument(
        "--closed-loop-k2",
        type=int,
        default=4000,
        help="Stage II short budget inside closed-loop epochs (default 4000)",
    )
    parser.add_argument(
        "--closed-loop-min-labels-gate",
        type=int,
        default=24,
        help="Soft Surrogate gate until this many labels exist (default 24)",
    )
    parser.add_argument(
        "--closed-loop-refit-every",
        type=int,
        default=8,
        help="Refit Surrogate after this many new Stage II labels (default 8)",
    )
    parser.add_argument(
        "--closed-loop-min-stage2",
        type=int,
        default=4,
        help="Min candidates to Stage II-label per closed-loop epoch (default 4)",
    )

    args = parser.parse_args()

    if args.scale == "paper":
        # Multi-seed paper budgets live in a dedicated human-triggered script.
        print(
            "Paper-scale (Tables 3–6, multi-seed) is only available via:\n"
            "  python scripts/run_evonav_paper_scale.py\n"
            "Pass --seeds / --device there. This keeps pytest/--fast unchanged "
            "and avoids accidental CI runs of K3=1e7.",
            file=sys.stderr,
        )
        return 2

    # Fail closed before Config / GST / dataset / simulator work.
    if (
        not args.fast
        and str(args.llm).strip().lower() == "seed"
        and not args.allow_seed_llm
    ):
        print(
            "Refusing to run a non-fast pipeline with the seed (no real LLM) "
            "provider — pass --llm groq|ollama|vllm explicitly, or pass "
            "--allow-seed-llm if this is intentional (e.g. debugging Stage II/III "
            "wiring without LLM cost).",
            file=sys.stderr,
        )
        return 2

    # Protect Config.get_args() class-body from our CLI flags.
    sys.argv = [sys.argv[0], "--no-cuda" if args.device == "cpu" else "--seed", str(args.seed)]

    from crowd_nav.reward_search import console as _console

    _console.configure_run_logging(verbose=bool(args.verbose))

    import crowd_sim  # noqa: F401

    cfg = EvoNavRunConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        domain=args.domain,
        llm_provider=args.llm,
        llm_model=args.llm_model,
        score1_mode="smoke" if args.fast else args.score1,
        stage1_dataset_path=args.stage1_dataset,
        stage1_population=args.stage1_population,
        stage1_generations=args.stage1_generations,
        stage2_rounds=args.stage2_rounds,
        stage2_train_steps=args.stage2_train_steps,
        stage2_k2_unit=args.k2_unit,
        stage2_eval_episodes=args.stage2_eval_episodes,
        stage2_use_stub=args.stage2_stub or args.fast,
        stage3_rounds=args.stage3_rounds,
        stage3_train_steps=args.stage3_train_steps,
        stage3_eval_episodes=args.stage3_eval_episodes,
        stage3_use_stub=args.stage3_stub or args.fast,
        stage3_run_h_sweep=not args.no_h_sweep,
        elitism=bool(args.elitism),
        final_rank=args.final_rank,
        surrogate_model_dir=args.surrogate,
        surrogate_dataset=args.surrogate_dataset,
        surrogate_gate_stage3=not bool(args.no_surrogate_gate),
        surrogate_drop_fraction=float(args.surrogate_drop_fraction),
        active_learning=bool(args.active_learning),
        active_learning_max_queries=int(args.al_max_queries),
        closed_loop=bool(args.closed_loop),
        closed_loop_no_al=bool(args.closed_loop_no_al),
        closed_loop_al_max_per_epoch=int(args.closed_loop_al_max),
        closed_loop_min_labels_for_gate=int(args.closed_loop_min_labels_gate),
        closed_loop_k2=int(args.closed_loop_k2),
        closed_loop_refit_every_new_labels=int(args.closed_loop_refit_every),
        closed_loop_min_stage2_per_gen=int(args.closed_loop_min_stage2),
        device=args.device,
        num_processes=args.num_processes,
        randomization_regime=args.regime,
        predict_method="none" if args.fast else "inferred",
        human_num=20,
    )
    if args.regime == "both":
        print(
            "regime=both requires two separate full Algorithm-1 passes "
            "(doubled budget). Use --regime without_random or with_random.",
            file=sys.stderr,
        )
        return 2
    if args.fast:
        cfg.apply_fast_profile()
        cfg.output_dir = args.output_dir
        cfg.seed = args.seed
    if args.easy and not args.fast:
        cfg.apply_easy_profile()
    if args.predict_method is not None and not args.fast:
        cfg.predict_method = args.predict_method
    if args.human_num is not None:
        cfg.human_num = max(1, int(args.human_num))

    from crowd_nav.domains import available_domains, load_domain

    try:
        load_domain(cfg.domain)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"Available domains: {', '.join(available_domains())}",
            file=sys.stderr,
        )
        return 2

    logging.info(
        "EvoNav Algorithm 1 → %s (domain=%s, fast=%s, easy=%s, humans=%d, predict=%s, K3=%d)",
        cfg.output_dir,
        cfg.domain,
        cfg.fast,
        bool(args.easy),
        cfg.human_num,
        cfg.predict_method,
        cfg.stage3_train_steps,
    )
    artifacts = EvoNavPipeline(cfg).run()
    logging.info("Done. Final candidate: %s", artifacts.best_stage3.candidate_id)
    logging.info("Artifacts: %s", artifacts.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

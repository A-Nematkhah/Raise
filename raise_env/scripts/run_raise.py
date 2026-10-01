#!/usr/bin/env python
"""
Single entry point for RAISE (faithful replication baseline).

  seed generation → Stage I → Stage II → Stage III

No AMFRS mechanisms (novelty archive, Pareto ranking, adaptive controller).

Examples (from ``raise_env/`` with system Python)::

    # Seconds-scale dry run (stub trainers + seed LLM)
    python scripts/run_raise.py --fast --output-dir results/raise_fast

    # Practical local run (real trainers, reduced Stage III K3)
    python scripts/run_raise.py --llm seed --output-dir results/raise_local

    # Paper-faithful Stage III budget on a GPU cluster
    python scripts/run_raise.py --llm vllm --stage3-train-steps 10000000 \\
        --device cuda --output-dir results/raise_paper

Named run profiles (``raise_core.presets.CLOSED_LOOP_PROFILES``)::

    python scripts/run_raise.py --profile smoke   # real A2C/PPO, tiny budgets
    python scripts/run_raise.py --profile 1h
    python scripts/run_raise.py --profile 12h --warm-surrogate artifacts/surr_warm
    python scripts/run_raise.py --profile 18h
    python scripts/run_raise.py --profile paper_scale --seeds 425,426 --llm groq

    # Resume: rerun with the same --output-dir (resume is on by default)
    python scripts/run_raise.py --profile 12h --output-dir results/raise_12h_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401 — arms domains/crowdnav/runtime on sys.path
import argparse
import logging
import os
import shutil
import sys
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASELINES_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "baselines_openai"))
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)
os.chdir(_ROOT)


def _flag_value(argv: list, flag: str):
    """Last value given for ``flag`` (``--x V`` or ``--x=V``), else None."""
    val = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            val = argv[i + 1]
        elif isinstance(a, str) and a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


def _inject_profile_argv(argv: list) -> tuple:
    """
    Strip ``--profile``; prepend profile flags so later CLI args override.

    Profiles with ``output_dir_prefix`` get a stamped ``--output-dir`` and,
    when closed-loop, nested ``surrogate_model`` / ``surrogate_dataset`` dirs
    unless the CLI passes them. Returns ``(argv, profile_spec or {})``.
    """
    profile = None
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--profile" and i + 1 < len(argv):
            profile = argv[i + 1]
            i += 2
            continue
        if isinstance(a, str) and a.startswith("--profile="):
            profile = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    if not profile:
        return out, {}
    from raise_core.presets import get_closed_loop_profile, profile_to_run_raise_argv

    spec = get_closed_loop_profile(profile)
    if spec.get("runner"):
        return out, spec
    paths = []
    out_dir = _flag_value(out, "--output-dir")
    if out_dir is None and spec.get("output_dir_prefix"):
        out_dir = spec["output_dir_prefix"] + datetime.now().strftime("%Y%m%d_%H%M%S")
        paths += ["--output-dir", out_dir]
    if out_dir is not None and spec.get("closed_loop"):
        if _flag_value(out, "--surrogate") is None:
            paths += ["--surrogate", os.path.join(out_dir, "surrogate_model")]
        if _flag_value(out, "--surrogate-dataset") is None:
            paths += ["--surrogate-dataset", os.path.join(out_dir, "surrogate_dataset")]
    return profile_to_run_raise_argv(profile) + paths + out, spec


def _resolve_warm_surrogate(warm_root: str) -> tuple:
    """``warm_root/{model,dataset}`` or one dir holding model.joblib + features.jsonl."""
    root = os.path.abspath(warm_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Warm surrogate dir not found: {root}")
    model_dir = os.path.join(root, "model")
    data_dir = os.path.join(root, "dataset")
    if not (os.path.isdir(model_dir) and os.path.isdir(data_dir)):
        model_dir, data_dir = root, root
    if not os.path.isfile(os.path.join(model_dir, "model.joblib")):
        raise FileNotFoundError(
            f"No model.joblib under {model_dir!r} "
            f"(expected {root}/model/model.joblib or {root}/model.joblib)"
        )
    if not os.path.isfile(os.path.join(data_dir, "features.jsonl")):
        raise FileNotFoundError(
            f"No features.jsonl under {data_dir!r} "
            f"(expected {root}/dataset/features.jsonl or {root}/features.jsonl)"
        )
    return model_dir, data_dir


def _copy_warm_surrogate(args) -> int:
    """Copy a bootstrapped surrogate into this run's surrogate dirs (fresh runs only)."""
    if not args.closed_loop or not args.surrogate:
        print(
            "--warm-surrogate requires --closed-loop with --surrogate "
            "(e.g. --profile 12h).",
            file=sys.stderr,
        )
        return 2
    ckpt = os.path.join(args.output_dir, "closed_loop", "checkpoint.json")
    if args.resume and os.path.isfile(ckpt):
        print(
            f"{ckpt} exists — drop --warm-surrogate to resume, "
            "or pick a new --output-dir.",
            file=sys.stderr,
        )
        return 2
    try:
        src_model, src_data = _resolve_warm_surrogate(args.warm_surrogate)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    os.makedirs(args.surrogate, exist_ok=True)
    os.makedirs(args.surrogate_dataset, exist_ok=True)
    for name in os.listdir(src_model):
        if name.startswith("_stage2"):
            continue  # bulky Stage II train trees from bootstrap
        s = os.path.join(src_model, name)
        d = os.path.join(args.surrogate, name)
        if os.path.isdir(s):
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    for name in os.listdir(src_data):
        s = os.path.join(src_data, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.surrogate_dataset, name))
    with open(
        os.path.join(args.surrogate_dataset, "features.jsonl"), encoding="utf-8"
    ) as fh:
        n = sum(1 for line in fh if line.strip())
    print(
        f"Warm surrogate copied → {args.surrogate} , {args.surrogate_dataset} "
        f"(n_features≈{n})"
    )
    return 0


def main() -> int:
    rest, profile_spec = _inject_profile_argv(sys.argv[1:])
    if profile_spec.get("runner") == "paper_scale":
        from raise_core.paper_scale import run_paper_scale_cli

        return run_paper_scale_cli(rest)
    sys.argv = [sys.argv[0]] + rest

    from raise_core.pipeline import RaisePipeline, RaiseRunConfig
    from raise_core.validate import STAGE3_PAPER_STEPS, STAGE3_STEPS

    parser = argparse.ArgumentParser(description="RAISE end-to-end")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print effective argparse namespace (after --profile merge) and exit",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Named preset (smoke/1h/12h/18h/highway/paper_scale); applied "
            "before other flags"
        ),
    )
    parser.add_argument(
        "--warm-surrogate",
        type=str,
        default=None,
        help=(
            "Closed-loop: copy a bootstrapped surrogate (DIR/model + DIR/dataset, "
            "see scripts/bootstrap_surrogate.py) into --surrogate / "
            "--surrogate-dataset before a fresh run"
        ),
    )
    parser.add_argument("--output-dir", type=str, default="results/raise_run")
    parser.add_argument("--seed", type=int, default=425)
    parser.add_argument(
        "--domain",
        type=str,
        default="crowdnav",
        help="Domain pack under raise_core.domains (default: crowdnav baseline)",
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
        default="domains/crowdnav/data/stage1_dataset",
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
            "Named budget preset. 'paper' points to --profile paper_scale "
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
        default="domains/crowdnav/data/surrogate_dataset",
        help="Surrogate label jsonl root for AL append/refit",
    )
    parser.add_argument(
        "--closed-loop",
        action="store_true",
        help=(
            "Innovation path: Gen epochs with Score1 -> Surrogate gate -> "
            "in-loop AL -> Stage II short labels -> refit (not paper Alg.1 linear)"
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
        default=8000,
        help="Stage II short budget inside closed-loop epochs (default 8000)",
    )
    parser.add_argument(
        "--closed-loop-min-labels-gate",
        type=int,
        default=24,
        help="Soft Surrogate gate until this many labels exist (default 24)",
    )
    parser.add_argument(
        "--closed-loop-max-val-mae-gate",
        type=float,
        default=None,
        help=(
            "Optional: enable hard gate early when mean surrogate val MAE "
            "≤ this (highway Phase 4; default off)"
        ),
    )
    parser.add_argument(
        "--closed-loop-evolve-rank",
        default="",
        choices=("", "score1", "scalar", "hybrid", "pareto"),
        help=(
            "Parent order for next generation after an epoch: score1 (CrowdNav "
            "default), scalar (highway default: highway_fitness), "
            "pareto (diagnostic NSGA-II), hybrid. "
            "Empty → scalar for highway, score1 otherwise"
        ),
    )
    parser.add_argument(
        "--closed-loop-evolve-rank-score1-weight",
        type=float,
        default=0.4,
        help="Weight on Score1 inside hybrid evolve-rank (default 0.4)",
    )
    parser.add_argument(
        "--highway-n-envs",
        type=int,
        default=1,
        help="DummyVecEnv count for highway PPO (same K2; default 1)",
    )
    parser.add_argument(
        "--highway-label-workers",
        type=int,
        default=1,
        help="Parallel Stage II label threads for highway (default 1)",
    )
    parser.add_argument(
        "--highway-eval-mode",
        choices=("both", "holdout_only"),
        default="both",
        help="highway Stage II eval: both train+holdout, or holdout_only",
    )
    parser.add_argument(
        "--highway-warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start highway PPO from parent checkpoint when available",
    )
    parser.add_argument(
        "--highway-action-mode",
        choices=("meta_default", "meta_fine", "continuous"),
        default="meta_default",
        help=(
            "Highway action space: meta_default=11 gears DiscreteMetaAction (default), "
            "meta_fine=linspace gears, continuous=ContinuousAction Box"
        ),
    )
    parser.add_argument(
        "--highway-action-continuous-lateral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ContinuousAction: include steering (Box(2,)); False → accel-only Box(1,)",
    )
    parser.add_argument(
        "--highway-meta-fine-n",
        type=int,
        default=21,
        help="meta_fine: number of target_speeds in linspace(low, high)",
    )
    parser.add_argument(
        "--highway-calibration-mode",
        choices=("no_speed_floor", "env_measured", "population"),
        default="no_speed_floor",
        help=(
            "Highway Pareto feasibility: no_speed_floor=SR>0 only (default); "
            "env_measured=SR>0 + measured non-ego traffic speeds shown as info; "
            "population=legacy percentile speed floor + CR/TR ceilings"
        ),
    )
    parser.add_argument(
        "--highway-pareto-use-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include forward progress as a Pareto objective (default on)",
    )
    parser.add_argument(
        "--highway-pareto-use-lane-change",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include lane_change_rate as a Pareto objective (default on)",
    )
    parser.add_argument(
        "--highway-pareto-use-overtake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include overtakes_per_km as a Pareto objective (default on)",
    )
    parser.add_argument(
        "--highway-elite-archive",
        choices=("auto", "pareto", "fitness"),
        default="auto",
        help=(
            "Runtime elite kept across generations: pareto rank-0 or legacy "
            "highway_fitness; auto=fitness only with --highway-calibration-mode population"
        ),
    )
    parser.add_argument(
        "--diagnostics-groundtruth-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Highway only: during PPO, run cheap holdout evals at 20/40/60/80/100%% "
            "of K2/K3 and log to diagnostics_groundtruth.jsonl (observational; "
            "default off)"
        ),
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
    parser.add_argument(
        "--closed-loop-proxy-feedback",
        action="store_true",
        help=(
            "Attach selective Stage-II SR/CR/TR blocks to D.2 mutation prompts "
            "and reflection (after min labels + epoch>=1)"
        ),
    )
    parser.add_argument(
        "--closed-loop-proxy-feedback-min-labels",
        type=int,
        default=16,
        help="Min surrogate labels before proxy feedback attaches (default 16)",
    )
    parser.add_argument(
        "--closed-loop-proxy-d3",
        type=int,
        default=0,
        help=(
            "Max in-loop D.3 rewrites per epoch from worst proxy scalars "
            "(0=off; requires --closed-loop-proxy-feedback)"
        ),
    )
    parser.add_argument(
        "--closed-loop-refine",
        action="store_true",
        help="Legacy: enable 1 in-loop D.3 per epoch (same as --closed-loop-proxy-d3 1)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help=(
            "Resume from checkpoints when present (default: on). "
            "Closed-loop: --output-dir/closed_loop/checkpoint.json. "
            "Stage III: --output-dir/stage3/checkpoint.json (+ mid-PPO progress)."
        ),
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing closed-loop / Stage III checkpoints and start fresh",
    )
    parser.add_argument(
        "--stage3-save-interval",
        type=int,
        default=50,
        help=(
            "Write Stage III PPO weights every N updates for mid-train resume "
            "(0 = final save only; default 50)"
        ),
    )

    args = parser.parse_args()

    if getattr(args, "print_config", False):
        for k, v in sorted(vars(args).items()):
            print(f"{k}={v!r}")
        return 0

    if args.scale == "paper":
        print(
            "Paper-scale (Tables 3–6, multi-seed) is only available via:\n"
            "  python scripts/run_raise.py --profile paper_scale\n"
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
            "provider - pass --llm groq|ollama|vllm explicitly, or pass "
            "--allow-seed-llm if this is intentional (e.g. debugging Stage II/III "
            "wiring without LLM cost).",
            file=sys.stderr,
        )
        return 2

    if args.warm_surrogate:
        code = _copy_warm_surrogate(args)
        if code:
            return code

    # Protect Config.get_args() class-body from our CLI flags.
    sys.argv = [sys.argv[0], "--no-cuda" if args.device == "cpu" else "--seed", str(args.seed)]

    from raise_core import console as _console

    _console.configure_run_logging(verbose=bool(args.verbose))

    import crowd_sim  # noqa: F401

    cfg = RaiseRunConfig(
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
        closed_loop_max_val_mae_for_gate=(
            float(args.closed_loop_max_val_mae_gate)
            if args.closed_loop_max_val_mae_gate is not None
            else None
        ),
        closed_loop_evolve_rank=str(args.closed_loop_evolve_rank or ""),
        closed_loop_evolve_rank_score1_weight=float(
            args.closed_loop_evolve_rank_score1_weight
        ),
        highway_n_envs=max(1, int(args.highway_n_envs)),
        highway_label_workers=max(1, int(args.highway_label_workers)),
        highway_warm_start=bool(args.highway_warm_start),
        highway_eval_mode=str(args.highway_eval_mode or "both"),
        highway_diagnostics_groundtruth=bool(
            args.diagnostics_groundtruth_checkpoints
        ),
        highway_action_mode=str(args.highway_action_mode or "meta_default"),
        highway_action_continuous_lateral=bool(
            args.highway_action_continuous_lateral
        ),
        highway_meta_fine_n=int(args.highway_meta_fine_n),
        highway_calibration_mode=str(args.highway_calibration_mode),
        highway_pareto_use_progress=bool(args.highway_pareto_use_progress),
        highway_pareto_use_lane_change=bool(args.highway_pareto_use_lane_change),
        highway_pareto_use_overtake=bool(args.highway_pareto_use_overtake),
        highway_elite_archive=str(args.highway_elite_archive),
        closed_loop_k2=int(args.closed_loop_k2),
        closed_loop_refit_every_new_labels=int(args.closed_loop_refit_every),
        closed_loop_min_stage2_per_gen=int(args.closed_loop_min_stage2),
        closed_loop_resume=bool(args.resume),
        closed_loop_proxy_feedback=bool(args.closed_loop_proxy_feedback),
        closed_loop_proxy_feedback_min_labels=int(
            args.closed_loop_proxy_feedback_min_labels
        ),
        closed_loop_proxy_feedback_d3_per_epoch=int(args.closed_loop_proxy_d3),
        closed_loop_enable_refine=bool(args.closed_loop_refine),
        stage3_resume=bool(args.resume),
        stage3_save_interval_updates=int(args.stage3_save_interval),
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

    from raise_core.domains import available_domains, load_domain

    try:
        pack = load_domain(cfg.domain)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        print(
            f"Available domains: {', '.join(available_domains())}",
            file=sys.stderr,
        )
        return 2

    # If caller left the CrowdNav Stage I default path, prefer the pack default.
    if (
        str(cfg.stage1_dataset_path).replace("\\", "/") == "domains/crowdnav/data/stage1_dataset"
        and pack.stage1_dataset_default
        and pack.name != "crowdnav"
    ):
        cfg.stage1_dataset_path = str(pack.stage1_dataset_default)

    logging.info(
        "RAISE -> %s (domain=%s, fast=%s, easy=%s, humans=%d, predict=%s, K3=%d)",
        cfg.output_dir,
        cfg.domain,
        cfg.fast,
        bool(args.easy),
        cfg.human_num,
        cfg.predict_method,
        cfg.stage3_train_steps,
    )
    artifacts = RaisePipeline(cfg).run()
    logging.info("Done. Final candidate: %s", artifacts.best_stage3.candidate_id)
    logging.info("Artifacts: %s", artifacts.output_dir)
    report = os.path.join(cfg.output_dir, "closed_loop", "REPORT.txt")
    if profile_spec and cfg.closed_loop and os.path.isfile(report):
        with open(report, encoding="utf-8") as fh:
            print("\n--- closed_loop/REPORT.txt ---\n" + fh.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

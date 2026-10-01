#!/usr/bin/env python
"""RAISE full closed-loop on highway-fast-v0.

  closed-loop (Score1 ↔ Stage II short ↔ Surrogate gate + AL + proxy
  feedback) → Stage III validate

Budget (N, G, K2, K3, …) lives in ``raise_core.presets.CLOSED_LOOP_PROFILES
["highway"]``; edit it there (or pass --population / --generations) to change
run length.

From raise_env/:

  # once (auto-run if Stage I dataset missing)
  python scripts/collect_highway_stage1_dataset.py

  python scripts/run_raise_highway.py --llm groq
  python scripts/run_raise_highway.py --llm groq --resume results/highway_YYYYMMDD_HHMMSS

  # Re-generate plots offline:
  python scripts/plot_raise_run.py --run-dir results/highway_YYYYMMDD_HHMMSS

  # optional warm surrogate
  python scripts/bootstrap_surrogate.py --domain highway --llm groq
  python scripts/run_raise_highway.py --llm groq --warm-surrogate artifacts/highway_surr_warm
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
os.chdir(_ROOT)
for _path in (_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from raise_core.presets import CLOSED_LOOP_PROFILES  # noqa: E402

CHECKPOINT_LOCATIONS = (
    os.path.join("closed_loop", "checkpoint.json"),
    "checkpoint.json",
)

PROFILE_NAME = "highway"
PROFILE = dict(CLOSED_LOOP_PROFILES[PROFILE_NAME])


def _dataset_ready(path: str) -> bool:
    traj = os.path.join(path, "trajectories.jsonl")
    return os.path.isfile(traj) and os.path.getsize(traj) > 0


def _warm_ready(warm_root: str) -> bool:
    model = os.path.join(warm_root, "model", "model.joblib")
    feats = os.path.join(warm_root, "dataset", "features.jsonl")
    return os.path.isfile(model) and os.path.isfile(feats)


def _ensure_dataset(path: str, *, episodes_per_behavior: int = 10) -> int:
    if _dataset_ready(path):
        return 0
    print(f"Stage I dataset missing at {path}; collecting…")
    cmd = [
        sys.executable,
        os.path.join(_SCRIPTS, "collect_highway_stage1_dataset.py"),
        "--out",
        path,
        "--episodes-per-behavior",
        str(int(episodes_per_behavior)),
        "--seed",
        str(int(PROFILE["seed"])),
    ]
    return int(subprocess.call(cmd))


def _ensure_warm(warm_root: str, *, llm: str, device: str) -> int:
    if _warm_ready(warm_root):
        return 0
    print(f"Warm surrogate missing at {warm_root}; bootstrapping…")
    cmd = [
        sys.executable,
        os.path.join(_SCRIPTS, "bootstrap_surrogate.py"),
        "--domain",
        "highway",
        "--stage1-dataset",
        PROFILE["stage1_dataset"],
        "--out",
        os.path.join(warm_root, "dataset"),
        "--model-out",
        os.path.join(warm_root, "model"),
        "--n-candidates",
        str(int(PROFILE["warm_bootstrap_n"])),
        "--stage2-train-steps",
        str(int(PROFILE["warm_k2"])),
        "--k2-unit",
        PROFILE["k2_unit"],
        "--llm",
        str(llm),
        "--device",
        str(device),
        "--seed",
        str(int(PROFILE["seed"])),
        "--eval-episodes",
        str(int(PROFILE["stage2_eval"])),
    ]
    if str(llm) == "seed":
        # bootstrap itself does not need allow-seed; seed provider is local
        pass
    return int(subprocess.call(cmd))


def _load_resume_paths(resume_dir: str) -> tuple[str, str, str]:
    out = os.path.abspath(resume_dir)
    payload: dict = {}
    for rel in CHECKPOINT_LOCATIONS:
        path = os.path.join(out, rel)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            break
    if not payload and not os.path.isdir(out):
        raise FileNotFoundError(f"Resume dir not found: {out}")
    extra = payload.get("extra") or {}
    surr_model = (
        payload.get("surrogate_model_dir")
        or extra.get("surrogate_model_dir")
        or os.path.join(out, "surrogate_model")
    )
    surr_data = (
        payload.get("surrogate_dataset")
        or extra.get("surrogate_dataset")
        or os.path.join(out, "surrogate_dataset")
    )
    return out, str(surr_model), str(surr_data)


def _resolve_warm_surrogate(warm_root: str) -> tuple[str, str]:
    root = os.path.abspath(warm_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Warm surrogate dir not found: {root}")
    model = os.path.join(root, "model")
    data = os.path.join(root, "dataset")
    if not os.path.isfile(os.path.join(model, "model.joblib")):
        raise FileNotFoundError(f"Missing {model}/model.joblib")
    if not os.path.isfile(os.path.join(data, "features.jsonl")):
        raise FileNotFoundError(f"Missing {data}/features.jsonl")
    return model, data


def _copy_warm_into_run(src_model: str, src_data: str, dst_model: str, dst_data: str) -> int:
    if os.path.isdir(dst_model):
        shutil.rmtree(dst_model)
    if os.path.isdir(dst_data):
        shutil.rmtree(dst_data)
    shutil.copytree(src_model, dst_model)
    shutil.copytree(src_data, dst_data)
    feats = os.path.join(dst_data, "features.jsonl")
    if not os.path.isfile(feats):
        return 0
    with open(feats, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _print_profile(args, *, warm_n=None, population=None, generations=None) -> None:
    print(f"=== highway RAISE closed-loop PROFILE ({PROFILE_NAME}) ===")
    shown = dict(PROFILE)
    if population is not None:
        shown["population"] = int(population)
    if generations is not None:
        shown["generations"] = int(generations)
    for key in (
        "domain",
        "seed",
        "population",
        "generations",
        "k2",
        "k2_unit",
        "min_labels_gate",
        "max_val_mae_gate",
        "al_max",
        "min_stage2",
        "refit_every",
        "proxy_feedback",
        "proxy_d3_per_epoch",
        "stage2_eval",
        "stage3_k3",
        "stage3_eval",
        "stage3_rounds",
        "warm_bootstrap_n",
        "warm_k2",
        "elitism",
        "evolve_rank",
        "evolve_rank_score1_weight",
        "highway_n_envs",
        "highway_label_workers",
        "highway_warm_start",
        "highway_eval_mode",
        "highway_calibration_mode",
        "highway_pareto_use_progress",
        "highway_pareto_use_lane_change",
        "highway_pareto_use_overtake",
        "highway_elite_archive",
    ):
        print(f"  {key}: {shown.get(key)}")
    print(f"  llm: {args.llm}")
    print(f"  device: {args.device}")
    print(f"  dataset: {PROFILE['stage1_dataset']}")
    print(f"  warm_root: {PROFILE['warm_root']}")
    print(
        f"  diagnostics_groundtruth: "
        f"{bool(getattr(args, 'diagnostics_groundtruth_checkpoints', False))}"
    )
    if warm_n is not None:
        print(f"  warm labels copied: ≈{warm_n}")
    print("============================================")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default=PROFILE["llm"], choices=("seed", "groq", "ollama", "vllm"))
    parser.add_argument("--device", default=PROFILE["device"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: <profile output_dir_prefix>YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--warm-surrogate",
        default=PROFILE["warm_root"],
        help=(
            "Optional warm surrogate parent (model/ + dataset/). "
            "Default empty = skip warm; collect labels inside closed-loop."
        ),
    )
    parser.add_argument(
        "--resume",
        default="",
        help="Resume a previous results/highway_* directory",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Do not auto-collect Stage I dataset if missing",
    )
    parser.add_argument(
        "--skip-warm",
        action="store_true",
        help="Do not auto-bootstrap warm surrogate if missing",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-prereq-check",
        action="store_true",
        help="Start even if Groq keys / dataset pre-flight checks fail",
    )
    parser.add_argument(
        "--diagnostics-groundtruth-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Observational: cheap mid-train holdout evals at 20/40/60/80/100%% of "
            "each candidate's train budget (see diagnostics_groundtruth.jsonl). "
            "Default off — enables Part 2 of the adaptive-budget study."
        ),
    )
    parser.add_argument(
        "--highway-action-mode",
        choices=("meta_default", "meta_fine", "continuous"),
        default=None,
        help="Override PROFILE highway_action_mode (default: meta_default)",
    )
    parser.add_argument(
        "--no-highway-action-continuous-lateral",
        action="store_true",
        help="ContinuousAction: acceleration only (Box(1,))",
    )
    parser.add_argument(
        "--highway-calibration-mode",
        choices=("no_speed_floor", "env_measured", "population"),
        default=None,
        help="Override PROFILE highway_calibration_mode (default: no_speed_floor)",
    )
    parser.add_argument(
        "--no-highway-pareto-use-progress",
        action="store_true",
        help="Drop forward progress from the Pareto objectives",
    )
    parser.add_argument(
        "--no-highway-pareto-use-lane-change",
        action="store_true",
        help="Drop lane_change_rate from the Pareto objectives",
    )
    parser.add_argument(
        "--no-highway-pareto-use-overtake",
        action="store_true",
        help="Drop overtakes_per_km from the Pareto objectives",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Override PROFILE population (closed-loop pop size / Stage I N)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override PROFILE generations (closed-loop epochs / Stage I G)",
    )
    parser.add_argument(
        "--highway-elite-archive",
        choices=("auto", "pareto", "fitness"),
        default=None,
        help="Override PROFILE highway_elite_archive (default: auto)",
    )
    args = parser.parse_args()

    pop_n = int(args.population if args.population is not None else PROFILE["population"])
    gen_n = int(
        args.generations if args.generations is not None else PROFILE["generations"]
    )
    if pop_n < 2:
        raise SystemExit("--population must be >= 2")
    if gen_n < 1:
        raise SystemExit("--generations must be >= 1")

    from _prereqs import check_groq_keys, report_problems

    if not args.skip_prereq_check:
        problems = check_groq_keys(_ROOT, args.llm)
        if not report_problems(problems, stream=sys.stderr):
            print(
                "\nFix credentials, or run without Groq:\n"
                "  python scripts/run_raise_highway.py --llm seed\n"
                "Or copy raise_env/groq_keys.json.example → groq_keys.json",
                file=sys.stderr,
            )
            return 2

    if str(args.resume).strip() and str(args.warm_surrogate).strip():
        # Resume owns its surrogate paths; warm copy is skipped below.
        pass

    if not args.skip_collect:
        code = _ensure_dataset(PROFILE["stage1_dataset"])
        if code != 0:
            print("Dataset collection failed.", file=sys.stderr)
            return code
    elif not _dataset_ready(PROFILE["stage1_dataset"]):
        print(
            f"Missing {PROFILE['stage1_dataset']}/trajectories.jsonl.",
            file=sys.stderr,
        )
        return 2

    warm_root = str(args.warm_surrogate).strip()
    if not str(args.resume).strip() and warm_root and not args.skip_warm:
        code = _ensure_warm(warm_root, llm=str(args.llm), device=str(args.device))
        if code != 0:
            print("Warm surrogate bootstrap failed.", file=sys.stderr)
            return code

    if str(args.resume).strip():
        try:
            out, surr_model, surr_data = _load_resume_paths(str(args.resume).strip())
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        resuming = True
        warm_n = None
    else:
        out = str(args.output_dir).strip()
        if not out:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = f"{PROFILE['output_dir_prefix']}{stamp}"
        surr_model = os.path.join(out, "surrogate_model")
        surr_data = os.path.join(out, "surrogate_dataset")
        resuming = False
        warm_n = None
        if warm_root:
            try:
                src_model, src_data = _resolve_warm_surrogate(warm_root)
            except FileNotFoundError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            os.makedirs(out, exist_ok=True)
            warm_n = _copy_warm_into_run(src_model, src_data, surr_model, surr_data)
            print(
                f"Warm surrogate copied → {surr_model} , {surr_data} "
                f"(n_features≈{warm_n})"
            )
    for path in (out, surr_model, surr_data):
        os.makedirs(path, exist_ok=True)

    _print_profile(args, warm_n=warm_n, population=pop_n, generations=gen_n)

    cmd = [
        sys.executable,
        os.path.join(_SCRIPTS, "run_raise.py"),
        "--domain",
        PROFILE["domain"],
        "--closed-loop",
        "--llm",
        str(args.llm),
        "--device",
        str(args.device),
        "--stage1-dataset",
        PROFILE["stage1_dataset"],
        "--score1",
        "dataset",
        "--stage1-population",
        str(pop_n),
        "--stage1-generations",
        str(gen_n),
        "--closed-loop-k2",
        str(PROFILE["k2"]),
        "--k2-unit",
        PROFILE["k2_unit"],
        "--closed-loop-min-labels-gate",
        str(PROFILE["min_labels_gate"]),
        "--closed-loop-max-val-mae-gate",
        str(PROFILE["max_val_mae_gate"]),
        "--closed-loop-al-max",
        str(PROFILE["al_max"]),
        "--closed-loop-min-stage2",
        str(PROFILE["min_stage2"]),
        "--closed-loop-refit-every",
        str(PROFILE["refit_every"]),
        "--closed-loop-proxy-feedback",
        "--closed-loop-proxy-feedback-min-labels",
        str(PROFILE["proxy_feedback_min_labels"]),
        "--closed-loop-proxy-d3",
        str(PROFILE["proxy_d3_per_epoch"]),
        "--surrogate",
        surr_model,
        "--surrogate-dataset",
        surr_data,
        "--stage2-eval-episodes",
        str(PROFILE["stage2_eval"]),
        "--stage3-rounds",
        str(PROFILE["stage3_rounds"]),
        "--stage3-train-steps",
        str(PROFILE["stage3_k3"]),
        "--stage3-eval-episodes",
        str(PROFILE["stage3_eval"]),
        "--no-h-sweep",
        "--seed",
        str(PROFILE["seed"]),
        "--output-dir",
        out,
        "--final-rank",
        "scalar",
        "--resume",
        "--allow-seed-llm",
    ]
    if bool(PROFILE.get("elitism")):
        cmd.append("--elitism")
    er = str(PROFILE.get("evolve_rank") or "").strip()
    if er:
        cmd.extend(["--closed-loop-evolve-rank", er])
        cmd.extend(
            [
                "--closed-loop-evolve-rank-score1-weight",
                str(PROFILE.get("evolve_rank_score1_weight", 0.4)),
            ]
        )
    cmd.extend(
        [
            "--highway-n-envs",
            str(int(PROFILE.get("highway_n_envs", 4))),
            "--highway-label-workers",
            str(int(PROFILE.get("highway_label_workers", 1))),
            "--highway-eval-mode",
            str(PROFILE.get("highway_eval_mode") or "both"),
        ]
    )
    if bool(PROFILE.get("highway_warm_start", True)):
        cmd.append("--highway-warm-start")
    else:
        cmd.append("--no-highway-warm-start")
    action_mode = str(
        args.highway_action_mode
        or PROFILE.get("highway_action_mode")
        or "meta_default"
    )
    cmd.extend(["--highway-action-mode", action_mode])
    if bool(args.no_highway_action_continuous_lateral) or not bool(
        PROFILE.get("highway_action_continuous_lateral", True)
    ):
        cmd.append("--no-highway-action-continuous-lateral")
    else:
        cmd.append("--highway-action-continuous-lateral")
    calib_mode = str(
        args.highway_calibration_mode
        or PROFILE.get("highway_calibration_mode")
        or "no_speed_floor"
    )
    cmd.extend(["--highway-calibration-mode", calib_mode])
    if bool(args.no_highway_pareto_use_progress) or not bool(
        PROFILE.get("highway_pareto_use_progress", True)
    ):
        cmd.append("--no-highway-pareto-use-progress")
    else:
        cmd.append("--highway-pareto-use-progress")
    if bool(args.no_highway_pareto_use_lane_change) or not bool(
        PROFILE.get("highway_pareto_use_lane_change", True)
    ):
        cmd.append("--no-highway-pareto-use-lane-change")
    else:
        cmd.append("--highway-pareto-use-lane-change")
    if bool(args.no_highway_pareto_use_overtake) or not bool(
        PROFILE.get("highway_pareto_use_overtake", True)
    ):
        cmd.append("--no-highway-pareto-use-overtake")
    else:
        cmd.append("--highway-pareto-use-overtake")
    elite_archive = str(
        args.highway_elite_archive or PROFILE.get("highway_elite_archive") or "auto"
    )
    cmd.extend(["--highway-elite-archive", elite_archive])
    if args.llm_model:
        cmd.extend(["--llm-model", str(args.llm_model)])
    if args.verbose:
        cmd.append("--verbose")
    if bool(args.diagnostics_groundtruth_checkpoints):
        cmd.append("--diagnostics-groundtruth-checkpoints")

    resume_cmd = f"python scripts/run_raise_highway.py --resume {out}"
    if bool(args.diagnostics_groundtruth_checkpoints):
        resume_cmd += " --diagnostics-groundtruth-checkpoints"
    if args.highway_action_mode:
        resume_cmd += f" --highway-action-mode {args.highway_action_mode}"
    if args.highway_calibration_mode:
        resume_cmd += f" --highway-calibration-mode {args.highway_calibration_mode}"
    if args.no_highway_pareto_use_progress:
        resume_cmd += " --no-highway-pareto-use-progress"
    if args.no_highway_pareto_use_lane_change:
        resume_cmd += " --no-highway-pareto-use-lane-change"
    if args.no_highway_pareto_use_overtake:
        resume_cmd += " --no-highway-pareto-use-overtake"
    if args.highway_elite_archive:
        resume_cmd += f" --highway-elite-archive {args.highway_elite_archive}"
    if args.population is not None:
        resume_cmd += f" --population {pop_n}"
    if args.generations is not None:
        resume_cmd += f" --generations {gen_n}"
    print("cmd:", " ".join(cmd))
    print(f"mode:   {'RESUME' if resuming else 'NEW'}")
    print(f"action_mode: {action_mode}")
    print(f"calibration_mode: {calib_mode}  elite_archive: {elite_archive}")
    print(f"population={pop_n}  generations={gen_n}  (K2={PROFILE['k2']} K3={PROFILE['stage3_k3']})")
    print(f"output: {out}")
    print(f"If interrupted: {resume_cmd}")
    code = int(subprocess.call(cmd))
    report = os.path.join(out, "closed_loop", "REPORT.txt")
    if os.path.isfile(report):
        print()
        print("--- closed_loop/REPORT.txt ---")
        with open(report, encoding="utf-8") as fh:
            print(fh.read())
    if code == 0:
        plots_dir = os.path.join(out, "plots")
        already = os.path.isdir(plots_dir) and any(
            f.endswith((".png", ".gif"))
            for _, _, files in os.walk(plots_dir)
            for f in files
        )
        # Pipeline writes plots/viz at end; only fill gaps (e.g. older resumes).
        if already:
            print()
            print(f"--- plots already present: {plots_dir} ---")
        else:
            try:
                from domains.highway.post_run import write_highway_run_artifacts

                print()
                print("--- plots + animation ---")
                art = write_highway_run_artifacts(
                    out, episodes=2, seed=int(PROFILE["seed"])
                )
                for p in art.get("plots") or []:
                    print(f"  plot: {p}")
                viz = art.get("viz") or {}
                if viz.get("ok"):
                    for p in viz.get("written") or []:
                        print(f"  viz:  {p}")
                elif viz:
                    print(f"  viz skipped: {viz.get('reason')}")
            except Exception as exc:  # noqa: BLE001
                print(f"post-run artifacts failed (non-fatal): {exc}", file=sys.stderr)
    if code != 0:
        print(f"Run exited {code}. Resume with: {resume_cmd}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

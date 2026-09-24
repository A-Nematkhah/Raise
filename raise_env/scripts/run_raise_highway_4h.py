#!/usr/bin/env python
"""RAISE full closed-loop on highway-fast-v0 — ~4h wall budget.

This is the thesis RAISE path (not Stage I→II→III without surrogate):

  warm surrogate bootstrap → closed-loop (Score1 ↔ Stage II short ↔
  Surrogate gate + AL + proxy feedback) → Stage III validate

Locked PROFILE (SB3 PPO, env_steps):

  Warm:   n=16 labels @ K2=15_000 env steps
  Loop:   N=6, G=4, K2=15_000, min_labels_gate=12, AL=2/epoch
  Stage III: R=1, K3=80_000, E3=30, no H-sweep

From raise_env/:

  # once (auto-run if Stage I dataset missing)
  python scripts/collect_highway_stage1_dataset.py

  # ~4h RAISE — warm surrogate OFF by default; labels collected in closed-loop
  python scripts/run_raise_highway_4h.py
  python scripts/run_raise_highway_4h.py --llm groq

  # optional later: warm bootstrap
  python scripts/bootstrap_surrogate.py --domain highway --llm groq
  python scripts/run_raise_highway_4h.py --llm groq --warm-surrogate artifacts/highway_surr_warm
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

CHECKPOINT_LOCATIONS = (
    os.path.join("closed_loop", "checkpoint.json"),
    "checkpoint.json",
)

PROFILE = {
    "domain": "highway",
    "seed": 425,
    "llm": "seed",
    "device": "cuda",
    "stage1_dataset": "domains/highway/data/stage1_dataset",
    # Warm disabled by default: labels are collected inside closed-loop.
    "warm_root": "",
    "warm_bootstrap_n": 16,
    "warm_k2": 15_000,
    "population": 6,
    "generations": 4,
    "k2": 15_000,
    "k2_unit": "env_steps",
    "min_labels_gate": 12,
    "al_max": 2,
    "min_stage2": 3,
    "refit_every": 6,
    "proxy_feedback": True,
    "proxy_feedback_min_labels": 12,
    "proxy_d3_per_epoch": 1,
    "stage2_eval": 15,
    "stage3_k3": 80_000,
    "stage3_eval": 30,
    "stage3_rounds": 1,
}


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


def _print_profile(args, *, warm_n=None) -> None:
    print("=== highway ~4h RAISE closed-loop PROFILE ===")
    for key in (
        "domain",
        "seed",
        "population",
        "generations",
        "k2",
        "k2_unit",
        "min_labels_gate",
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
    ):
        print(f"  {key}: {PROFILE[key]}")
    print(f"  llm: {args.llm}")
    print(f"  device: {args.device}")
    print(f"  dataset: {PROFILE['stage1_dataset']}")
    print(f"  warm_root: {PROFILE['warm_root']}")
    if warm_n is not None:
        print(f"  warm labels copied: ≈{warm_n}")
    print("  estimate: ~3.5–5 h wall (GPU + seed/groq LLM)")
    print("============================================")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default=PROFILE["llm"], choices=("seed", "groq", "ollama", "vllm"))
    parser.add_argument("--device", default=PROFILE["device"])
    parser.add_argument("--llm-model", default=None)
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: results/highway_4h_YYYYMMDD_HHMMSS",
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
        help="Resume a previous results/highway_4h_* directory",
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
    args = parser.parse_args()

    from _prereqs import check_groq_keys, report_problems

    if not args.skip_prereq_check:
        problems = check_groq_keys(_ROOT, args.llm)
        if not report_problems(problems, stream=sys.stderr):
            print(
                "\nFix credentials, or run without Groq:\n"
                "  python scripts/run_raise_highway_4h.py --llm seed\n"
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
            out = f"results/highway_4h_{stamp}"
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

    _print_profile(args, warm_n=warm_n)

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
        str(PROFILE["population"]),
        "--stage1-generations",
        str(PROFILE["generations"]),
        "--closed-loop-k2",
        str(PROFILE["k2"]),
        "--k2-unit",
        PROFILE["k2_unit"],
        "--closed-loop-min-labels-gate",
        str(PROFILE["min_labels_gate"]),
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
    if args.llm_model:
        cmd.extend(["--llm-model", str(args.llm_model)])
    if args.verbose:
        cmd.append("--verbose")

    resume_cmd = f"python scripts/run_raise_highway_4h.py --resume {out}"
    print("cmd:", " ".join(cmd))
    print(f"mode:   {'RESUME' if resuming else 'NEW'}")
    print(f"output: {out}")
    print(f"If interrupted: {resume_cmd}")
    code = int(subprocess.call(cmd))
    report = os.path.join(out, "closed_loop", "REPORT.txt")
    if os.path.isfile(report):
        print()
        print("--- closed_loop/REPORT.txt ---")
        with open(report, encoding="utf-8") as fh:
            print(fh.read())
    if code != 0:
        print(f"Run exited {code}. Resume with: {resume_cmd}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

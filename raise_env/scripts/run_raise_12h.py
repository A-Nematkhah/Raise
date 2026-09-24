#!/usr/bin/env python
"""RAISE closed-loop ~12h REAL scaled run (innovation path).

Real stack (no --easy, no Validate stub):
  - GST on:            predict_method=inferred
  - Randomization on:  regime=with_random (GST *_rand checkpoint)
  - human_num=5 (scaled crowd; paper uses 20), num_processes=2 (phase=train; nproc=1 would force test)
  - Real Validate with scaled K3 / eval / rounds

Budget (~12h on one GPU): N=8, G=3, K2=8000 gradient steps, nproc=2 (phase=train), K3=350k env steps.

Warm-start Surrogate (same env, K2=8000) before the loop:

  python scripts/bootstrap_surrogate.py
  python scripts/run_raise_12h.py --warm-surrogate artifacts/surr_warm

Defaults are locked in PROFILE (human_num=5, with_random, inferred, K2=8000).

From raise_env/:
  python scripts/run_raise_12h.py

Resume after an interrupt (Ctrl-C, crash, reboot):
  python scripts/run_raise_12h.py --resume results/raise_12h_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _path in (_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _prereqs import (  # noqa: E402
    check_gst_model,
    check_groq_keys,
    check_stage1_dataset,
    report_problems,
)

CHECKPOINT_LOCATIONS = (
    os.path.join("closed_loop", "checkpoint.json"),  # current layout
    "checkpoint.json",  # mirrored copy at the run root
)

# Locked profile for this thesis run (warm bootstrap + closed loop must match).
# num_processes must be >1 so make_env sets phase='train' (nproc=1 → phase='test').
PROFILE = {
    "seed": 425,
    "llm": "groq",
    "device": "cuda",
    "num_processes": 2,
    "predict_method": "inferred",
    "regime": "with_random",
    "human_num": 5,
    "horizon_steps": 100,
    "population": 8,
    "generations": 3,
    "k2": 8000,
    "k2_unit": "gradient_steps",
    "min_labels_gate": 24,
    "al_max": 2,
    "min_stage2": 4,
    "refit_every": 8,
    "proxy_feedback": True,
    "proxy_feedback_min_labels": 16,
    "proxy_d3_per_epoch": 1,
    "stage3_k3": 350_000,
    "stage3_eval": 50,
    "stage3_rounds": 1,
    "stage3_h_sweep": False,
    "eval_episodes_stage2": 50,
    "warm_bootstrap_n": 40,
    "warm_root": "artifacts/surr_warm",
}


def _print_locked_config(args, *, warm_n=None) -> None:
    """Dump the effective profile so logs record exact settings."""
    rows = [
        ("seed", PROFILE["seed"]),
        ("llm", args.llm),
        ("device", args.device),
        ("nproc", f"{args.num_processes} → phase={'train' if int(args.num_processes) > 1 else 'test'}"),
        ("predict_method", PROFILE["predict_method"]),
        ("regime", args.regime),
        ("human_num", args.human_num),
        ("horizon", PROFILE["horizon_steps"]),
        ("N (population)", args.population),
        ("G (generations)", args.generations),
        ("K2", f"{args.k2} {PROFILE['k2_unit']}"),
        ("Stage2 eval eps", PROFILE["eval_episodes_stage2"]),
        ("min_labels_gate", args.min_labels_gate),
        ("al_max / min_stage2", f"{args.al_max} / {args.min_stage2}"),
        ("refit_every", args.refit_every),
        ("proxy_feedback", f"on (min_labels={PROFILE['proxy_feedback_min_labels']})"),
        ("proxy_d3_per_epoch", PROFILE["proxy_d3_per_epoch"]),
        ("K3 / eval / rounds", f"{args.stage3_k3} / {args.stage3_eval} / {args.stage3_rounds}"),
        ("H-sweep", "off"),
        ("warm_bootstrap_n", PROFILE["warm_bootstrap_n"]),
    ]
    if warm_n is not None:
        rows.append(("warm labels copied", warm_n))
    print("--- locked config ---")
    for k, v in rows:
        print(f"  {k:22s} {v}")
    print("---------------------")



def _load_resume_paths(resume_dir: str) -> tuple[str, str, str]:
    """Return (output_dir, surrogate_model, surrogate_dataset) for a prior run."""
    out = os.path.abspath(resume_dir)
    if not os.path.isdir(out):
        raise FileNotFoundError(f"Resume dir not found: {out}")
    tried = [os.path.join(out, rel) for rel in CHECKPOINT_LOCATIONS]
    ckpt = next((p for p in tried if os.path.isfile(p)), "")
    if not ckpt:
        raise FileNotFoundError(
            "No closed-loop checkpoint found. Looked in:\n  " + "\n  ".join(tried)
        )
    with open(ckpt, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    extra = payload.get("extra") or {}
    surr_model = str(
        payload.get("surrogate_model_dir")
        or extra.get("surrogate_model_dir")
        or os.path.join(out, "surrogate_model")
    )
    surr_data = str(
        payload.get("surrogate_dataset")
        or extra.get("surrogate_dataset")
        or os.path.join(out, "surrogate_dataset")
    )
    print(
        f"Resuming from {ckpt}\n"
        f"  status={payload.get('status')} phase={payload.get('phase')} "
        f"next_epoch={payload.get('next_epoch')} "
        f"labels={payload.get('n_labeled')}"
    )
    return out, surr_model, surr_data


def _resolve_warm_surrogate(warm_root: str) -> tuple[str, str]:
    """
    Accept either:
      warm_root/model + warm_root/dataset
    or explicit leaf dirs that already contain model.joblib / features.jsonl.
    """
    root = os.path.abspath(warm_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Warm surrogate dir not found: {root}")
    model_a = os.path.join(root, "model")
    data_a = os.path.join(root, "dataset")
    if os.path.isdir(model_a) and os.path.isdir(data_a):
        model_dir, data_dir = model_a, data_a
    else:
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


def _copy_warm_into_run(src_model: str, src_data: str, dst_model: str, dst_data: str) -> int:
    """Copy warm artifacts into isolated run paths. Returns n feature rows if known."""
    os.makedirs(dst_model, exist_ok=True)
    os.makedirs(dst_data, exist_ok=True)
    for name in os.listdir(src_model):
        if name.startswith("_stage2"):
            continue  # skip bulky Stage II train trees from bootstrap
        s = os.path.join(src_model, name)
        d = os.path.join(dst_model, name)
        if os.path.isdir(s):
            if os.path.isdir(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    for name in os.listdir(src_data):
        s = os.path.join(src_data, name)
        d = os.path.join(dst_data, name)
        if os.path.isdir(s):
            continue
        if os.path.isfile(s):
            shutil.copy2(s, d)
    n = 0
    feats = os.path.join(dst_data, "features.jsonl")
    if os.path.isfile(feats):
        with open(feats, encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip())
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default="groq", choices=("groq", "seed", "ollama", "vllm"))
    parser.add_argument("--allow-seed-llm", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=PROFILE["num_processes"],
        help=(
            "Parallel envs for Stage II/III (default 2 → phase=train; "
            "1 forces phase=test in make_env)"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a previous results/raise_12h_* directory to continue",
    )
    parser.add_argument(
        "--warm-surrogate",
        type=str,
        default="",
        help=(
            "Copy a pre-fit surrogate into this run before start. "
            "Pass parent dir with model/ + dataset/ (from bootstrap_surrogate.py)"
        ),
    )
    # Closed-loop search budget (~12h) — defaults from PROFILE
    parser.add_argument("--population", type=int, default=PROFILE["population"])
    parser.add_argument("--generations", type=int, default=PROFILE["generations"])
    parser.add_argument(
        "--k2",
        type=int,
        default=PROFILE["k2"],
        help="In-loop K2 (gradient steps; default 8000 — match warm bootstrap)",
    )
    parser.add_argument("--min-labels-gate", type=int, default=PROFILE["min_labels_gate"])
    parser.add_argument("--al-max", type=int, default=PROFILE["al_max"])
    parser.add_argument("--min-stage2", type=int, default=PROFILE["min_stage2"])
    parser.add_argument("--refit-every", type=int, default=PROFILE["refit_every"])
    # Real Validate (scaled)
    parser.add_argument(
        "--stage3-k3",
        type=int,
        default=PROFILE["stage3_k3"],
        help="K3 env steps (paper 1e7)",
    )
    parser.add_argument(
        "--stage3-eval",
        type=int,
        default=PROFILE["stage3_eval"],
        help="Eval episodes (paper 500)",
    )
    parser.add_argument(
        "--stage3-rounds",
        type=int,
        default=PROFILE["stage3_rounds"],
        help="Refine rounds (paper 3)",
    )
    parser.add_argument(
        "--human-num",
        type=int,
        default=PROFILE["human_num"],
        help="Crowd size (paper 20)",
    )
    parser.add_argument(
        "--regime",
        default=PROFILE["regime"],
        choices=("with_random", "without_random"),
        help="Randomization regime (default: with_random = ON)",
    )
    parser.add_argument(
        "--skip-prereq-check",
        action="store_true",
        help="Start even if the dataset / GST / LLM pre-flight checks fail",
    )
    args, extra = parser.parse_known_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    if not args.skip_prereq_check:
        problems = (
            check_stage1_dataset(root)
            + check_gst_model(root, args.regime)
            + check_groq_keys(root, args.llm)
        )
        if not report_problems(problems, stream=sys.stderr):
            print(
                "\nFix the above (or pass --skip-prereq-check to override).",
                file=sys.stderr,
            )
            return 2

    if str(args.resume).strip() and str(args.warm_surrogate).strip():
        print("Use either --resume or --warm-surrogate, not both.", file=sys.stderr)
        return 2

    if str(args.resume).strip():
        try:
            out, surr_model, surr_data = _load_resume_paths(str(args.resume).strip())
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        resuming = True
        warm_n = None
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"results/raise_12h_{stamp}"
        surr_model = os.path.join(out, "surrogate_model")
        surr_data = os.path.join(out, "surrogate_dataset")
        resuming = False
        warm_n = None
        if str(args.warm_surrogate).strip():
            try:
                src_model, src_data = _resolve_warm_surrogate(
                    str(args.warm_surrogate).strip()
                )
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

    cmd = [
        sys.executable,
        "scripts/run_raise.py",
        "--closed-loop",
        "--llm",
        args.llm,
        "--device",
        args.device,
        "--num-processes",
        str(max(1, int(args.num_processes))),
        "--predict-method",
        str(PROFILE["predict_method"]),
        "--human-num",
        str(max(1, int(args.human_num))),
        "--regime",
        args.regime,
        "--stage1-population",
        str(int(args.population)),
        "--stage1-generations",
        str(int(args.generations)),
        "--closed-loop-k2",
        str(int(args.k2)),
        "--k2-unit",
        str(PROFILE["k2_unit"]),
        "--closed-loop-min-labels-gate",
        str(int(args.min_labels_gate)),
        "--closed-loop-al-max",
        str(int(args.al_max)),
        "--closed-loop-min-stage2",
        str(int(args.min_stage2)),
        "--closed-loop-refit-every",
        str(int(args.refit_every)),
        "--closed-loop-proxy-feedback",
        "--closed-loop-proxy-feedback-min-labels",
        str(int(PROFILE["proxy_feedback_min_labels"])),
        "--closed-loop-proxy-d3",
        str(int(PROFILE["proxy_d3_per_epoch"])),
        "--surrogate",
        surr_model,
        "--surrogate-dataset",
        surr_data,
        "--output-dir",
        out,
        "--seed",
        str(int(PROFILE["seed"])),
        "--stage3-rounds",
        str(int(args.stage3_rounds)),
        "--stage3-train-steps",
        str(int(args.stage3_k3)),
        "--stage3-eval-episodes",
        str(int(args.stage3_eval)),
        "--no-h-sweep",
        "--resume",
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.llm == "seed" or args.allow_seed_llm:
        cmd.append("--allow-seed-llm")
    cmd.extend(extra)

    resume_cmd = f"python scripts/run_raise_12h.py --resume {out}"
    print("=== RAISE closed-loop ~12h REAL (scaled) ===")
    print(f"mode:            {'RESUME' if resuming else 'NEW'}")
    print(f"output:          {out}")
    print(f"surrogate model: {surr_model}")
    print(f"surrogate data:  {surr_data}")
    if warm_n is not None:
        print(f"warm labels:     ≈{warm_n} (hard gate from epoch 1 if ≥ min_labels_gate)")
    print(f"checkpoint:      {os.path.join(out, 'closed_loop', 'checkpoint.json')}")
    _print_locked_config(args, warm_n=warm_n)
    print("IMPORTANT: prevent sleep; free disk >=30GB; GST checkpoint required.")
    print("If interrupted, resume with:")
    print(f"  {resume_cmd}")
    print(f"Watch: {out}/closed_loop/epochs.jsonl , RESUME.json , REPORT.txt")
    print()
    code = subprocess.call(cmd)
    report = os.path.join(out, "closed_loop", "REPORT.txt")
    if os.path.isfile(report):
        print()
        print("--- closed_loop/REPORT.txt ---")
        with open(report, encoding="utf-8") as fh:
            print(fh.read())
    if code != 0:
        print()
        print(f"Run exited with code {code}. Resume with:", file=sys.stderr)
        print(f"  {resume_cmd}", file=sys.stderr)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

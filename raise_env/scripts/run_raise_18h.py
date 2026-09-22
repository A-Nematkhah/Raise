#!/usr/bin/env python
"""RAISE loop ~18h REAL scaled overnight run (innovation path).

Real stack (not --easy / not Validate stub):
  - GST on:  predict_method=inferred
  - Randomization on: regime=with_random
  - human_num=5 (scaled crowd; paper uses 20)
  - Real Validate with scaled K3 / eval / rounds

Resume after interrupt:
  python scripts/run_closed_loop_18h.py --resume results/closed_loop_18h_YYYYMMDD_HHMMSS

From raise_env/:
  python scripts/run_closed_loop_18h.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime


def _load_resume_paths(resume_dir: str) -> tuple[str, str, str]:
    """Return (output_dir, surrogate_model, surrogate_dataset) for a prior run."""
    out = os.path.abspath(resume_dir)
    if not os.path.isdir(out):
        raise FileNotFoundError(f"Resume dir not found: {out}")
    ckpt = os.path.join(out, "closed_loop", "checkpoint.json")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt}. Did the run create closed_loop/checkpoint.json?"
        )
    with open(ckpt, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    # Prefer paths stored in checkpoint; fall back to isolated layout / sibling stamp.
    surr_model = str(
        payload.get("surrogate_model_dir")
        or os.path.join(out, "surrogate_model")
    )
    surr_data = str(
        payload.get("surrogate_dataset")
        or os.path.join(out, "surrogate_dataset")
    )
    # Legacy 18h layout: artifacts/ + data/ with same stamp suffix.
    if not os.path.isdir(surr_model):
        base = os.path.basename(out.rstrip("/\\"))
        stamp = base.replace("closed_loop_18h_", "", 1)
        alt = os.path.join("artifacts", f"surrogate_closed_loop_18h_{stamp}")
        if os.path.isdir(alt):
            surr_model = alt
    return out, surr_model, surr_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default="groq", choices=("groq", "seed", "ollama", "vllm"))
    parser.add_argument("--allow-seed-llm", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a previous results/closed_loop_18h_* directory to continue",
    )
    # RAISE loop search budget
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=7)
    parser.add_argument("--k2", type=int, default=3000, help="Refine short K2 (gradient steps)")
    parser.add_argument("--min-labels-gate", type=int, default=24)
    parser.add_argument("--al-max", type=int, default=2)
    parser.add_argument("--min-stage2", type=int, default=4)
    parser.add_argument("--refit-every", type=int, default=8)
    # Real Validate (scaled)
    parser.add_argument("--stage3-k3", type=int, default=500_000, help="K3 env steps (paper 1e7)")
    parser.add_argument("--stage3-eval", type=int, default=50, help="Eval episodes (paper 500)")
    parser.add_argument("--stage3-rounds", type=int, default=1, help="Refine rounds (paper 3)")
    parser.add_argument("--human-num", type=int, default=5, help="Crowd size (paper 20)")
    parser.add_argument(
        "--regime",
        default="with_random",
        choices=("with_random", "without_random"),
        help="Randomization regime (default: with_random = ON)",
    )
    args, extra = parser.parse_known_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    ds = os.path.join("data", "stage1_dataset", "stage1_dataset.npz")
    if not os.path.isfile(ds) and not os.path.isdir("data/stage1_dataset"):
        print("Missing data/stage1_dataset — collect Score1 dataset first.", file=sys.stderr)
        return 2
    if args.llm == "groq":
        keys = os.path.join(root, "groq_keys.json")
        if not os.path.isfile(keys) and not os.environ.get("GROQ_API_KEY"):
            print(
                "Groq: set GROQ_API_KEY or create raise_env/groq_keys.json",
                file=sys.stderr,
            )
            return 2

    if str(args.resume).strip():
        try:
            out, surr_model, surr_data = _load_resume_paths(str(args.resume).strip())
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        resuming = True
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"results/closed_loop_18h_{stamp}"
        # Nest under output_dir so printed paths match isolate_run_artifacts.
        surr_model = os.path.join(out, "surrogate_model")
        surr_data = os.path.join(out, "surrogate_dataset")
        resuming = False
    for path in (out, surr_model, surr_data):
        os.makedirs(path, exist_ok=True)

    # REAL scaled: GST + randomization + real Validate. No --easy / no --stage3-stub.
    cmd = [
        sys.executable,
        "scripts/run_raise.py",
        "--RAISE loop",
        "--llm",
        args.llm,
        "--device",
        args.device,
        "--num-processes",
        "1",
        "--predict-method",
        "inferred",
        "--human-num",
        str(max(1, int(args.human_num))),
        "--regime",
        args.regime,
        "--stage1-population",
        str(int(args.population)),
        "--stage1-generations",
        str(int(args.generations)),
        "--RAISE loop-k2",
        str(int(args.k2)),
        "--k2-unit",
        "gradient_steps",
        "--RAISE loop-min-labels-gate",
        str(int(args.min_labels_gate)),
        "--RAISE loop-al-max",
        str(int(args.al_max)),
        "--RAISE loop-min-stage2",
        str(int(args.min_stage2)),
        "--RAISE loop-refit-every",
        str(int(args.refit_every)),
        "--surrogate",
        surr_model,
        "--surrogate-dataset",
        surr_data,
        "--output-dir",
        out,
        "--seed",
        "425",
        # Real Validate (scaled-down paper stack)
        "--stage3-rounds",
        str(int(args.stage3_rounds)),
        "--stage3-train-steps",
        str(int(args.stage3_k3)),
        "--stage3-eval-episodes",
        str(int(args.stage3_eval)),
        "--no-h-sweep",  # with human_num=5 only H=5 anyway; saves wall clock
        "--final-rank",
        "llm",
        "--resume",
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.llm == "seed" or args.allow_seed_llm:
        cmd.append("--allow-seed-llm")
    cmd.extend(extra)

    print("=== RAISE loop ~18h REAL (scaled) ===")
    print(f"mode:            {'RESUME' if resuming else 'NEW'}")
    print(f"output:          {out}")
    print(f"surrogate model: {surr_model}")
    print(f"surrogate data:  {surr_data}")
    print(f"checkpoint:      {os.path.join(out, 'closed_loop', 'checkpoint.json')}")
    print(
        f"env: humans={args.human_num} GST=inferred regime={args.regime} nproc=1"
    )
    print(
        f"loop: N={args.population} G={args.generations} K2={args.k2} "
        f"gate>={args.min_labels_gate} al_max={args.al_max} min_s2={args.min_stage2}"
    )
    print(
        f"Validate REAL: K3={args.stage3_k3} eval={args.stage3_eval} "
        f"rounds={args.stage3_rounds} (no stub)"
    )
    print("IMPORTANT: prevent sleep; free disk >=30GB; GST checkpoint required.")
    print(f"If interrupted, resume with:")
    print(f"  python scripts/run_closed_loop_18h.py --resume {out}")
    print(f"Watch: {out}/closed_loop/epochs.jsonl , RESUME.json , REPORT.txt")
    print()
    code = subprocess.call(cmd)
    report = os.path.join(out, "closed_loop", "REPORT.txt")
    if os.path.isfile(report):
        print()
        print("--- closed_loop/REPORT.txt ---")
        with open(report, encoding="utf-8") as fh:
            print(fh.read())
    return code


if __name__ == "__main__":
    raise SystemExit(main())

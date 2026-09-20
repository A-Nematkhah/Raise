#!/usr/bin/env python
"""Closed-loop ~18h REAL scaled overnight run (innovation path).

Real stack (not --easy / not Stage3 stub):
  - GST on:  predict_method=inferred
  - Randomization on: regime=with_random
  - human_num=5 (scaled crowd; paper uses 20)
  - Real Stage III with scaled K3 / eval / rounds

Paper vs this scale (approx):
  humans 20 -> 5
  N=8 G=10 K2=4000 (loop)  ~ kept / slightly trimmed
  Stage III: paper K3=1e7, E=500, R=3, H-sweep
             here   K3=2e5, E=50,  R=1, H={5} only

From evonav_env/:
  python scripts/run_closed_loop_18h.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default="groq", choices=("groq", "seed", "ollama", "vllm"))
    parser.add_argument("--allow-seed-llm", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    # Closed-loop search budget
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=7)
    parser.add_argument("--k2", type=int, default=3000, help="Stage II short K2 (gradient steps)")
    parser.add_argument("--min-labels-gate", type=int, default=24)
    parser.add_argument("--al-max", type=int, default=2)
    parser.add_argument("--min-stage2", type=int, default=4)
    parser.add_argument("--refit-every", type=int, default=8)
    # Real Stage III (scaled)
    parser.add_argument("--stage3-k3", type=int, default=200_000, help="K3 env steps (paper 1e7)")
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
                "Groq: set GROQ_API_KEY or create evonav_env/groq_keys.json",
                file=sys.stderr,
            )
            return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/closed_loop_18h_{stamp}"
    surr_model = f"artifacts/surrogate_closed_loop_18h_{stamp}"
    surr_data = f"data/surrogate_dataset_closed_loop_18h_{stamp}"
    for path in (out, surr_model, surr_data):
        os.makedirs(path, exist_ok=True)

    # REAL scaled: GST + randomization + real Stage III. No --easy / no --stage3-stub.
    cmd = [
        sys.executable,
        "scripts/run_evonav.py",
        "--closed-loop",
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
        "--closed-loop-k2",
        str(int(args.k2)),
        "--k2-unit",
        "gradient_steps",
        "--closed-loop-min-labels-gate",
        str(int(args.min_labels_gate)),
        "--closed-loop-al-max",
        str(int(args.al_max)),
        "--closed-loop-min-stage2",
        str(int(args.min_stage2)),
        "--closed-loop-refit-every",
        str(int(args.refit_every)),
        "--surrogate",
        surr_model,
        "--surrogate-dataset",
        surr_data,
        "--output-dir",
        out,
        "--seed",
        "425",
        # Real Stage III (scaled-down paper stack)
        "--stage3-rounds",
        str(int(args.stage3_rounds)),
        "--stage3-train-steps",
        str(int(args.stage3_k3)),
        "--stage3-eval-episodes",
        str(int(args.stage3_eval)),
        "--no-h-sweep",  # with human_num=5 only H=5 anyway; saves wall clock
        "--final-rank",
        "llm",
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.llm == "seed" or args.allow_seed_llm:
        cmd.append("--allow-seed-llm")
    cmd.extend(extra)

    print("=== Closed-loop ~18h REAL (scaled) ===")
    print(f"output:          {out}")
    print(f"surrogate model: {surr_model}")
    print(f"surrogate data:  {surr_data}")
    print(
        f"env: humans={args.human_num} GST=inferred regime={a
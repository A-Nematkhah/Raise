#!/usr/bin/env python
"""RAISE loop ~1h smoke: Gen0 labels -> Surrogate fit -> Gen>=1 gate+AL -> refit.

Exercises the innovation path end-to-end (not paper Alg.1 linear):
  Score1 -> Surrogate gate -> in-loop AL -> Refine short -> append -> refit
  (+ Validate stub so the pipeline finishes)

From evonav_env/:
  python scripts/run_closed_loop_1h.py
  python scripts/run_closed_loop_1h.py --llm seed --allow-seed-llm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        default="groq",
        choices=("groq", "seed", "ollama", "vllm"),
        help="LLM provider (default: groq)",
    )
    parser.add_argument(
        "--allow-seed-llm",
        action="store_true",
        help="Required when --llm seed on a non-fast run",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass through --verbose (crowd_nav DEBUG; HTTP dumps stay quiet)",
    )
    args, extra = parser.parse_known_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/closed_loop_1h_{stamp}"
    surr_model = os.path.join(out, "surrogate_model")
    surr_data = os.path.join(out, "surrogate_dataset")
    for path in (out, surr_model, surr_data):
        os.makedirs(path, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_evonav.py",
        "--RAISE loop",
        "--llm",
        args.llm,
        "--device",
        args.device,
        "--num-processes",
        "1",
        "--easy",
        "--predict-method",
        "none",
        "--human-num",
        "5",
        "--stage1-population",
        "4",
        "--stage1-generations",
        "3",
        "--RAISE loop-k2",
        "2000",
        "--k2-unit",
        "gradient_steps",
        # Soft→hard gate after Gen0 (4 labels); AL + gate run on Gen≥1.
        "--RAISE loop-min-labels-gate",
        "4",
        "--RAISE loop-al-max",
        "2",
        # Allow gate to drop (min 2 Refine / epoch, not force all 4).
        "--RAISE loop-min-stage2",
        "2",
        # Refit after Gen1+ as soon as ≥2 new labels land.
        "--RAISE loop-refit-every",
        "2",
        "--surrogate",
        surr_model,
        "--surrogate-dataset",
        surr_data,
        "--output-dir",
        out,
        "--seed",
        "425",
        "--regime",
        "without_random",
        "--stage3-stub",
        "--no-h-sweep",
        "--stage3-rounds",
        "1",
        "--final-rank",
        "llm",
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.llm == "seed" or args.allow_seed_llm:
        cmd.append("--allow-seed-llm")
    cmd.extend(extra)

    print("=== RAISE loop 1h smoke ===")
    print(f"output:          {out}")
    print(f"surrogate model: {surr_model}")
    print(f"surrogate data:  {surr_data}")
    print(f"llm:             {args.llm}")
    print(
        "covers: Gen0 Score1+Refine+fit | Gen>=1 gate+AL+Refine+refit | Validate stub"
    )
    print("logs: INFO only (pass --verbose for crowd_nav DEBUG; HTTP dumps stay quiet)")
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

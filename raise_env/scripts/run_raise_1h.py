#!/usr/bin/env python
"""RAISE loop ~1h smoke: Gen0 labels -> Surrogate fit -> Gen>=1 gate+AL -> refit.

Exercises the innovation path end-to-end (not paper Alg.1 linear):
  Score1 -> Surrogate gate -> in-loop AL -> Refine short -> append -> refit
  (+ Validate stub so the pipeline finishes)

From raise_env/:
  python scripts/run_raise_1h.py
  python scripts/run_raise_1h.py --llm seed --allow-seed-llm

Thin wrapper — budget defaults live in ``raise_core.presets.CLOSED_LOOP_PROFILES['1h']``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"results/raise_1h_{stamp}"
    surr_model = os.path.join(out, "surrogate_model")
    surr_data = os.path.join(out, "surrogate_dataset")
    for path in (out, surr_model, surr_data):
        os.makedirs(path, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/run_raise.py",
        "--profile",
        "1h",
        "--surrogate",
        surr_model,
        "--surrogate-dataset",
        surr_data,
        "--output-dir",
        out,
        *sys.argv[1:],
    ]
    print("=== RAISE loop 1h smoke (profile=1h) ===")
    print(f"output: {out}")
    print("cmd:", " ".join(cmd))
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

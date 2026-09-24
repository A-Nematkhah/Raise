#!/usr/bin/env python
"""RAISE classic Algorithm 1 on highway-fast-v0 — ~4h wall budget.

Locked PROFILE (SB3 PPO, env_steps — not CrowdNav GST/SRNN):

  Stage I:  N=6, G=6, Score1 on highway Stage I dataset
  Stage II: R=3, K2=30_000 env steps, E2=20
  Stage III: R=1, K3=100_000 env steps, E3=40, no H-sweep

Rough wall time on one GPU (highway ≈50–80 env steps/s + seed/groq LLM):
  Stage I  ~10–40 min (LLM-bound)
  Stage II ~1.5–2.5 h
  Stage III ~1.5–2.0 h
  Total    ~3.5–5 h (aim ~4 h)

From raise_env/:

  # once
  python scripts/collect_highway_stage1_dataset.py --out domains/highway/data/stage1_dataset

  # 4h run (seed LLM, no API key)
  python scripts/run_raise_highway_4h.py

  # or Groq
  python scripts/run_raise_highway_4h.py --llm groq
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401

import argparse
import os
import subprocess
import sys
from datetime import datetime

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
os.chdir(_ROOT)
for _path in (_ROOT, _SCRIPTS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

PROFILE = {
    "domain": "highway",
    "seed": 425,
    "llm": "seed",
    "device": "cuda",
    "stage1_dataset": "domains/highway/data/stage1_dataset",
    "score1": "dataset",
    # Stage I
    "population": 6,
    "generations": 6,
    # Stage II (highway SB3 reads train steps as env timesteps)
    "stage2_rounds": 3,
    "stage2_train_steps": 30_000,
    "k2_unit": "env_steps",
    "stage2_eval": 20,
    # Stage III
    "stage3_rounds": 1,
    "stage3_train_steps": 100_000,
    "stage3_eval": 40,
}


def _dataset_ready(path: str) -> bool:
    traj = os.path.join(path, "trajectories.jsonl")
    return os.path.isfile(traj) and os.path.getsize(traj) > 0


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


def _print_profile(args) -> None:
    print("=== highway ~4h PROFILE ===")
    for key in (
        "domain",
        "seed",
        "population",
        "generations",
        "stage2_rounds",
        "stage2_train_steps",
        "k2_unit",
        "stage2_eval",
        "stage3_rounds",
        "stage3_train_steps",
        "stage3_eval",
    ):
        print(f"  {key}: {PROFILE[key]}")
    print(f"  llm: {args.llm}")
    print(f"  device: {args.device}")
    print(f"  dataset: {PROFILE['stage1_dataset']}")
    print("  estimate: ~3.5–5 h wall (GPU + seed/groq LLM)")
    print("===========================")


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
        "--skip-collect",
        action="store_true",
        help="Do not auto-collect Stage I dataset if missing",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.skip_collect:
        code = _ensure_dataset(PROFILE["stage1_dataset"])
        if code != 0:
            print("Dataset collection failed.", file=sys.stderr)
            return code
    elif not _dataset_ready(PROFILE["stage1_dataset"]):
        print(
            f"Missing {PROFILE['stage1_dataset']}/trajectories.jsonl. "
            "Run collect_highway_stage1_dataset.py or drop --skip-collect.",
            file=sys.stderr,
        )
        return 2

    out = str(args.output_dir).strip()
    if not out:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"results/highway_4h_{stamp}"

    _print_profile(args)

    cmd = [
        sys.executable,
        os.path.join(_SCRIPTS, "run_raise.py"),
        "--domain",
        PROFILE["domain"],
        "--llm",
        str(args.llm),
        "--device",
        str(args.device),
        "--stage1-dataset",
        PROFILE["stage1_dataset"],
        "--score1",
        PROFILE["score1"],
        "--stage1-population",
        str(PROFILE["population"]),
        "--stage1-generations",
        str(PROFILE["generations"]),
        "--stage2-rounds",
        str(PROFILE["stage2_rounds"]),
        "--stage2-train-steps",
        str(PROFILE["stage2_train_steps"]),
        "--k2-unit",
        PROFILE["k2_unit"],
        "--stage2-eval-episodes",
        str(PROFILE["stage2_eval"]),
        "--stage3-rounds",
        str(PROFILE["stage3_rounds"]),
        "--stage3-train-steps",
        str(PROFILE["stage3_train_steps"]),
        "--stage3-eval-episodes",
        str(PROFILE["stage3_eval"]),
        "--no-h-sweep",
        "--seed",
        str(PROFILE["seed"]),
        "--output-dir",
        out,
        "--final-rank",
        "scalar",
        "--allow-seed-llm",
    ]
    if args.llm_model:
        cmd.extend(["--llm-model", str(args.llm_model)])
    if args.verbose:
        cmd.append("--verbose")

    print("cmd:", " ".join(cmd))
    print(f"output: {out}")
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())

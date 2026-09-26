#!/usr/bin/env python
"""RAISE loop ~18h REAL scaled overnight run (innovation path).

Real stack (not --easy / not Validate stub):
  - GST on:  predict_method=inferred
  - Randomization on: regime=with_random
  - human_num=5 (scaled crowd; paper uses 20)
  - Real Validate with scaled K3 / eval / rounds

Resume after interrupt:
  python scripts/run_raise_18h.py --resume results/closed_loop_18h_YYYYMMDD_HHMMSS

From raise_env/:
  python scripts/run_raise_18h.py
"""

from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401 — arms domains/crowdnav/runtime on sys.path
import argparse
import json
import os
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
from raise_core.presets import CLOSED_LOOP_PROFILES  # noqa: E402

_P18 = dict(CLOSED_LOOP_PROFILES["18h"])


def _load_resume_paths(resume_dir: str) -> tuple[str, str, str]:
    """Return (output_dir, surrogate_model, surrogate_dataset) for a prior run."""
    out = os.path.abspath(resume_dir)
    if not os.path.isdir(out):
        raise FileNotFoundError(f"Resume dir not found: {out}")
    # closed_loop/ is the current layout; run root is the mirrored copy.
    tried = [
        os.path.join(out, "closed_loop", "checkpoint.json"),
        os.path.join(out, "checkpoint.json"),
    ]
    ckpt = next((p for p in tried if os.path.isfile(p)), "")
    if not ckpt:
        raise FileNotFoundError(
            "No closed-loop checkpoint found. Looked in:\n  "
            + "\n  ".join(tried)
        )
    with open(ckpt, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    extra = payload.get("extra") or {}
    # Prefer paths stored in checkpoint; fall back to isolated layout / sibling stamp.
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
    # RAISE loop search budget (defaults from raise_core.presets CLOSED_LOOP_PROFILES['18h'])
    parser.add_argument("--population", type=int, default=int(_P18["population"]))
    parser.add_argument("--generations", type=int, default=int(_P18["generations"]))
    parser.add_argument(
        "--k2", type=int, default=int(_P18["k2"]), help="Refine short K2 (gradient steps)"
    )
    parser.add_argument("--min-labels-gate", type=int, default=int(_P18["min_labels_gate"]))
    parser.add_argument("--al-max", type=int, default=int(_P18["al_max"]))
    parser.add_argument("--min-stage2", type=int, default=int(_P18["min_stage2"]))
    parser.add_argument("--refit-every", type=int, default=int(_P18["refit_every"]))
    # Real Validate (scaled)
    parser.add_argument(
        "--stage3-k3",
        type=int,
        default=int(_P18["stage3_k3"]),
        help="K3 env steps (paper 1e7)",
    )
    parser.add_argument(
        "--stage3-eval",
        type=int,
        default=int(_P18["stage3_eval"]),
        help="Eval episodes (paper 500)",
    )
    parser.add_argument(
        "--stage3-rounds",
        type=int,
        default=int(_P18["stage3_rounds"]),
        help="Refine rounds (paper 3)",
    )
    parser.add_argument(
        "--human-num",
        type=int,
        default=int(_P18["human_num"]),
        help="Crowd size (paper 20)",
    )
    parser.add_argument(
        "--regime",
        default=str(_P18["regime"]),
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

    problems = (
        check_stage1_dataset(root)
        + check_gst_model(root, args.regime)
        + check_groq_keys(root, args.llm)
    )
    if not report_problems(problems, stream=sys.stderr) and not args.skip_prereq_check:
        print(
            "\nFix the above (or pass --skip-prereq-check to override).",
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
    print("If interrupted, resume with:")
    print(f"  python scripts/run_raise_18h.py --resume {out}")
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

#!/usr/bin/env python
"""RAISE closed-loop ~12h REAL scaled run (innovation path).

Real stack (no --easy, no Validate stub):
  - GST on:            predict_method=inferred
  - Randomization on:  regime=with_random (GST *_rand checkpoint)
  - human_num=5 (scaled crowd; paper uses 20), num_processes=1
  - Real Validate with scaled K3 / eval / rounds

Budget (~12h on one GPU): N=8, G=5, K2=3000 gradient steps, K3=350k env steps.

From raise_env/:
  python scripts/run_raise_12h.py

Resume after an interrupt (Ctrl-C, crash, reboot):
  python scripts/run_raise_12h.py --resume results/raise_12h_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

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

CHECKPOINT_LOCATIONS = (
    os.path.join("closed_loop", "checkpoint.json"),  # current layout
    "checkpoint.json",  # mirrored copy at the run root
)


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
        help="Path to a previous results/raise_12h_* directory to continue",
    )
    # Closed-loop search budget (~12h)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--k2", type=int, default=3000, help="In-loop K2 (gradient steps)")
    parser.add_argument("--min-labels-gate", type=int, default=24)
    parser.add_argument("--al-max", type=int, default=2)
    parser.add_argument("--min-stage2", type=int, default=4)
    parser.add_argument("--refit-every", type=int, default=8)
    # Real Validate (scaled)
    parser.add_argument("--stage3-k3", type=int, default=350_000, help="K3 env steps (paper 1e7)")
    parser.add_argument("--stage3-eval", type=int, default=50, help="Eval episodes (paper 500)")
    parser.add_argument("--stage3-rounds", type=int, default=1, help="Refine rounds (paper 3)")
    parser.add_argument("--human-num", type=int, default=5, help="Crowd size (paper 20)")
    parser.add_argument(
        "--regime",
        default="with_random",
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
        out = f"results/raise_12h_{stamp}"
        # Surrogate artifacts live under output_dir (isolate_run_artifacts).
        surr_model = os.path.join(out, "surrogate_model")
        surr_data = os.path.join(out, "surrogate_dataset")
        resuming = False
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
        "--no-h-sweep",  # humans=5 → only H=5 anyway; saves wall clock
        "--final-rank",
        "llm",
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
    print(f"checkpoint:      {os.path.join(out, 'closed_loop', 'checkpoint.json')}")
    print(f"env: humans={args.human_num} GST=inferred regime={args.regime} nproc=1")
    print(
        f"loop: N={args.population} G={args.generations} K2={args.k2} gradient_steps "
        f"gate>={args.min_labels_gate} al_max={args.al_max} "
        f"min_s2={args.min_stage2} refit_every={args.refit_every}"
    )
    print(
        f"Validate REAL: K3={args.stage3_k3} eval={args.stage3_eval} "
        f"rounds={args.stage3_rounds} (no stub, no H-sweep)"
    )
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
    return code


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Human-triggered paper-scale EvoNav run (Tables 3–6 budgets).

  K2=8000, G2=16, K3=1e7, G3=3, E3=500, N=8, G1=10, M=100, N_traj=10

Multi-seed (default 5): full Stage I→II→III once per seed; report mean±std
**across seeds**. Resume per (seed, stage, round, candidate). Cost log →
``results/paper_scale/cost_log.json``.

This script is **never** invoked by pytest / CI. Refuse to start if ``CI`` is set.

Examples::

    python scripts/run_evonav_paper_scale.py
    python scripts/run_evonav_paper_scale.py --seeds 425,426,427 --device cuda
    python scripts/run_evonav_paper_scale.py --dry-run-stubs  # wiring only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)


def _parse_seeds(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def main() -> int:
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        print(
            "Refusing paper-scale run under CI. "
            "This entry point is human-triggered only "
            "(K3=1e7 × N × G3 × seeds is far too large for CI).",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(
        description="EvoNav paper-scale multi-seed Algorithm 1 (Tables 3–6)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper_scale.yaml",
        help="Paper-scale YAML (default: configs/paper_scale.yaml)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (default: 5 seeds from YAML / 425–429)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output dir (default: results/paper_scale)",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        choices=["seed", "groq", "vllm", "ollama", "scripted"],
    )
    parser.add_argument(
        "--allow-seed-llm",
        action="store_true",
        help="Permit llm_provider=seed (YAML default or --llm seed) without a real LLM",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Extra DEBUG-level terminal detail",
    )
    parser.add_argument(
        "--dry-run-stubs",
        action="store_true",
        help="Use --fast stub budgets to exercise resume/report wiring only",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        choices=["without_random", "with_random", "both"],
        help="Override EVOLUTION_RANDOMIZATION_REGIME from YAML (AUDIT.md §8.1)",
    )
    args = parser.parse_args()
    if args.regime == "both":
        print(
            "regime=both requires two separate paper-scale passes (doubled budget).",
            file=sys.stderr,
        )
        return 2

    # Isolate Config.get_args() from our CLI.
    sys.argv = [
        sys.argv[0],
        "--no-cuda" if (args.device or "cuda") == "cpu" else "--seed",
        "425",
    ]

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    from crowd_nav.reward_search import console as _console

    _console.set_verbose(bool(args.verbose))

    from crowd_nav.reward_search.paper_scale import PaperScaleRunner
    from crowd_nav.reward_search.presets import load_paper_scale_yaml
    from dataclasses import replace

    spec = load_paper_scale_yaml(args.config)
    if args.regime is not None:
        spec = replace(spec, randomization_regime=args.regime)
    resolved_llm = (args.llm or spec.llm_provider or "seed").strip().lower()
    if resolved_llm == "seed" and not args.allow_seed_llm and not args.dry_run_stubs:
        print(
            "Refusing to run a non-fast pipeline with the seed (no real LLM) "
            "provider — pass --llm groq|ollama|vllm explicitly, or pass "
            "--allow-seed-llm if this is intentional (e.g. debugging Stage II/III "
            "wiring without LLM cost).",
            file=sys.stderr,
        )
        return 2
    seeds = _parse_seeds(args.seeds)
    runner = PaperScaleRunner(
        spec,
        seeds=seeds,
        output_dir=args.output_dir,
        device=args.device,
        llm_provider=args.llm,
        dry_run_stubs=args.dry_run_stubs,
    )
    logging.info(
        "Paper-scale start: seeds=%s K3=%d regime=%s predict_method=%s "
        "output=%s dry_run_stubs=%s",
        runner.seeds,
        spec.K3,
        spec.randomization_regime,
        spec.predict_method,
        runner.output_dir,
        args.dry_run_stubs,
    )
    logging.info("Methodology: %s", spec.methodology_note())
    report = runner.run()
    logging.info(
        "Done. Aggregate keys=%s cost_log=%s",
        list((report.get("table1_style_aggregate") or {}).get("metrics") or {}),
        runner.cost_log_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

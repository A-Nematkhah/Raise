#!/usr/bin/env python
"""
Single entry point for RAISE on CrowdNav++ CrowdSimVarNum.

Edit ``raise_config.py`` or pass ``--config`` / ``--set section.key=value``.
See README_RAISE.md Configuration for migrated CLI flags.

Examples (from ``raise_env/``)::

    python scripts/run_raise.py --dry-run --output-dir results/raise_fast
    python scripts/run_raise.py --set llm.provider=groq --output-dir results/crowdnav_run
    python scripts/run_raise.py --print-config
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402

from raise_config import config_to_dict, resolve_config  # noqa: E402
from raise_core import console  # noqa: E402
from raise_core.config_factory import ensure_config_snapshot, make_run_config  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.pipeline import RaisePipeline  # noqa: E402

os.chdir(_ROOT)

_MIGRATION_HINT = (
    "Removed run_raise flags map to --set keys; see README_RAISE.md Configuration."
)


class RaiseArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        if "unrecognized arguments" in message:
            self.exit(
                2,
                f"{message}\n\n{_MIGRATION_HINT}\n",
            )
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = RaiseArgumentParser(description="RAISE end-to-end (CrowdNav)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Python file exporting CONFIG: RaiseConfig (overrides raise_config.py)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override raise_config (repeatable); e.g. budget.population_size=8",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print effective raise_config JSON and exit",
    )
    parser.add_argument(
        "--output-dir",
        "--run-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Run directory (default: results/crowdnav_<timestamp>)",
    )
    parser.add_argument(
        "--dry-run",
        "--fast",
        dest="dry_run",
        action="store_true",
        help="Stub trainers + smoke Score1 (seconds-scale wiring check)",
    )
    parser.add_argument(
        "--allow-seed-llm",
        action="store_true",
        help="Permit llm.provider=seed on a non-dry run (debug without API cost)",
    )
    parser.add_argument(
        "--allow-config-drift",
        action="store_true",
        help="Allow resume when raise_config differs from config_snapshot.json",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from checkpoints when present (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoints and start fresh",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="After the run, invoke scripts/run_baselines.py on this output dir",
    )
    return parser


def main() -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args()

    try:
        raise_cfg = resolve_config(
            config_path=args.config,
            set_overrides=tuple(args.set or ()),
        )
    except (ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.print_config:
        print(json.dumps(config_to_dict(raise_cfg), indent=2, sort_keys=True))
        return 0

    output_dir = args.output_dir or (
        "results/crowdnav_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    run_cfg = make_run_config(
        raise_cfg,
        output_dir=output_dir,
        resume=bool(args.resume),
        allow_config_drift=bool(args.allow_config_drift),
        use_stub=bool(args.dry_run),
        fast=bool(args.dry_run),
    )

    if (
        not args.dry_run
        and str(run_cfg.llm_provider).strip().lower() == "seed"
        and not args.allow_seed_llm
    ):
        print(
            "Refusing a non-dry run with llm.provider=seed — set "
            "--set llm.provider=groq|ollama|vllm or pass --allow-seed-llm.",
            file=sys.stderr,
        )
        return 2

    try:
        operators = run_cfg.operators()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        ensure_config_snapshot(
            output_dir,
            raise_cfg,
            resume=bool(args.resume),
            allow_config_drift=bool(args.allow_config_drift),
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    console.configure_run_logging(verbose=bool(args.verbose), output_dir=run_cfg.output_dir)
    logging.info(
        "RAISE -> %s (dry_run=%s, N=%d, G=%d, K2=%d, K3=%d, operators=%s)",
        run_cfg.output_dir,
        run_cfg.fast,
        run_cfg.population,
        run_cfg.generations,
        run_cfg.stage2_train_steps,
        run_cfg.stage3_train_steps,
        operators.to_dict(),
    )
    artifacts = RaisePipeline(run_cfg).run()
    logging.info("Done. Final candidate: %s", artifacts.best_stage3.candidate_id)
    logging.info("Artifacts: %s", artifacts.output_dir)
    report = os.path.join(run_cfg.output_dir, "closed_loop", "REPORT.txt")
    if os.path.isfile(report):
        with open(report, encoding="utf-8") as fh:
            print("\n--- closed_loop/REPORT.txt ---\n" + fh.read())
    if bool(args.baselines):
        import importlib.util

        bas_path = os.path.join(_ROOT, "scripts", "run_baselines.py")
        spec = importlib.util.spec_from_file_location("run_baselines", bas_path)
        if spec is None or spec.loader is None:
            logging.error("Could not load %s", bas_path)
            return 1
        bas = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bas)
        logging.info("Running baselines on %s", run_cfg.output_dir)
        bas.run_baselines(run_cfg.output_dir)
        logging.info("Baselines written to %s", os.path.join(run_cfg.output_dir, "baselines.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""P1-1 baselines for a finished RAISE CrowdNav run.

Examples (from ``raise_env/``)::

    python scripts/run_baselines.py --run-dir <run-dir>
    python scripts/run_baselines.py --run-dir <run-dir> --seeds 3 --extra-ids ini_0000
"""

from __future__ import annotations

import os as _os
import sys as _sys

_RAISE_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _RAISE_ROOT not in _sys.path:
    _sys.path.insert(0, _RAISE_ROOT)

from raise_core.script_utils import bootstrap_raise_env

bootstrap_raise_env(chdir=True)

import logging  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.post_run_baselines import (  # noqa: E402
    DEFAULT_EVAL_EPISODES,
    build_parser,
    run_baselines,
)

__all__ = ["DEFAULT_EVAL_EPISODES", "build_parser", "main", "run_baselines"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    extra = [x.strip() for x in str(args.extra_ids or "").split(",") if x.strip()]
    run_baselines(
        args.run_dir,
        seeds=max(1, int(args.seeds)),
        extra_ids=extra,
        eval_episodes=args.eval_episodes,
        train_steps=args.train_steps,
        collapse_cr=args.collapse_cr,
        skip_a=bool(args.skip_a),
        skip_b=bool(args.skip_b),
        skip_c=bool(args.skip_c),
        ppo_gamma=args.ppo_gamma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

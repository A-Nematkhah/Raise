#!/usr/bin/env python
"""Print a readable closed-loop REPORT for an existing run directory.

Usage (from raise_env/):
  python scripts/print_raise_report.py results/raise_12h_YYYYMMDD_HHMMSS

Highway reports include Pareto front0 ids when epochs recorded ``pareto``
(see raise_env/docs/SELECTION.md for deliberate final pick).
"""

from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401 — arms domains/crowdnav/runtime on sys.path
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Run output dir containing closed_loop/epochs.jsonl",
    )
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from raise_core.raise_loop.report import (
        build_closed_loop_report,
        write_closed_loop_report,
    )

    out = args.output_dir
    if not os.path.isdir(out):
        print(f"not a directory: {out}", file=sys.stderr)
        return 2
    path = write_closed_loop_report(out)
    print(build_closed_loop_report(out), end="")
    print(f"(wrote {path})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

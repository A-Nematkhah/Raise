#!/usr/bin/env python
"""Print a readable closed-loop REPORT for an existing run directory.

Usage (from raise_env/):
  python scripts/print_raise_report.py results/raise_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import os as _os
import sys as _sys

_RAISE_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _RAISE_ROOT not in _sys.path:
    _sys.path.insert(0, _RAISE_ROOT)

from raise_core.script_utils import bootstrap_raise_env

bootstrap_raise_env()
import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.raise_loop.report import (  # noqa: E402
    build_closed_loop_report,
    write_closed_loop_report,
)


def main() -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        help="Run output dir containing closed_loop/epochs.jsonl",
    )
    args = parser.parse_args()

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

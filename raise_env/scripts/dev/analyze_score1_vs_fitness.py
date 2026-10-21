#!/usr/bin/env python3
"""Print Spearman(Score1, Stage II fitness) for a RAISE run directory (P1-6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from raise_core.score1_fitness import analyze_run, format_report_line


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="RAISE output directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary dict as JSON",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2
    summary = analyze_run(run_dir)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_report_line(summary))
    return 0


if __name__ == "__main__":
    import os as _os
    import sys as _sys

    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from raise_core.script_utils import bootstrap_raise_env

    bootstrap_raise_env()
    raise SystemExit(main())

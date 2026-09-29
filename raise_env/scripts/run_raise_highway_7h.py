#!/usr/bin/env python
"""RAISE full closed-loop on highway-fast-v0 — ~6–7h wall budget.

Same K2/K3 as highway_4h (12k / 70k env steps); deeper search: N=8, G=7.

From raise_env/:

  python scripts/run_raise_highway_7h.py --llm groq --skip-collect

  # resume
  python scripts/run_raise_highway_7h.py --llm groq --resume results/highway_7h_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Inject before importing the shared wrapper so PROFILE peeks correctly.
sys.argv[1:1] = ["--profile", "highway_7h"]

from run_raise_highway_4h import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

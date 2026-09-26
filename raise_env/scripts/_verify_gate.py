#!/usr/bin/env python
"""Print new vs old speed-gate values for verification report."""

from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from domains.highway.metrics import V_MIN, V_TARGET, _speed_gate  # noqa: E402
from domains.highway.objective_constants import K_GATE  # noqa: E402


def _sigmoid(x: float) -> float:
    z = float(x)
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def old_gate(v: float) -> float:
    """Pre-fix: sigmoid(8 * (v - V_MIN) / V_TARGET)."""
    return _sigmoid(8.0 * (float(v) - 10.0) / 25.0)


def main() -> None:
    speeds = [0, 5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35]
    print(f"V_MIN={V_MIN} V_TARGET={V_TARGET} K_GATE(new)={K_GATE}")
    print(f"{'v':>6}  {'gate_NEW':>10}  {'gate_OLD':>10}  {'delta_new-old':>14}")
    print("-" * 48)
    for v in speeds:
        g_new = _speed_gate(float(v))
        g_old = old_gate(float(v))
        print(f"{v:6.1f}  {g_new:10.6f}  {g_old:10.6f}  {g_new - g_old:14.6f}")
    g20 = _speed_gate(20.0)
    g25 = _speed_gate(25.0)
    print()
    print(f"gate(20)           = {g20:.6f}")
    print(f"gate(25)           = {g25:.6f}")
    print(f"gate(25)-gate(20)  = {g25 - g20:.6f}")
    print(f"gate(V_MIN={V_MIN}) = {_speed_gate(V_MIN):.6f}")
    print(f"gate(V_TARGET={V_TARGET}) = {_speed_gate(V_TARGET):.6f}")
    print(f"old gate(20)       = {old_gate(20.0):.6f}")
    print(f"old gate(25)       = {old_gate(25.0):.6f}")
    print(f"old (25)-(20)      = {old_gate(25.0) - old_gate(20.0):.6f}")


if __name__ == "__main__":
    main()

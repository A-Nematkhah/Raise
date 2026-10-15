#!/usr/bin/env python
"""Sanity-check the CrowdNav environment against the published ORCA numbers.

Runs the upstream ORCA policy as the robot (safety_space 0, invisible robot)
on every case of the test split in order, like upstream ``test.py``, and
compares success / collision / timeout rate, mean navigation time of
successful episodes and the Explorer discounted reward against Table I of
Chen et al., "Crowd-Robot Interaction" (ICRA 2019).

Two discounted rewards are reported:

* ``reward_full``: every native step reward, discomfort penalties included
  (what RAISE uses as fitness).
* ``reward_terminal``: success / collision reward only. The paper's ORCA row
  matches this one; ``reward_full`` is lower because ORCA spends roughly a
  third of its steps inside ``discomfort_dist``.

Exit code 1 when a checked metric is outside its tolerance.

Example (from ``raise_env/``)::

    python scripts/check_orca_reference.py --out results/orca_reference_check.json
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv  # noqa: E402
from domains.crowdnav.metrics import discounted_task_return  # noqa: E402
from domains.crowdnav.objective_constants import CROWDNAV_EVAL_GAMMA  # noqa: E402
from domains.crowdnav.settings import CrowdNavSettings, set_settings  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.json_io import atomic_write_json  # noqa: E402

# Chen et al. 2019, Table I, ORCA row (invisible robot, 5 humans, circle crossing).
PAPER_ORCA = {"SR": 0.43, "CR": 0.57, "nav_time": 10.86, "reward_terminal": 0.054}
# SR/CR: ~1.5 binomial SE at n=500; time in seconds; reward absolute.
TOLERANCE = {"SR": 0.03, "CR": 0.03, "nav_time": 0.5, "reward_terminal": 0.01}


def _load_collect_module() -> Any:
    path = os.path.join(_ROOT, "scripts", "collect_crowdnav_stage1_dataset.py")
    spec = importlib.util.spec_from_file_location("collect_crowdnav_stage1_dataset", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_orca_check(
    *,
    env_config_path: Optional[str] = None,
    phase: str = "test",
    n_cases: Optional[int] = None,
    behaviour: str = "orca",
) -> Dict[str, Any]:
    """ORCA robot on cases ``0..n_cases-1`` of ``phase``; paper-style metrics."""
    collect = _load_collect_module()
    settings = set_settings(CrowdNavSettings(env_config_path=env_config_path))
    env = RewardInjectedCrowdNavEnv(None, phase=phase, settings=settings)
    try:
        size = int(env.sim.case_size[phase])
        n = size if n_cases is None else max(1, min(int(n_cases), size))
        dt = float(env.time_step)
        v_pref = float(env.sim.robot.v_pref)
        discomfort = float(env.sim.discomfort_dist)
        outcomes: List[str] = []
        times: List[float] = []
        full: List[float] = []
        terminal: List[float] = []
        undiscounted: List[float] = []
        discomfort_steps = total_steps = 0
        for case in range(n):
            env.reset(options={"case": case})
            act = collect._make_controller(behaviour, env, np.random.default_rng(0))
            native: List[float] = []
            done = False
            info: Dict[str, Any] = {}
            while not done:
                _obs, reward, terminated, truncated, info = env.step_upstream(act())
                native.append(float(reward))
                total_steps += 1
                if env.last_state.min_separation < discomfort:
                    discomfort_steps += 1
                done = bool(terminated or truncated)
            outcome = str(info["raise_outcome"])
            outcomes.append(outcome)
            if outcome == "success":
                times.append(float(env.last_state.global_time))
            last = native[-1] if outcome in ("success", "collision") else 0.0
            term_only = [0.0] * (len(native) - 1) + [last]
            full.append(discounted_task_return(
                native, gamma=CROWDNAV_EVAL_GAMMA, time_step=dt, v_pref=v_pref
            ))
            terminal.append(discounted_task_return(
                term_only, gamma=CROWDNAV_EVAL_GAMMA, time_step=dt, v_pref=v_pref
            ))
            undiscounted.append(float(sum(native)))
    finally:
        env.close()
    return {
        "behaviour": behaviour,
        "phase": phase,
        "n_cases": n,
        "case_set_size": size,
        "env_config_path": env_config_path,
        "SR": outcomes.count("success") / n,
        "CR": outcomes.count("collision") / n,
        "TR": outcomes.count("timeout") / n,
        "nav_time": float(np.mean(times)) if times else float("nan"),
        "reward_full": float(np.mean(full)),
        "reward_terminal": float(np.mean(terminal)),
        "reward_undiscounted": float(np.mean(undiscounted)),
        "discomfort_frequency": discomfort_steps / max(1, total_steps),
    }


def compare_to_reference(
    measured: Dict[str, Any],
    reference: Dict[str, float] = PAPER_ORCA,
    tolerance: Dict[str, float] = TOLERANCE,
) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for key, ref in reference.items():
        got = float(measured.get(key, float("nan")))
        tol = float(tolerance[key])
        diff = got - float(ref)
        rows[key] = {
            "measured": got,
            "reference": float(ref),
            "diff": diff,
            "tolerance": tol,
            "ok": bool(math.isfinite(got) and abs(diff) <= tol),
        }
    return {"metrics": rows, "all_ok": all(r["ok"] for r in rows.values())}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CrowdNav ORCA vs published reference")
    p.add_argument("--env-config", default=None, help="Default: vendored upstream env.config")
    p.add_argument("--phase", default="test", choices=("train", "val", "test"))
    p.add_argument("--cases", type=int, default=None, help="Default: the whole case set")
    p.add_argument("--out", default=None, help="Write the report as JSON")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    measured = run_orca_check(env_config_path=args.env_config, phase=args.phase, n_cases=args.cases)
    report = {"measured": measured, **compare_to_reference(measured)}
    print(
        f"ORCA on {measured['phase']} ({measured['n_cases']}/{measured['case_set_size']} cases): "
        f"SR={measured['SR']:.3f} CR={measured['CR']:.3f} TR={measured['TR']:.3f} "
        f"time={measured['nav_time']:.2f}s"
    )
    print(
        f"reward: terminal-only={measured['reward_terminal']:.4f} "
        f"full={measured['reward_full']:.4f} undiscounted={measured['reward_undiscounted']:.4f} "
        f"| discomfort freq={measured['discomfort_frequency']:.3f}"
    )
    for key, row in report["metrics"].items():
        flag = "ok" if row["ok"] else "MISMATCH"
        print(f"  {key:16s} {row['measured']:8.4f} vs paper {row['reference']:8.4f} "
              f"(tol {row['tolerance']:g}) {flag}")
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        atomic_write_json(args.out, report)
        print(f"wrote {args.out}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

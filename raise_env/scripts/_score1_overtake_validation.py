#!/usr/bin/env python
"""Part 3/4 validation: Score1 zoo on regenerated Stage I dataset.

Reports audit rewards + passive_safe_crawl vs overtake_aware, plus a
weight-sensitivity check (with/without overtake_alignment term).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import raise_paths  # noqa: E402,F401
from domains.highway.reward_checks import make_highway_validator  # noqa: E402
from domains.highway.stage1 import (  # noqa: E402
    DEFAULT_STAGE1_DATASET,
    _W_OVERTAKE,
    _W_THROUGHPUT,
    load_highway_dataset,
    score_highway_dataset,
    _overtake_alignment,
    _throughput_alignment,
    _episode_return,
)
from domains.highway.state import default_smoke_states  # noqa: E402

ZOO = {
    "speed_only": "def compute_reward(state, memory):\n    return float(state.speed)\n",
    "progress_only": "def compute_reward(state, memory):\n    return float(state.progress)\n",
    "const_plus1": "def compute_reward(state, memory):\n    return 1.0\n",
    "prefer_20mps": (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n        return -50.0\n"
        "    return 1.0 - 0.2 * abs(state.speed - 20.0)\n"
    ),
    "passive_safe_crawl": (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n        return -50.0\n"
        "    if state.off_road:\n        return -20.0\n"
        "    # Survival + modest speed; NO credit for lane-change / passing.\n"
        "    return float(1.0 - 0.2 * abs(state.speed - 20.0) + 0.01 * state.progress)\n"
    ),
    "overtake_aware": (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n        return -50.0\n"
        "    if state.off_road:\n        return -20.0\n"
        "    r = 0.08 * state.progress + 0.02 * min(state.speed, 24.0)\n"
        "    prev = memory.get('prev_others')\n"
        "    cur = list(state.others or ())\n"
        "    if prev is not None:\n"
        "        for o in cur:\n"
        "            for px, py in prev:\n"
        "                if abs(float(o.y) - float(py)) < 2.0 and float(px) > 2.0 "
        "and float(o.x) <= 0.0:\n"
        "                    r += 5.0\n"
        "                    break\n"
        "    stored = []\n"
        "    for o in cur:\n"
        "        stored.append((float(o.x), float(o.y)))\n"
        "    memory['prev_others'] = stored\n"
        "    for o in cur:\n"
        "        if abs(float(o.y)) < 2.5 and float(o.x) > 0.0 and float(o.vx) < -0.5:\n"
        "            r += 0.2\n"
        "    return float(r)\n"
    ),
}


def _score_without_overtake(fn, trajs) -> float:
    """Recompute Score1 with overtake weight folded back into throughput."""
    from domains.highway.stage1 import (
        _spearman_component,
        _preference_auc,
        _crawl_penalty,
        _collision_decoy_penalty,
        _W_SPEARMAN,
        _W_PREFERENCE,
        _W_CRAWL_PEN,
        _W_CRASH_PEN,
    )
    import numpy as np

    spearman_mean, _, _, _ = _spearman_component(fn, trajs)
    returns = {t.trajectory_id: _episode_return(fn, t) for t in trajs}
    pref = _preference_auc(returns, trajs)
    thr = _throughput_alignment(returns, trajs)
    crawl = _crawl_penalty(returns, trajs)
    crash = _collision_decoy_penalty(returns, trajs)
    raw = (
        _W_SPEARMAN * spearman_mean
        + _W_PREFERENCE * pref
        + (_W_THROUGHPUT + _W_OVERTAKE) * thr
        - _W_CRAWL_PEN * crawl
        - _W_CRASH_PEN * crash
    )
    return float(np.clip(raw, -1.5, 1.5))


def main() -> int:
    path = str(ROOT / DEFAULT_STAGE1_DATASET)
    trajs = load_highway_dataset(path)
    validator = make_highway_validator(smoke_states=default_smoke_states())

    by_beh = {}
    for t in trajs:
        by_beh.setdefault(str(t.behavior), []).append(t)
    print(f"dataset={path} n={len(trajs)}")
    print("behavior counts (success only):")
    for beh, rows in sorted(by_beh.items()):
        succ = [t for t in rows if t.label == "success"]
        if not succ:
            continue
        ov = [float((t.metadata or {}).get("overtakes") or 0) for t in succ]
        sp = [float((t.metadata or {}).get("mean_speed") or 0) for t in succ]
        print(
            f"  {beh:16s} n_succ={len(succ):3d} "
            f"speed=[{min(sp):.1f},{max(sp):.1f}] "
            f"overtakes_sum={sum(ov):.0f} max_ov={max(ov):.0f}"
        )

    print(
        f"\n{'reward':22s} {'S1':>7s} {'ovt':>7s} {'thr':>7s} "
        f"{'pref':>7s} {'crawl':>7s} {'S1_no_ovt':>9s}"
    )
    scores = {}
    for name, code in ZOO.items():
        fn = validator.validate_code(code)
        res = score_highway_dataset(fn, trajs)
        s = res.scenario_scores or {}
        alt = _score_without_overtake(fn, trajs)
        scores[name] = float(res.score)
        print(
            f"{name:22s} {res.score:+7.3f} {s.get('overtake_alignment', 0):+7.3f} "
            f"{s.get('throughput', 0):+7.3f} {s.get('preference_auc', 0):+7.3f} "
            f"{s.get('crawl_penalty', 0):7.3f} {alt:+9.3f}"
        )

    print("\n--- Acceptance ---")
    ok_ovt = scores["overtake_aware"] > scores["passive_safe_crawl"]
    print(
        f"overtake_aware ({scores['overtake_aware']:+.3f}) > "
        f"passive_safe_crawl ({scores['passive_safe_crawl']:+.3f}): "
        f"{'PASS' if ok_ovt else 'FAIL'}"
    )
    # prefer_20mps should not be unfairly crushed now that safe_cruise exists;
    # it should beat pure speed_only (which loves collisions/high speed).
    ok_cruise = scores["prefer_20mps"] > scores["speed_only"] - 0.05
    print(
        f"prefer_20mps ({scores['prefer_20mps']:+.3f}) vs "
        f"speed_only ({scores['speed_only']:+.3f}): "
        f"{'PASS (not dominated)' if ok_cruise else 'CHECK'}"
    )
    top = max(scores, key=scores.get)
    print(f"top reward by Score1: {top} ({scores[top]:+.3f})")
    ok_not_passive_top = top != "passive_safe_crawl"
    print(
        f"passive_safe_crawl is not top: {'PASS' if ok_not_passive_top else 'FAIL'}"
    )

    # Weight sensitivity: does overtake term change rank of aware vs passive?
    # Already printed S1_no_ovt; report delta.
    print(
        f"\nWeight choice: overtake={_W_OVERTAKE}, throughput={_W_THROUGHPUT} "
        f"(taken from former throughput 0.35)."
    )
    return 0 if (ok_ovt and ok_not_passive_top) else 2


if __name__ == "__main__":
    raise SystemExit(main())

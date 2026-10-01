"""Score1 (highway v1) on trivial rewards + final-population codes of one run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import raise_paths  # noqa: E402,F401
from domains.highway.reward_checks import make_highway_validator  # noqa: E402
from domains.highway.stage1 import (  # noqa: E402
    DEFAULT_STAGE1_DATASET,
    load_highway_dataset,
    score_highway_dataset,
)
from domains.highway.state import default_smoke_states  # noqa: E402

RUN = ROOT / "results" / "highway_4h_20260930_124808"

ZOO = {
    "const_plus1": "def compute_reward(state, memory):\n    return 1.0\n",
    "const_minus1": "def compute_reward(state, memory):\n    return -1.0\n",
    "const_plus1_crash_minus1000": (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n        return -1000.0\n"
        "    return 1.0\n"
    ),
    "crash_only_minus100": (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n        return -100.0\n"
        "    return 0.0\n"
    ),
    "speed_only": "def compute_reward(state, memory):\n    return float(state.speed)\n",
    "progress_only": "def compute_reward(state, memory):\n    return float(state.progress)\n",
    "speed_ge25_bonus": (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n        return -50.0\n"
        "    return 1.0 if state.speed >= 25.0 else 0.2\n"
    ),
    "prefer_20mps": (
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n        return -50.0\n"
        "    return 1.0 - 0.2 * abs(state.speed - 20.0)\n"
    ),
}


def main() -> None:
    trajs = load_highway_dataset(str(ROOT / DEFAULT_STAGE1_DATASET))
    validator = make_highway_validator(smoke_states=default_smoke_states())
    print(f"dataset trajectories: {len(trajs)}")
    header = f"{'reward':30s} {'S1':>7s} {'spear':>7s} {'pref':>7s} {'thr':>7s} {'crawl':>7s} {'decoy':>7s}"
    print(header)

    def _row(name: str, code: str) -> float:
        fn = validator.validate_code(code)
        res = score_highway_dataset(fn, trajs)
        s = res.scenario_scores
        print(
            f"{name:30s} {res.score:+7.3f} {s['spearman']:+7.3f} {s['preference_auc']:+7.3f} "
            f"{s['throughput']:+7.3f} {s['crawl_penalty']:7.3f} {s['collision_decoy_penalty']:7.3f}"
        )
        return float(res.score)

    for name, code in ZOO.items():
        _row(name, code)

    print("\nfinal population (Stage II SR from stage2_population.json):")
    pop = json.loads((RUN / "stage2_population.json").read_text(encoding="utf-8"))["population"]
    s1s, srs = [], []
    for cand in pop:
        sr = float(cand["metadata"]["last_metrics"]["SR"])
        s1s.append(_row(f"{cand['candidate_id']} (SR={sr:.2f})", cand["code"]))
        srs.append(sr)
    print(f"\nspearman(Score1_recomputed, StageII_SR) n={len(srs)}: {spearmanr(s1s, srs).statistic:+.3f}")


if __name__ == "__main__":
    main()

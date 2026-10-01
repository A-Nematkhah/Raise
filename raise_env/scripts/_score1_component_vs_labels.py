"""Correlate highway Score1 components with Stage II labels for one run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def main(run: str) -> None:
    run_dir = ROOT / run
    comps = {}
    for path in sorted((run_dir / "closed_loop").glob("pop_epoch_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pop = data if isinstance(data, list) else (data.get("population") or data.get("candidates") or [])
        for cand in pop:
            scen = (cand.get("metadata") or {}).get("score1_scenario_scores")
            if scen:
                comps[cand["candidate_id"]] = dict(scen, score1=cand.get("score"))
    feats = {
        row["example_id"]: row
        for row in map(json.loads, (run_dir / "surrogate_dataset" / "features.jsonl").read_text(encoding="utf-8").splitlines())
    }
    labels = [json.loads(l) for l in (run_dir / "surrogate_dataset" / "labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for lab in labels:
        cid = feats[lab["example_id"]]["candidate_id"]
        if cid in comps:
            rows.append((cid, lab["SR"], lab["mean_speed"], comps[cid]))
    print(f"n with components: {len(rows)} of {len(labels)}")
    for key in sorted(rows[0][3]):
        xs = [float(r[3].get(key, 0.0)) for r in rows]
        rho = spearmanr(xs, [r[1] for r in rows]).statistic
        print(f"{key:26s} rho_vs_SR={rho:+.3f} range=({min(xs):.3f},{max(xs):.3f})")
    print()
    for cid, sr, spd, s in sorted(rows, key=lambda r: -float(r[3]["score1"])):
        print(
            f"{cid:9s} SR={sr:.2f} v={spd:.1f} S1={s['score1']:.3f} "
            f"sp={s.get('spearman', 0):.3f} pref={s.get('preference_auc', 0):.3f} "
            f"thr={s.get('throughput', 0):.3f} crawl={s.get('crawl_penalty', 0):.3f} "
            f"decoy={s.get('collision_decoy_penalty', 0):.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/highway_4h_20260930_124808")

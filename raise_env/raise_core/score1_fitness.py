"""Spearman correlation between Stage I Score1 and Stage II fitness for a run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None

    def ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx <= 0.0 or deny <= 0.0:
        return None
    return num / (denx * deny)


def _load_trained(run_dir: Path) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    cl = run_dir / "closed_loop"
    for path in sorted(cl.glob("pop_epoch_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for cand in data.get("candidates") or []:
            md = cand.get("metadata") or {}
            if md.get("fitness") is None:
                continue
            if md.get("score1_rejected"):
                continue
            cid = str(cand.get("candidate_id"))
            hold = md.get("last_metrics") or {}
            if isinstance(hold.get("holdout"), dict):
                hold = hold["holdout"]
            score = cand.get("score")
            if score is None:
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score_f):
                continue
            rows[cid] = {
                "id": cid,
                "score1": score_f,
                "fitness": float(md["fitness"]),
                "SR": float(hold.get("SR", hold.get("sr", 0.0)) or 0.0),
                "CR": float(hold.get("CR", hold.get("cr", 0.0)) or 0.0),
            }
    return list(rows.values())


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    trained = _load_trained(run_dir)
    functional = [r for r in trained if r["SR"] >= 0.3]
    collapsed = [r for r in trained if r["CR"] >= 0.8]
    others = [r for r in trained if r["CR"] < 0.8]

    def pack(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {"n": 0, "spearman": None}
        rho = _spearman([r["score1"] for r in subset], [r["fitness"] for r in subset])
        return {"n": len(subset), "spearman": None if rho is None else round(rho, 4)}

    def score_dist(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {"n": 0}
        vals = sorted(r["score1"] for r in subset)
        return {
            "n": len(vals),
            "min": round(vals[0], 4),
            "median": round(vals[len(vals) // 2], 4),
            "max": round(vals[-1], 4),
            "mean": round(sum(vals) / len(vals), 4),
        }

    return {
        "n_trained": len(trained),
        "all": pack(trained),
        "functional_sr_ge_0.3": pack(functional),
        "collapsed_cr_ge_0.8_score1": score_dist(collapsed),
        "noncollapsed_score1": score_dist(others),
    }


def format_report_line(summary: Dict[str, Any]) -> str:
    all_p = summary["all"]
    fun = summary["functional_sr_ge_0.3"]
    col = summary["collapsed_cr_ge_0.8_score1"]
    oth = summary["noncollapsed_score1"]
    return (
        "Score1 vs fitness: "
        f"Spearman={all_p['spearman']} (n={all_p['n']}); "
        f"functional SR>=0.3 Spearman={fun['spearman']} (n={fun['n']}); "
        f"collapsed CR>=0.8 Score1 mean={col.get('mean', 'n/a')} (n={col.get('n', 0)}); "
        f"others Score1 mean={oth.get('mean', 'n/a')} (n={oth.get('n', 0)})"
    )


__all__ = ["analyze_run", "format_report_line"]

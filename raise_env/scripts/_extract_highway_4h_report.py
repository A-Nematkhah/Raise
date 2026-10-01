"""Deep extract for highway_4h_20260930_124808 report."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "highway_4h_20260930_124808"
OUT = ROOT / "results" / "_reward_components_verification" / "highway_4h_20260930_124808_extract.txt"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def walk_cands(obj: Any, out: Dict[str, Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        cid = obj.get("candidate_id")
        if cid and (obj.get("code") or obj.get("metadata")):
            prev = out.get(cid, {})
            merged = dict(prev)
            for k, v in obj.items():
                if v in (None, "", [], {}):
                    continue
                if k == "metadata" and isinstance(v, dict):
                    md = dict(prev.get("metadata") or {})
                    md.update(v)
                    merged["metadata"] = md
                else:
                    merged[k] = v
            out[cid] = merged
        for v in obj.values():
            walk_cands(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_cands(v, out)


def main() -> None:
    lines: List[str] = []
    cfg = load(RUN / "config.json")
    man = load(RUN / "manifest.json")
    lines.append("=== CONFIG (subset) ===")
    for k in sorted(cfg):
        if any(
            s in k
            for s in (
                "population",
                "generation",
                "k2",
                "k3",
                "stage3",
                "llm",
                "seed",
                "evolve",
                "calibration",
                "action",
                "pareto",
                "proxy",
                "surrogate",
                "elit",
                "device",
                "domain",
            )
        ):
            lines.append(f"{k}={cfg[k]!r}")
    lines.append("")
    lines.append("=== MANIFEST top keys ===")
    lines.append(str(sorted(man.keys())))
    for k in ("wall_clock_s", "wall_clock", "total_wall_s", "stage1_best", "stage2_best", "stage3_best", "final_candidate", "proxy_consistency", "timings"):
        if k in man:
            lines.append(f"{k}={json.dumps(man[k], ensure_ascii=False)[:1500]}")
    lines.append("")

    cands: Dict[str, Dict[str, Any]] = {}
    for p in [
        RUN / "stage1_population.json",
        RUN / "stage2_population.json",
        RUN / "stage3_population.json",
        RUN / "closed_loop" / "checkpoint.json",
        RUN / "best_stage1.json",
        RUN / "best_stage2.json",
        RUN / "best_stage3.json",
        RUN / "final_candidate.json",
    ]:
        if p.is_file():
            walk_cands(load(p), cands)

    lines.append(f"=== CANDIDATES n={len(cands)} ===")
    for cid in sorted(cands):
        c = cands[cid]
        md = c.get("metadata") or {}
        pm = md.get("proxy_metrics") or md.get("stage2_metrics") or md.get("metrics") or {}
        keys_interest = {}
        for key in (
            "SR",
            "CR",
            "TR",
            "mean_speed",
            "progress",
            "highway_fitness",
            "fitness",
            "lane_change_rate",
            "overtakes_per_km",
            "soft_success",
        ):
            if isinstance(pm, dict) and key in pm:
                keys_interest[key] = pm[key]
            elif key in md:
                keys_interest[key] = md[key]
        trends = md.get("reward_component_trends") or {}
        lines.append(
            f"{cid}\torigin={c.get('origin')}\tparents={c.get('parent_ids')}\t"
            f"score={c.get('score')}\tpareto_rank={md.get('pareto_rank')}\tfront={md.get('pareto_front')}\t"
            f"feasible={md.get('pareto_feasible')}\ttrend_keys={sorted(trends)[:10]}\t"
            f"metrics={json.dumps(keys_interest, ensure_ascii=False)[:400]}"
        )
        diag = md.get("llm_diagnosis")
        if diag:
            lines.append(f"    diagnosis[{cid}]: {str(diag)[:500]}")

    lines.append("")
    lines.append("=== stage1_rejections categories ===")
    rej_path = RUN / "closed_loop" / "stage1_rejections.jsonl"
    cats = Counter()
    phases = Counter()
    for line in rej_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cats[str(row.get("category"))] += 1
        phases[str(row.get("phase"))] += 1
    lines.append(f"by_category={dict(cats)}")
    lines.append(f"by_phase={dict(phases)}")

    lines.append("")
    lines.append("=== surrogate metrics ===")
    lines.append((RUN / "surrogate_model" / "metrics.json").read_text(encoding="utf-8"))
    lines.append("=== last_refit ===")
    lines.append((RUN / "closed_loop" / "last_refit.json").read_text(encoding="utf-8"))
    lines.append("=== al_steps ===")
    lines.append((RUN / "closed_loop" / "al_steps.jsonl").read_text(encoding="utf-8"))
    lines.append("=== proxy_consistency ===")
    lines.append((RUN / "proxy_consistency.json").read_text(encoding="utf-8"))
    lines.append("=== surrogate_gate_stage3 ===")
    lines.append((RUN / "surrogate_gate_stage3.json").read_text(encoding="utf-8"))
    lines.append("=== stage3_elite_assembly ===")
    lines.append((RUN / "stage3_elite_assembly.json").read_text(encoding="utf-8"))
    lines.append("=== surrogate_preds_stage2 ===")
    lines.append((RUN / "surrogate_preds_stage2.json").read_text(encoding="utf-8")[:3000])

    lines.append("")
    lines.append("=== stage3 checkpoint summary ===")
    s3 = load(RUN / "stage3" / "checkpoint.json")
    lines.append(f"keys={sorted(s3.keys())}")
    for k in ("status", "phase", "best", "best_id", "history", "rounds", "candidates", "results"):
        if k in s3:
            v = s3[k]
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:2500]}")

    lines.append("")
    lines.append("=== stage2 diagnostics component summary per candidate ===")
    for folder in sorted((RUN / "closed_loop" / "stage2_train").iterdir()):
        diag = folder / "diagnostics_rollout.jsonl"
        if not diag.is_file():
            continue
        comps: Dict[str, List[float]] = {}
        ep_rew = []
        for line in diag.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("ep_rew_mean") is not None:
                ep_rew.append(float(row["ep_rew_mean"]))
            for k, v in (row.get("reward_component_means") or {}).items():
                comps.setdefault(k, []).append(float(v))
        flat = [k for k, vs in comps.items() if len(vs) > 2 and max(vs) - min(vs) < 1e-6]
        lines.append(
            f"{folder.name}: n_rollouts={len(next(iter(comps.values()), []))} keys={sorted(comps)} "
            f"flat={flat} ep_rew_last={ep_rew[-1] if ep_rew else None}"
        )

    lines.append("")
    lines.append("=== stage3 diagnostics component summary ===")
    for folder in sorted((RUN / "stage3_train").iterdir()):
        diag = folder / "diagnostics_rollout.jsonl"
        if not diag.is_file():
            continue
        comps: Dict[str, List[float]] = {}
        for line in diag.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for k, v in (row.get("reward_component_means") or {}).items():
                comps.setdefault(k, []).append(float(v))
        flat = [k for k, vs in comps.items() if len(vs) > 2 and max(vs) - min(vs) < 1e-6]
        lines.append(f"{folder.name}: keys={sorted(comps)} flat={flat} n={len(next(iter(comps.values()), []))}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} lines={len(lines)}")


if __name__ == "__main__":
    main()

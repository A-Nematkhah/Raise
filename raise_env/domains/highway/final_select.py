"""Deliberate final policy pick from a highway Pareto front (no silent scalar)."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from raise_core.explore import RewardCandidate


def crowdnav_history_scalar(metrics: Mapping[str, Any]) -> float:
    """CrowdNav / legacy ``--best-ever`` scalar: SR − CR − 0.5·TR only."""
    return float(
        float(metrics.get("SR", 0) or 0)
        - float(metrics.get("CR", 0) or 0)
        - 0.5 * float(metrics.get("TR", 0) or 0)
    )


def _metrics_src(md: Mapping[str, Any]) -> Mapping[str, Any]:
    m = md.get("last_metrics") if isinstance(md.get("last_metrics"), dict) else md
    if not isinstance(m, dict):
        return {}
    holdout = m.get("holdout")
    if isinstance(holdout, dict) and holdout:
        return holdout
    return m


def raw_objective_row(candidate: RewardCandidate) -> Dict[str, Any]:
    """Side-by-side raw metrics for a human reading the front."""
    md = candidate.metadata or {}
    src = _metrics_src(md)

    def _f(*keys: str, default: float = float("nan")) -> float:
        for k in keys:
            if k in src and src[k] is not None:
                try:
                    return float(src[k])
                except (TypeError, ValueError):
                    continue
        return float(default)

    return {
        "candidate_id": str(candidate.candidate_id),
        "SR": _f("SR", "sr"),
        "CR": _f("CR", "cr"),
        "TR": _f("TR", "tr"),
        "mean_speed": _f("mean_speed", "ITR"),
        "progress": _f("PL", "mean_progress", "progress"),
        "soft_success": _f("soft_success", default=0.0),
        "pareto_rank": md.get("pareto_rank"),
        "pareto_front": md.get("pareto_front"),
        "pareto_front0": md.get("pareto_front0"),
        "pareto_feasible": md.get("pareto_feasible"),
        "legacy_scalar_SR_CR_TR": crowdnav_history_scalar(src),
    }


def ensure_pareto_stamped(
    candidates: Sequence[RewardCandidate],
    *,
    ref: Any = None,
) -> List[RewardCandidate]:
    """Stamp pareto_* if missing (recompute from last_metrics)."""
    pop = list(candidates)
    if not pop:
        return pop
    have_front = all(
        "pareto_front" in (c.metadata or {})
        and (c.metadata or {}).get("pareto_rank") is not None
        for c in pop
        if (c.metadata or {}).get("last_metrics")
    )
    labeled = [
        c for c in pop if isinstance((c.metadata or {}).get("last_metrics"), dict)
    ]
    if have_front and labeled:
        return pop

    from raise_core.raise_loop.evolve_rank import rank_population_pareto

    return rank_population_pareto(pop, ref=ref)


def front0_candidates(
    candidates: Sequence[RewardCandidate],
    *,
    ref: Any = None,
) -> List[RewardCandidate]:
    """Feasible Pareto front 0 only (after ensuring stamps)."""
    stamped = ensure_pareto_stamped(candidates, ref=ref)
    front = [
        c
        for c in stamped
        if bool((c.metadata or {}).get("pareto_front0"))
        or (
            (c.metadata or {}).get("pareto_front") == 0
            and (c.metadata or {}).get("pareto_feasible", True)
        )
    ]
    if front:
        return front
    # Fallback: feasible with best ordinal ranks sharing front via recompute.
    return [
        c
        for c in stamped
        if (c.metadata or {}).get("pareto_feasible") is True
        and int((c.metadata or {}).get("pareto_front") or -1) == 0
    ]


def format_front_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "(empty Pareto front 0 — no feasible non-dominated candidates)"
    header = (
        f"{'id':<16} {'SR':>5} {'CR':>5} {'TR':>5} {'speed':>6} "
        f"{'prog':>7} {'soft':>5} {'rank':>4} {'feas':>5} {'legacy':>7}"
    )
    lines = [
        "Pareto front 0 (feasible non-dominated). No automatic winner — "
        "pick one trade-off explicitly with --candidate-id.",
        header,
        "-" * len(header),
    ]
    for r in rows:
        lines.append(
            f"{str(r['candidate_id']):<16} "
            f"{float(r['SR']):5.2f} {float(r['CR']):5.2f} {float(r['TR']):5.2f} "
            f"{float(r['mean_speed']):6.1f} {float(r['progress']):7.0f} "
            f"{float(r['soft_success']):5.2f} "
            f"{str(r.get('pareto_rank')):>4} "
            f"{str(r.get('pareto_feasible')):>5} "
            f"{float(r['legacy_scalar_SR_CR_TR']):7.3f}"
        )
    return "\n".join(lines)


def load_population_dicts(run_dir: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load serialized population payloads from a highway/CrowdNav run dir."""
    search = [
        os.path.join(run_dir, "stage2_population.json"),
        os.path.join(run_dir, "stage3_population.json"),
        os.path.join(run_dir, "stage3_elite_assembly.json"),
    ]
    # Newest closed-loop pop snapshot.
    pops = sorted(glob.glob(os.path.join(run_dir, "closed_loop", "pop_epoch_*.json")))
    if pops:
        search.insert(0, pops[-1])

    for path in search:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh) or {}
        if "population" in data and isinstance(data["population"], list):
            return list(data["population"]), path
        # pop_epoch_*.json may only store id lists — skip those.
        if isinstance(data, list):
            return list(data), path
    raise FileNotFoundError(
        f"No population JSON with candidates under {run_dir} "
        f"(expected stage2_population.json or similar)"
    )


def dicts_to_candidates(items: Sequence[Mapping[str, Any]]) -> List[RewardCandidate]:
    out: List[RewardCandidate] = []
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        cid = str(raw.get("candidate_id") or "")
        if not cid:
            continue
        md = dict(raw.get("metadata") or {})
        # Some serializers flatten metrics onto the candidate.
        if "last_metrics" not in md and any(k in raw for k in ("SR", "mean_speed", "CR")):
            md["last_metrics"] = {
                k: raw[k]
                for k in (
                    "SR",
                    "CR",
                    "TR",
                    "mean_speed",
                    "PL",
                    "mean_progress",
                    "soft_success",
                    "domain",
                    "holdout",
                )
                if k in raw
            }
        for k in (
            "pareto_rank",
            "pareto_front",
            "pareto_front0",
            "pareto_feasible",
            "pareto_n",
            "pareto_score",
            "fitness",
            "selection_scalar",
        ):
            if k in raw and k not in md:
                md[k] = raw[k]
        out.append(
            RewardCandidate(
                candidate_id=cid,
                code=str(raw.get("code") or "def compute_reward(state, memory):\n    return 0.0\n"),
                valid=True,
                origin=str(raw.get("origin") or "loaded"),
                score=raw.get("score"),
                metadata=md,
            )
        )
    return out


def list_front0_from_run(
    run_dir: str,
    *,
    ref: Any = None,
) -> Tuple[List[Dict[str, Any]], List[RewardCandidate], str]:
    """Return (rows, front_candidates, source_path) for a run directory."""
    items, path = load_population_dicts(run_dir)
    cands = dicts_to_candidates(items)
    if not cands:
        raise ValueError(f"Population file {path} has no candidates")
    front = front0_candidates(cands, ref=ref)
    rows = [raw_objective_row(c) for c in front]
    return rows, front, path


def run_domain(run_dir: str) -> str:
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh) or {}
        return str(cfg.get("domain") or "crowdnav").strip().lower()
    return "crowdnav"

"""Append-only per-candidate / per-generation ledger for closed-loop runs.

One ``ledger.jsonl`` row per candidate per generation and one
``generations.jsonl`` row per generation, written after breeding order is
fixed. Rows are built from values already decided by the loop; nothing here
feeds back into selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from raise_core.explore import RewardCandidate
from raise_core.raise_loop.logging_io import closed_loop_dir

LEDGER_FILE = "ledger.jsonl"
GENERATIONS_FILE = "generations.jsonl"

_PROMPT_TYPE_BY_ORIGIN = {
    "initial": "D1",
    "random": "D1_random",
    "crossover": "D2_crossover",
    "mutation": "D2_mutation",
    "in_loop_d3": "D3_in_loop",
    "seed_fallback": "seed_fallback",
    "crossover_fallback": "crossover_fallback_clone",
    "mutation_fallback": "mutation_fallback_clone",
}

_FALLBACK_ORIGINS = frozenset(
    {"seed_fallback", "crossover_fallback", "mutation_fallback"}
)


def code_hash(code: Optional[str]) -> str:
    return hashlib.sha256(str(code or "").encode("utf-8")).hexdigest()[:16]


def is_fallback_clone(cand: RewardCandidate) -> bool:
    md = cand.metadata or {}
    return str(cand.origin) in _FALLBACK_ORIGINS or bool(md.get("llm_fallback_clone"))


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def label_reason(
    candidate_id: str,
    *,
    selected_ids: Iterable[str],
    gate_report: Mapping[str, Any],
    al_report: Mapping[str, Any],
) -> str:
    """Why a candidate was (not) sent to Stage II this generation."""
    cid = str(candidate_id)
    selected = {str(x) for x in selected_ids}
    if cid not in selected:
        dropped = {str(x) for x in (gate_report.get("dropped_ids") or [])}
        return "gate_dropped" if cid in dropped else "not_selected"
    if bool(gate_report.get("soft", True)):
        return f"soft_gate:{gate_report.get('reason')}"
    if cid in {str(x) for x in (al_report.get("pick_ids") or [])}:
        kept = {str(x) for x in (gate_report.get("kept_ids") or [])}
        return "gate_kept+al_pick" if cid in kept else "al_pick"
    if cid in {str(x) for x in (gate_report.get("kept_ids") or [])}:
        return "gate_kept"
    return "top_up"


def candidate_row(
    cand: RewardCandidate,
    *,
    generation: int,
    evolve_index: int,
    elite_id: Optional[str],
    reason: str,
    label_status: Optional[str],
) -> Dict[str, Any]:
    md = cand.metadata or {}
    metrics = md.get("last_metrics") if isinstance(md.get("last_metrics"), dict) else {}
    warm_ckpt = md.get("warm_start_checkpoint")
    return {
        "generation": int(generation),
        "candidate_id": str(cand.candidate_id),
        "origin": str(cand.origin),
        "prompt_type": _PROMPT_TYPE_BY_ORIGIN.get(str(cand.origin), str(cand.origin)),
        "parent_ids": [str(p) for p in (cand.parent_ids or ())],
        "code_hash": code_hash(cand.code),
        "code_len": len(str(cand.code or "")),
        "valid": bool(cand.valid),
        "validation_error": cand.validation_error,
        "is_fallback_clone": is_fallback_clone(cand),
        "score1": _finite_or_none(cand.score),
        "score1_rejected": bool(md.get("score1_rejected", False)),
        "score1_reject_reason": md.get("score1_reject_reason"),
        "score1_components": md.get("score1_scenario_scores"),
        "selected_for_stage2": not reason.startswith(("gate_dropped", "not_selected")),
        "label_reason": reason,
        "label_status": label_status,
        "warm_start": (
            bool(metrics.get("warm_start")) if metrics else bool(warm_ckpt)
        ),
        "warm_start_checkpoint": os.path.basename(str(warm_ckpt)) if warm_ckpt else None,
        "train_steps": md.get("train_steps"),
        "train_seed": md.get("train_seed"),
        "eval_seed_base": md.get("eval_seed_base"),
        "eval_episodes": md.get("eval_episodes"),
        "ppo_n_steps": md.get("ppo_n_steps"),
        "eval_deterministic": md.get("eval_deterministic"),
        "SR": _finite_or_none(metrics.get("SR")),
        "CR": _finite_or_none(metrics.get("CR")),
        "TR": _finite_or_none(metrics.get("TR")),
        "mean_speed": _finite_or_none(metrics.get("mean_speed", metrics.get("ITR"))),
        "progress": _finite_or_none(metrics.get("mean_progress", metrics.get("PL"))),
        "fitness": _finite_or_none(md.get("fitness")),
        "evolve_index": int(evolve_index),
        "is_elite": elite_id is not None and str(cand.candidate_id) == str(elite_id),
        "pareto_front": md.get("pareto_front"),
        "pareto_feasible": md.get("pareto_feasible"),
    }


def _metric_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = r.get(key)
        if v is not None:
            out.append(float(v))
    return out


def generation_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    generation: int,
    buckets: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    al_report: Mapping[str, Any],
    d3_report: Mapping[str, Any],
    elite_id: Optional[str],
) -> Dict[str, Any]:
    statuses = [str(r.get("label_status") or "") for r in rows]
    sr = _metric_values(rows, "SR")
    cr = _metric_values(rows, "CR")
    prog = _metric_values(rows, "progress")
    seed_bases = sorted(
        {int(r["eval_seed_base"]) for r in rows if r.get("eval_seed_base") is not None}
    )
    return {
        "generation": int(generation),
        "n_population": len(rows),
        "n_valid": sum(1 for r in rows if r.get("valid")),
        "n_fallback_clones": sum(1 for r in rows if r.get("is_fallback_clone")),
        "n_score1_rejected": sum(1 for r in rows if r.get("score1_rejected")),
        "n_unique_code": len({r.get("code_hash") for r in rows}),
        "buckets": dict(buckets),
        "n_selected_for_stage2": sum(1 for r in rows if r.get("selected_for_stage2")),
        "n_trained_ok": statuses.count("ok"),
        "n_trained_failed": statuses.count("failed"),
        "n_reused_label": sum(1 for s in statuses if s.startswith("skipped")),
        "n_warm_started": sum(
            1 for r in rows if r.get("warm_start") and r.get("label_status") == "ok"
        ),
        "eval_seed_bases": seed_bases,
        "gate": {
            "soft": bool(gate_report.get("soft", True)),
            "reason": gate_report.get("reason"),
            "ready_reason": gate_report.get("ready_reason"),
            "dropped_ids": list(gate_report.get("dropped_ids") or []),
        },
        "al": {
            "enabled": bool(al_report.get("enabled", False)),
            "pick_ids": list(al_report.get("pick_ids") or []),
            "n_added_by_al": int(al_report.get("n_added_by_al", 0) or 0),
        },
        "d3": dict(d3_report),
        "elite_id": elite_id,
        "best_SR": max(sr) if sr else None,
        "median_SR": statistics.median(sr) if sr else None,
        "min_CR": min(cr) if cr else None,
        "best_progress": max(prog) if prog else None,
    }


def _append_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, default=str))
            fh.write("\n")


def write_generation_ledger(
    output_dir: str,
    ranked_for_evo: Sequence[RewardCandidate],
    *,
    generation: int,
    selected_ids: Iterable[str],
    label_statuses: Mapping[str, str],
    gate_report: Mapping[str, Any],
    al_report: Mapping[str, Any],
    d3_report: Mapping[str, Any],
    buckets: Mapping[str, Any],
    elite_id: Optional[str],
) -> Dict[str, Any]:
    """Append candidate rows + one generation summary; return the summary."""
    selected = [str(x) for x in selected_ids]
    rows = []
    for i, cand in enumerate(ranked_for_evo):
        cid = str(cand.candidate_id)
        rows.append(
            candidate_row(
                cand,
                generation=generation,
                evolve_index=i,
                elite_id=elite_id,
                reason=label_reason(
                    cid,
                    selected_ids=selected,
                    gate_report=gate_report,
                    al_report=al_report,
                ),
                label_status=label_statuses.get(cid),
            )
        )
    summary = generation_row(
        rows,
        generation=generation,
        buckets=buckets,
        gate_report=gate_report,
        al_report=al_report,
        d3_report=d3_report,
        elite_id=elite_id,
    )
    root = closed_loop_dir(output_dir)
    _append_jsonl(os.path.join(root, LEDGER_FILE), rows)
    _append_jsonl(os.path.join(root, GENERATIONS_FILE), [summary])
    return summary

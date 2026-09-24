"""Per-epoch selection: gate + in-loop AL → Stage II labeling set."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from raise_core.active_learning.acquire import execute_query
from raise_core.active_learning.query import score_queries
from raise_core.explore import RewardCandidate
from raise_core.surrogate.features import code_sha256
from raise_core.surrogate.gate import gate_population

logger = logging.getLogger(__name__)


def unique_by_code(candidates: Sequence[RewardCandidate]) -> List[RewardCandidate]:
    seen: set[str] = set()
    out: List[RewardCandidate] = []
    for c in candidates:
        digest = code_sha256(str(c.code or ""))
        if digest in seen:
            continue
        seen.add(digest)
        out.append(c)
    return out


def _by_id(population: Sequence[RewardCandidate]) -> Dict[str, RewardCandidate]:
    return {str(c.candidate_id): c for c in population}


def top_up_uncertain(
    population: Sequence[RewardCandidate],
    predictions: Sequence[Dict[str, Any]],
    already: Sequence[RewardCandidate],
    need: int,
) -> List[RewardCandidate]:
    """Add highest-uncertainty candidates until ``need`` unique codes."""
    have = unique_by_code(already)
    if len(have) >= need:
        return have[:need]
    have_ids = {str(c.candidate_id) for c in have}
    by_id = _by_id(population)
    ranked = sorted(
        predictions,
        key=lambda r: float(r.get("uncertainty") or 0.0),
        reverse=True,
    )
    for row in ranked:
        cid = str(row.get("candidate_id") or "")
        if cid in have_ids:
            continue
        cand = by_id.get(cid)
        if cand is None:
            continue
        have.append(cand)
        have_ids.add(cid)
        if len(have) >= need:
            break
    return have


def select_to_label(
    population: Sequence[RewardCandidate],
    *,
    epoch: int,
    n_labeled: int,
    min_labels_for_gate: int,
    predictions: Optional[Sequence[Dict[str, Any]]],
    model_ready: bool,
    drop_fraction: float,
    max_uncertainty_to_drop: float,
    min_keep: int,
    min_stage2_per_gen: int,
    al_enabled: bool,
    al_max_per_epoch: int,
    al_allow_stage1_requests: bool,
    al_root: str,
    score1_results: Optional[Sequence[Any]] = None,
) -> Tuple[List[RewardCandidate], Dict[str, Any], Dict[str, Any]]:
    """
    Gen0 / soft-gate: label everyone.
    Hard gate: survivors ∪ AL stage2 picks (+ optional stage1_scenario side requests).
    """
    pop = list(population)
    gate_report: Dict[str, Any] = {"enabled": False, "soft": True}
    al_report: Dict[str, Any] = {"enabled": False, "n_al_stage2": 0, "n_stage1_requests": 0}

    soft = (
        epoch <= 0
        or not model_ready
        or predictions is None
        or n_labeled < int(min_labels_for_gate)
    )
    if soft:
        gate_report = {
            "enabled": False,
            "soft": True,
            "reason": (
                "gen0"
                if epoch <= 0
                else (
                    "no_model"
                    if not model_ready
                    else f"n_labeled={n_labeled}<{min_labels_for_gate}"
                )
            ),
            "n_kept": len(pop),
            "n_dropped": 0,
        }
        return unique_by_code(pop), gate_report, al_report

    survivors, gate_report = gate_population(
        pop,
        list(predictions),
        drop_fraction=float(drop_fraction),
        max_uncertainty_to_drop=float(max_uncertainty_to_drop),
        min_keep=int(min_keep),
    )
    gate_report["soft"] = False
    gate_report["enabled"] = True
    to_label = unique_by_code(survivors)

    if al_enabled:
        queries = score_queries(
            list(pop),
            surrogate_preds=list(predictions),
            score1_results=list(score1_results) if score1_results is not None else None,
            top_k=max(int(al_max_per_epoch) + 1, int(al_max_per_epoch)),
            max_stage2_labels=int(al_max_per_epoch),
            max_stage1_scenarios=1 if al_allow_stage1_requests else 0,
        )
        by_id = _by_id(pop)
        al_picks: List[RewardCandidate] = []
        n_s1 = 0
        for q in queries:
            if q.kind == "stage2_label":
                cid = str((q.payload or {}).get("candidate_id") or "")
                cand = by_id.get(cid)
                if cand is not None:
                    al_picks.append(cand)
            elif q.kind == "stage1_scenario" and al_allow_stage1_requests:
                res = execute_query(
                    q,
                    config={
                        "queue_root": al_root,
                        "stage1_extra_dir": f"{al_root.rstrip('/')}/stage1_extra",
                    },
                )
                if res.get("status") == "ok":
                    n_s1 += 1
        to_label = unique_by_code(list(to_label) + al_picks)
        al_report = {
            "enabled": True,
            "n_al_stage2": len(unique_by_code(al_picks)),
            "n_stage1_requests": n_s1,
            "n_queries": len(queries),
        }

    if len(to_label) < int(min_stage2_per_gen):
        to_label = top_up_uncertain(
            pop, list(predictions), to_label, int(min_stage2_per_gen)
        )
        al_report["topped_up_to"] = len(to_label)

    return to_label, gate_report, al_report

"""
Surrogate predict + Stage-III gate helpers for ``RaisePipeline``.

Opt-in: only runs when a fitted model directory is provided and contains
``model.joblib``. Does not change paper Stage III claims when disabled.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from raise_core.surrogate.features import extract_candidate_features
from raise_core.surrogate.model import SurrogateModel, SurrogatePrediction
from raise_core.surrogate.targets import quality_from_y_hat

logger = logging.getLogger(__name__)


def surrogate_model_ready(model_dir: Optional[str]) -> bool:
    if not model_dir or not str(model_dir).strip():
        return False
    return os.path.isfile(os.path.join(str(model_dir), "model.joblib"))


def predict_population(
    population: Sequence[Any],
    model_dir: str,
    *,
    score_fn: Any = None,
) -> Tuple[List[Dict[str, Any]], SurrogateModel]:
    """
    Run surrogate on each candidate; return list of JSON rows + loaded model.
    """
    model = SurrogateModel.load(model_dir)
    rows: List[Dict[str, Any]] = []
    for cand in population:
        s1 = None
        if score_fn is not None:
            try:
                s1 = score_fn(
                    cand.as_reward_function(),
                    candidate_id=getattr(cand, "candidate_id", ""),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Score1 for surrogate features failed (%s): %s",
                    getattr(cand, "candidate_id", "?"),
                    exc,
                )
        feats = extract_candidate_features(cand, score1_result=s1)
        pred = model.predict(feats)
        quality = float(quality_from_y_hat(pred.y_hat))
        rows.append(
            {
                "candidate_id": getattr(cand, "candidate_id", ""),
                "y_hat": dict(pred.y_hat),
                "uncertainty": float(pred.uncertainty),
                "quality": float(quality),
                "score1": (
                    None
                    if s1 is None
                    else float(getattr(s1, "score", s1) or float("nan"))
                ),
                "model_id": pred.model_id,
            }
        )
        md = dict(getattr(cand, "metadata", None) or {})
        md["surrogate_y_hat"] = dict(pred.y_hat)
        md["surrogate_uncertainty"] = float(pred.uncertainty)
        md["surrogate_quality"] = float(quality)
        cand.metadata = md
    return rows, model


def gate_population(
    population: Sequence[Any],
    predictions: Sequence[Dict[str, Any]],
    *,
    drop_fraction: float = 0.25,
    max_uncertainty_to_drop: float = 0.15,
    min_keep: int = 2,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Drop confident weak candidates before Stage III.

    Only candidates with ``uncertainty <= max_uncertainty_to_drop`` are eligible
    to be dropped. Among those, drop the worst by predicted ``quality`` until
    ``drop_fraction`` of the full population is removed (or no more eligible),
    always keeping at least ``min_keep`` survivors.
    """
    pop = list(population)
    if not pop:
        return [], {"enabled": True, "kept": [], "dropped": [], "n_kept": 0, "n_dropped": 0}

    by_id = {str(r.get("candidate_id")): r for r in predictions}
    scored = []
    for c in pop:
        cid = str(c.candidate_id)
        row = by_id.get(cid) or {}
        quality = float(row.get("quality", float("-inf")))
        uncertainty = float(row.get("uncertainty", float("inf")))
        scored.append((c, quality, uncertainty))

    n = len(scored)
    target_drop = int(max(0, round(n * float(drop_fraction))))
    max_drop = max(0, n - max(1, int(min_keep)))
    target_drop = min(target_drop, max_drop)

    # Eligible: confident (low uncertainty), sorted worst quality first.
    eligible = sorted(
        [
            (c, q, u)
            for c, q, u in scored
            if u <= float(max_uncertainty_to_drop)
        ],
        key=lambda t: t[1],
    )
    drop_ids = {c.candidate_id for c, _q, _u in eligible[:target_drop]}

    kept = [c for c, _q, _u in scored if c.candidate_id not in drop_ids]
    dropped = [c for c, _q, _u in scored if c.candidate_id in drop_ids]

    # Safety: never empty.
    if not kept and pop:
        kept = [scored[0][0]]
        dropped = [c for c in pop if c.candidate_id != kept[0].candidate_id]

    report = {
        "enabled": True,
        "drop_fraction": float(drop_fraction),
        "max_uncertainty_to_drop": float(max_uncertainty_to_drop),
        "min_keep": int(min_keep),
        "n_population": n,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "kept_ids": [c.candidate_id for c in kept],
        "dropped_ids": [c.candidate_id for c in dropped],
        "dropped": [
            {
                "candidate_id": c.candidate_id,
                "quality": by_id.get(c.candidate_id, {}).get("quality"),
                "uncertainty": by_id.get(c.candidate_id, {}).get("uncertainty"),
            }
            for c in dropped
        ],
    }
    return kept, report

"""
Closed loop: score → enqueue → acquire → optional surrogate re-fit.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from crowd_nav.reward_search.active_learning import acquire as acquire_mod
from crowd_nav.reward_search.active_learning import queue as queue_mod
from crowd_nav.reward_search.active_learning.query import score_queries
from crowd_nav.reward_search.surrogate.features import extract_candidate_features
from crowd_nav.reward_search.surrogate.model import SurrogateModel

logger = logging.getLogger(__name__)


class SurrogateModelRequired(RuntimeError):
    """Raised / signaled when AL runs without a fitted surrogate on disk."""


def _model_ready(surrogate_model_dir: str) -> bool:
    return os.path.isfile(os.path.join(surrogate_model_dir, "model.joblib"))


def load_candidates_from_json(path: str) -> List[Any]:
    from crowd_nav.reward_search.reporting import load_candidate_dict

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    rows: List[Dict[str, Any]]
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = list(data.get("population") or [])
    else:
        raise ValueError(f"unsupported candidates file: {path}")
    cands = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cand = load_candidate_dict(row)
        if cand.valid and cand.reward_fn is not None:
            cands.append(cand)
    return cands


def _predict_batch(
    model: SurrogateModel,
    candidates: Sequence[Any],
    *,
    score_fn: Any = None,
) -> tuple[List[Any], List[Any], List[Dict[str, Any]]]:
    preds = []
    score1_results = []
    features_list = []
    for cand in candidates:
        s1 = None
        if score_fn is not None:
            try:
                s1 = score_fn(cand.as_reward_function(), candidate_id=cand.candidate_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Score1 failed for %s: %s", cand.candidate_id, exc)
        feats = extract_candidate_features(cand, score1_result=s1)
        features_list.append(feats)
        preds.append(model.predict(feats))
        score1_results.append(s1)
    return preds, score1_results, features_list


def _refit_surrogate(
    *,
    surrogate_dataset: str,
    surrogate_model_dir: str,
    seed: int,
) -> Dict[str, Any]:
    from crowd_nav.reward_search.surrogate.dataset_io import load_table

    feats, labs = load_table(surrogate_dataset)
    if len(feats) < 1:
        return {"status": "skipped", "reason": "empty_dataset"}
    model = SurrogateModel(random_seed=int(seed))
    metrics = model.fit(feats, labs, target_keys=("SR", "CR", "TR"))
    os.makedirs(surrogate_model_dir, exist_ok=True)
    model.save(surrogate_model_dir)
    metrics_path = os.path.join(surrogate_model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")
    return {"status": "ok", "n": len(feats), "metrics": metrics}


def run_active_learning_step(
    *,
    surrogate_model_dir: str = "artifacts/surrogate",
    queue_root: str = "data/active_learning",
    max_queries: int = 5,
    refit_every: int = 20,
    force_refit: bool = False,
    candidates: Optional[Sequence[Any]] = None,
    candidates_path: Optional[str] = None,
    surrogate_dataset: str = "data/surrogate_dataset",
    use_stub: bool = False,
    seed: int = 425,
    promote_threshold: float = 0.0,
    stage1_dataset_path: str = "data/stage1_dataset",
    enqueue_only: bool = False,
) -> Dict[str, Any]:
    """
    One AL iteration. See PLAN.md §5.

    Returns summary counts for logging / manifest.
    """
    t0 = time.perf_counter()
    if not _model_ready(surrogate_model_dir):
        summary = {
            "status": "error",
            "message": "Surrogate model required",
            "surrogate_model_dir": surrogate_model_dir,
        }
        queue_mod.append_step(queue_root, summary)
        return summary

    model = SurrogateModel.load(surrogate_model_dir)

    batch: List[Any] = list(candidates or [])
    if not batch and candidates_path:
        batch = load_candidates_from_json(candidates_path)

    score_fn = None
    if batch:
        from crowd_nav.domains import load_domain

        pack = load_domain("crowdnav")
        mode = "smoke" if use_stub else "dataset"
        try:
            score_fn, _ds = pack.make_score_fn(
                mode=mode,
                dataset_path=None if mode == "smoke" else stage1_dataset_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Score1 fn unavailable (%s); continuing without", exc)
            if use_stub:
                from crowd_nav.reward_search.scoring import make_smoke_score_fn

                score_fn = make_smoke_score_fn()

    n_enqueued = 0
    if batch:
        preds, score1_results, _feats = _predict_batch(model, batch, score_fn=score_fn)
        # Attach fingerprints onto candidate metadata for diversity.
        for cand, feat in zip(batch, _feats):
            md = dict(cand.metadata or {})
            md["behavior_fingerprint"] = feat.get("behavior_fingerprint") or []
            cand.metadata = md
        from crowd_nav.reward_search.surrogate.dataset_io import existing_example_ids

        items = score_queries(
            list(batch),
            surrogate_preds=preds,
            score1_results=score1_results,
            top_k=int(max_queries),
            promote_threshold=float(promote_threshold),
            reference_fingerprints=queue_mod.recent_fingerprints(queue_root),
            already_labeled_ids=list(existing_example_ids(surrogate_dataset)),
        )
        queue_mod.enqueue(queue_root, items)
        n_enqueued = len(items)

    if enqueue_only:
        summary = {
            "status": "ok",
            "n_candidates": len(batch),
            "n_enqueued": n_enqueued,
            "n_executed": 0,
            "refit": False,
            "wall_seconds": time.perf_counter() - t0,
        }
        queue_mod.append_step(queue_root, summary)
        return summary

    taken = queue_mod.dequeue_batch(queue_root, limit=int(max_queries))
    acquire_cfg = {
        "use_stub": bool(use_stub),
        "surrogate_dataset": surrogate_dataset,
        "surrogate_model_dir": surrogate_model_dir,
        "queue_root": queue_root,
        "stage1_dataset_path": stage1_dataset_path,
        "seed": int(seed),
        "stage1_extra_dir": os.path.join(queue_root, "stage1_extra"),
    }

    executed = []
    n_ok = 0
    for item in taken:
        result = acquire_mod.execute_query(item, config=acquire_cfg)
        queue_mod.mark_done(queue_root, item, result=result)
        executed.append({"kind": item.kind, "result": result})
        if result.get("status") == "ok":
            n_ok += 1

    man = queue_mod.read_manifest(queue_root)
    n_since = int(man.get("n_done_since_refit") or 0)
    did_refit = False
    refit_info: Dict[str, Any] = {}
    if force_refit or (refit_every > 0 and n_since >= int(refit_every)):
        refit_info = _refit_surrogate(
            surrogate_dataset=surrogate_dataset,
            surrogate_model_dir=surrogate_model_dir,
            seed=int(seed),
        )
        did_refit = refit_info.get("status") == "ok"
        if did_refit:
            queue_mod.reset_refit_counter(queue_root)

    # Persist a small AL config snapshot next to the surrogate.
    try:
        al_cfg_path = os.path.join(surrogate_model_dir, "active_learning_config.json")
        with open(al_cfg_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "max_queries": int(max_queries),
                    "refit_every": int(refit_every),
                    "promote_threshold": float(promote_threshold),
                    "use_stub": bool(use_stub),
                    "surrogate_dataset": surrogate_dataset,
                    "queue_root": queue_root,
                },
                fh,
                indent=2,
            )
            fh.write("\n")
    except OSError as exc:
        logger.warning("could not write AL config snapshot: %s", exc)

    summary = {
        "status": "ok",
        "n_candidates": len(batch),
        "n_enqueued": n_enqueued,
        "n_executed": len(executed),
        "n_ok": n_ok,
        "refit": did_refit,
        "refit_info": refit_info,
        "n_done_since_refit": int(queue_mod.read_manifest(queue_root).get("n_done_since_refit") or 0),
        "wall_seconds": time.perf_counter() - t0,
        "executed": executed,
    }
    queue_mod.append_step(queue_root, {k: v for k, v in summary.items() if k != "executed"})
    return summary

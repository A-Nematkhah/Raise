"""
Execute query items: Stage II short labels or Stage I side-pool requests.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from crowd_nav.reward_search.active_learning.query import QueryItem

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _candidate_from_payload(payload: Dict[str, Any]):
    from crowd_nav.reward_search.reporting import load_candidate_dict

    code = payload.get("code")
    if not code:
        raise ValueError("stage2_label payload missing code")
    return load_candidate_dict(
        {
            "candidate_id": payload.get("candidate_id") or "al_candidate",
            "code": code,
            "score": payload.get("score1"),
            "origin": "active_learning",
            "metadata": {
                "behavior_fingerprint": payload.get("behavior_fingerprint") or [],
            },
        }
    )


def _execute_stage2_label(item: QueryItem, *, config: Dict[str, Any]) -> Dict[str, Any]:
    from crowd_nav.domains import load_domain, make_stage2_trainer_for_domain
    from crowd_nav.reward_search.stage2 import Stage2Config
    from crowd_nav.reward_search.surrogate.bootstrap import label_and_append_candidate
    from crowd_nav.reward_search.surrogate.dataset_io import existing_example_ids

    use_stub = bool(config.get("use_stub", False))
    out_dir = str(config.get("surrogate_dataset") or "data/surrogate_dataset")
    stage1_dataset = config.get("stage1_dataset_path") or "data/stage1_dataset"
    score1_mode = "smoke" if use_stub else str(config.get("score1_mode") or "dataset")

    pack = load_domain("crowdnav")
    score_fn, _ds = pack.make_score_fn(
        mode=score1_mode,
        dataset_path=None if score1_mode == "smoke" else stage1_dataset,
    )
    trainer = make_stage2_trainer_for_domain(pack, use_stub=use_stub)

    train_steps = int(config.get("stage2_train_steps") or (64 if use_stub else 8000))
    if use_stub:
        train_steps = min(train_steps, 64)
    stage2_cfg = Stage2Config(
        train_env_steps=train_steps,
        k2_unit=str(config.get("k2_unit") or "gradient_steps"),
        eval_episodes=8 if use_stub else int(config.get("eval_episodes") or 50),
        horizon_steps=20 if use_stub else int(config.get("horizon_steps") or 100),
        seed=int(config.get("seed") or 425),
        device=str(config.get("device") or "cpu"),
        output_root=str(
            config.get("stage2_output_root")
            or os.path.join(str(config.get("surrogate_model_dir") or "artifacts/surrogate"), "_al_stage2")
        ),
        num_processes=1 if use_stub else None,
    )

    candidate = _candidate_from_payload(item.payload)
    if not candidate.valid or candidate.reward_fn is None:
        return {
            "status": "failed",
            "kind": "stage2_label",
            "error": candidate.validation_error or "invalid_reward_code",
        }

    known = existing_example_ids(out_dir)
    result = label_and_append_candidate(
        candidate,
        score_fn=score_fn,
        trainer=trainer,
        stage2_cfg=stage2_cfg,
        out_dir=out_dir,
        round_index=int(config.get("round_index") or 0),
        known_ids=known,
        label_rejected=bool(config.get("label_rejected", True)),
        use_stub=use_stub,
    )
    result["kind"] = "stage2_label"
    return result


def _execute_stage1_scenario(item: QueryItem, *, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe path (PLAN option A): record scenario requests under stage1_extra/
    without mutating the locked paper Stage I dataset.
    """
    queue_root = str(config.get("queue_root") or "data/active_learning")
    extra_dir = str(
        config.get("stage1_extra_dir")
        or os.path.join(queue_root, "stage1_extra")
    )
    os.makedirs(extra_dir, exist_ok=True)
    scenario_ids = item.payload.get("scenario_ids") or []
    if not isinstance(scenario_ids, list):
        scenario_ids = [scenario_ids]
    scenario_ids = [str(s) for s in scenario_ids if str(s).strip()]
    if not scenario_ids:
        return {
            "status": "skipped",
            "kind": "stage1_scenario",
            "reason": "empty_scenario_ids",
        }

    request = {
        "ts": _utc_now(),
        "scenario_ids": scenario_ids,
        "reason": item.reason,
        "status": "pending_collection",
        "note": "Side-pool request only; paper stage1_dataset untouched.",
    }
    path = os.path.join(extra_dir, "pending_requests.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(request, ensure_ascii=False))
        fh.write("\n")
    manifest_path = os.path.join(extra_dir, "manifest.json")
    man: Dict[str, Any] = {"n_requests": 0, "scenario_ids": []}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                man = loaded
    ids = list(man.get("scenario_ids") or [])
    for sid in scenario_ids:
        if sid not in ids:
            ids.append(sid)
    man["scenario_ids"] = ids
    man["n_requests"] = int(man.get("n_requests") or 0) + 1
    man["updated_at"] = _utc_now()
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
        fh.write("\n")
    return {
        "status": "ok",
        "kind": "stage1_scenario",
        "path": path,
        "scenario_ids": scenario_ids,
    }


def execute_query(item: QueryItem, *, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run one acquisition action and return a result dict for ``mark_done``.

    kind=stage1_scenario → side-pool request under stage1_extra/
    kind=stage2_label    → append surrogate_dataset example
    """
    cfg = dict(config or {})
    kind = str(item.kind).strip().lower()
    if kind == "stage2_label":
        return _execute_stage2_label(item, config=cfg)
    if kind == "stage1_scenario":
        return _execute_stage1_scenario(item, config=cfg)
    if kind == "stage3_label":
        return {"status": "skipped", "kind": kind, "reason": "not_in_v1"}
    return {"status": "failed", "kind": kind, "error": f"unknown_kind:{kind}"}

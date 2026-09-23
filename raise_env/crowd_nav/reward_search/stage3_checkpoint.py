"""
Persist / restore Stage III (Validate) run state.

Crash-safe resume at two granularities:
  1. After each finished candidate (round, refine, history).
  2. Mid-PPO via ``train_progress`` (update index + weights path).

Layout under the run output dir::

    {output_dir}/stage3/checkpoint.json
    {output_dir}/stage3/RESUME.json
    {output_dir}/stage3_train/rXX_{id}/checkpoints/{update:05d}.pt
    {output_dir}/stage3_train/rXX_{id}/train_progress.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from crowd_nav.reward_search.explore import RewardCandidate
from crowd_nav.reward_search.reporting import candidate_to_dict, load_candidate_dict

_CHECKPOINT_NAME = "checkpoint.json"
_RESUME_NAME = "RESUME.json"
_TRAIN_PROGRESS_NAME = "train_progress.json"
_SCHEMA_VERSION = "1"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def stage3_checkpoint_dir(output_dir: str) -> str:
    """``{run}/stage3`` next to ``stage3_train``."""
    return os.path.join(os.path.abspath(output_dir), "stage3")


def stage3_dir_from_train_root(output_root: str) -> str:
    """If ``output_root`` is ``.../stage3_train``, return sibling ``.../stage3``."""
    root = os.path.abspath(output_root)
    parent, base = os.path.split(root.rstrip("/\\"))
    if base == "stage3_train" and parent:
        return os.path.join(parent, "stage3")
    return os.path.join(root, "stage3")


def _serialize_pop(population: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    if not population:
        return []
    out: List[Dict[str, Any]] = []
    for cand in population:
        if hasattr(cand, "candidate_id"):
            out.append(candidate_to_dict(cand))
        elif isinstance(cand, dict):
            out.append(dict(cand))
    return out


def deserialize_population(rows: Optional[Sequence[Dict[str, Any]]]) -> List[RewardCandidate]:
    if not rows:
        return []
    return [load_candidate_dict(dict(row)) for row in rows if isinstance(row, dict)]


def build_checkpoint(
    *,
    status: str,
    phase: str,
    round_index: int,
    candidate_index: int,
    rounds: int,
    round_input_population: Optional[Sequence[Any]] = None,
    round_output_partial: Optional[Sequence[Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    best_trained: Any = None,
    trained_snapshots: Optional[Sequence[Any]] = None,
    run_h_sweep: bool = True,
    train_progress: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "status": str(status),
        "phase": str(phase),
        "round_index": int(round_index),
        "candidate_index": int(candidate_index),
        "rounds": int(rounds),
        "round_input_population": _serialize_pop(round_input_population),
        "round_output_partial": _serialize_pop(round_output_partial),
        "history": list(history or []),
        "best_trained": (
            candidate_to_dict(best_trained) if best_trained is not None else None
        ),
        "trained_snapshots": _serialize_pop(trained_snapshots),
        "run_h_sweep": bool(run_h_sweep),
        "train_progress": dict(train_progress) if train_progress else None,
        "config": dict(config or {}),
        "updated_at": _utc_now(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def save_checkpoint(run_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, _CHECKPOINT_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    resume = {
        "checkpoint": _CHECKPOINT_NAME,
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "round_index": payload.get("round_index"),
        "candidate_index": payload.get("candidate_index"),
        "updated_at": payload.get("updated_at") or _utc_now(),
    }
    with open(os.path.join(run_dir, _RESUME_NAME), "w", encoding="utf-8") as fh:
        json.dump(resume, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def load_checkpoint(run_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(run_dir, _CHECKPOINT_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def history_record_to_dict(
    *,
    round_index: int,
    candidate_id: str,
    metrics: Dict[str, Any],
    refined: bool,
    kept_previous: bool,
    checkpoint_path: Optional[str] = None,
    validation_error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "round_index": int(round_index),
        "candidate_id": str(candidate_id),
        "metrics": dict(metrics),
        "refined": bool(refined),
        "kept_previous": bool(kept_previous),
        "checkpoint_path": checkpoint_path,
        "validation_error": validation_error,
    }


def save_train_progress(
    out_dir: str,
    *,
    round_index: int,
    candidate_id: str,
    update_j: int,
    num_updates: int,
    weights_path: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _TRAIN_PROGRESS_NAME)
    payload = {
        "round_index": int(round_index),
        "candidate_id": str(candidate_id),
        "update_j": int(update_j),
        "num_updates": int(num_updates),
        "weights_path": str(weights_path),
        "updated_at": _utc_now(),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def load_train_progress(out_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(out_dir, _TRAIN_PROGRESS_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def clear_train_progress(out_dir: str) -> None:
    path = os.path.join(out_dir, _TRAIN_PROGRESS_NAME)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass

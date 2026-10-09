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

import os
from typing import Any, Dict, Optional, Sequence

from raise_core.json_io import atomic_write_json, read_json_dict, utc_now_z
from raise_core.population_io import (  # noqa: F401 — deserialize_population is re-exported
    deserialize_population,
    serialize_population,
)
from domains.crowdnav.reporting import candidate_to_dict

_CHECKPOINT_NAME = "checkpoint.json"
_RESUME_NAME = "RESUME.json"
_TRAIN_PROGRESS_NAME = "train_progress.json"
_SCHEMA_VERSION = "1"


def stage3_dir_from_train_root(output_root: str) -> str:
    """If ``output_root`` is ``.../stage3_train``, return sibling ``.../stage3``."""
    root = os.path.abspath(output_root)
    parent, base = os.path.split(root.rstrip("/\\"))
    if base == "stage3_train" and parent:
        return os.path.join(parent, "stage3")
    return os.path.join(root, "stage3")


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
        "round_input_population": serialize_population(round_input_population),
        "round_output_partial": serialize_population(round_output_partial),
        "history": list(history or []),
        "best_trained": (
            candidate_to_dict(best_trained) if best_trained is not None else None
        ),
        "trained_snapshots": serialize_population(trained_snapshots),
        "run_h_sweep": bool(run_h_sweep),
        "train_progress": dict(train_progress) if train_progress else None,
        "config": dict(config or {}),
        "updated_at": utc_now_z(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def save_checkpoint(run_dir: str, payload: Dict[str, Any]) -> str:
    path = os.path.join(run_dir, _CHECKPOINT_NAME)
    atomic_write_json(path, payload)
    resume = {
        "checkpoint": _CHECKPOINT_NAME,
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "round_index": payload.get("round_index"),
        "candidate_index": payload.get("candidate_index"),
        "updated_at": payload.get("updated_at") or utc_now_z(),
    }
    atomic_write_json(os.path.join(run_dir, _RESUME_NAME), resume)
    return path


def load_checkpoint(run_dir: str) -> Optional[Dict[str, Any]]:
    return read_json_dict(os.path.join(run_dir, _CHECKPOINT_NAME))


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
    path = os.path.join(out_dir, _TRAIN_PROGRESS_NAME)
    payload = {
        "round_index": int(round_index),
        "candidate_id": str(candidate_id),
        "update_j": int(update_j),
        "num_updates": int(num_updates),
        "weights_path": str(weights_path),
        "updated_at": utc_now_z(),
    }
    return atomic_write_json(path, payload)


def load_train_progress(out_dir: str) -> Optional[Dict[str, Any]]:
    return read_json_dict(os.path.join(out_dir, _TRAIN_PROGRESS_NAME))


def clear_train_progress(out_dir: str) -> None:
    path = os.path.join(out_dir, _TRAIN_PROGRESS_NAME)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass

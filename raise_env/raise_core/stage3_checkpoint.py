"""
Persist / restore Stage III run state (resume after each trained candidate).

Layout under the run output dir::

    {output_dir}/stage3/checkpoint.json
    {output_dir}/stage3/RESUME.json
    {output_dir}/stage3_train/r00_{id}/model.zip
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from raise_core.json_io import atomic_write_json, read_json_dict, utc_now_z
from raise_core.population_io import candidate_to_dict, serialize_population

_CHECKPOINT_NAME = "checkpoint.json"
_RESUME_NAME = "RESUME.json"
_SCHEMA_VERSION = "2"


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
    next_index: int,
    population: Optional[Sequence[Any]] = None,
    trained: Optional[Sequence[Any]] = None,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    best_trained: Any = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "status": str(status),
        "next_index": int(next_index),
        "population": serialize_population(population),
        "trained": serialize_population(trained),
        "history": list(history or []),
        "best_trained": (
            candidate_to_dict(best_trained) if best_trained is not None else None
        ),
        "config": dict(config or {}),
        "updated_at": utc_now_z(),
    }


def save_checkpoint(run_dir: str, payload: Dict[str, Any]) -> str:
    path = os.path.join(run_dir, _CHECKPOINT_NAME)
    atomic_write_json(path, payload)
    resume = {
        "checkpoint": _CHECKPOINT_NAME,
        "status": payload.get("status"),
        "next_index": payload.get("next_index"),
        "updated_at": payload.get("updated_at") or utc_now_z(),
    }
    atomic_write_json(os.path.join(run_dir, _RESUME_NAME), resume)
    return path


def load_checkpoint(run_dir: str) -> Optional[Dict[str, Any]]:
    return read_json_dict(os.path.join(run_dir, _CHECKPOINT_NAME))

"""
Persist / restore RAISE closed-loop run state.

Used for crash-safe resume (``RESUME.json`` + ``checkpoint.json`` under the run dir).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from raise_core.json_io import atomic_write_json, read_json_dict, utc_now_z
from raise_core.population_io import deserialize_population, serialize_population

_CHECKPOINT_NAME = "checkpoint.json"
_RESUME_NAME = "RESUME.json"


def build_checkpoint(
    *,
    status: str,
    phase: str,
    epoch: int,
    next_epoch: int,
    population: Optional[Sequence[Any]] = None,
    to_label: Optional[Sequence[Any]] = None,
    labeled_ids_this_epoch: Optional[Sequence[str]] = None,
    reflection: str = "",
    history: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable checkpoint payload."""
    payload: Dict[str, Any] = {
        "schema_version": "2",
        "status": str(status),
        "phase": str(phase),
        "epoch": int(epoch),
        "next_epoch": int(next_epoch),
        "population": serialize_population(population),
        "to_label": serialize_population(to_label),
        "labeled_ids_this_epoch": [str(x) for x in (labeled_ids_this_epoch or [])],
        "reflection": str(reflection or ""),
        "history": list(history or []),
        "config": dict(config or {}),
        "updated_at": utc_now_z(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def save_checkpoint(run_dir: str, payload: Dict[str, Any]) -> str:
    """Write ``checkpoint.json`` and a small ``RESUME.json`` pointer; return ckpt path."""
    path = os.path.join(run_dir, _CHECKPOINT_NAME)
    atomic_write_json(path, payload)
    resume = {
        "checkpoint": _CHECKPOINT_NAME,
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "epoch": payload.get("epoch"),
        "updated_at": payload.get("updated_at") or utc_now_z(),
    }
    atomic_write_json(os.path.join(run_dir, _RESUME_NAME), resume)
    return path


def load_checkpoint(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint from ``run_dir`` if present."""
    return read_json_dict(os.path.join(run_dir, _CHECKPOINT_NAME))


__all__ = [
    "build_checkpoint",
    "deserialize_population",
    "load_checkpoint",
    "save_checkpoint",
]

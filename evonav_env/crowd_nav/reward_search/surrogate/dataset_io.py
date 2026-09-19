"""
Surrogate dataset I/O: jsonl features/labels + manifest (stub).

Default root: ``evonav_env/data/surrogate_dataset/``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def default_dataset_root() -> str:
    return "data/surrogate_dataset"


def append_example(
    root: str,
    *,
    features: Dict[str, Any],
    labels: Dict[str, Any],
    example_id: str,
) -> None:
    raise NotImplementedError("surrogate.dataset_io.append_example — see PLAN.md")


def load_table(root: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raise NotImplementedError("surrogate.dataset_io.load_table — see PLAN.md")


def write_manifest(root: str, payload: Dict[str, Any]) -> None:
    raise NotImplementedError("surrogate.dataset_io.write_manifest — see PLAN.md")


def read_manifest(root: str) -> Optional[Dict[str, Any]]:
    raise NotImplementedError("surrogate.dataset_io.read_manifest — see PLAN.md")

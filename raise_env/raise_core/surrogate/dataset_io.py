"""
Surrogate dataset I/O: jsonl features/labels + manifest.

Default root: ``raise_env/data/surrogate_dataset/``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

LABEL_SCHEMA_VERSION = "1"

_FEATURES_NAME = "features.jsonl"
_LABELS_NAME = "labels.jsonl"
_MANIFEST_NAME = "manifest.json"


def default_dataset_root() -> str:
    return "data/surrogate_dataset"


def _ensure_dir(root: str) -> None:
    os.makedirs(root, exist_ok=True)


def _features_path(root: str) -> str:
    return os.path.join(root, _FEATURES_NAME)


def _labels_path(root: str) -> str:
    return os.path.join(root, _LABELS_NAME)


def _manifest_path(root: str) -> str:
    return os.path.join(root, _MANIFEST_NAME)


def _append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        fh.write("\n")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def existing_example_ids(root: str) -> Set[str]:
    """Return ``example_id`` values already present in ``features.jsonl``."""
    ids: Set[str] = set()
    for row in _load_jsonl(_features_path(root)):
        eid = row.get("example_id")
        if eid is not None:
            ids.add(str(eid))
    return ids


def append_example(
    root: str,
    *,
    features: Dict[str, Any],
    labels: Dict[str, Any],
    example_id: str,
) -> None:
    """
    Append one aligned (features, labels) pair.

    Both rows receive ``example_id``. Idempotent callers should skip ids in
    :func:`existing_example_ids` before calling.
    """
    if not example_id or not str(example_id).strip():
        raise ValueError("example_id must be a non-empty string")
    eid = str(example_id).strip()
    _ensure_dir(root)
    feat = dict(features)
    lab = dict(labels)
    feat["example_id"] = eid
    lab["example_id"] = eid
    _append_jsonl(_features_path(root), feat)
    _append_jsonl(_labels_path(root), lab)


def load_table(root: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load features/labels aligned by ``example_id``.

    Rows present in only one file are dropped (with no silent reordering of
    mismatched streams). Order follows ``features.jsonl``.
    """
    feats = _load_jsonl(_features_path(root))
    labs = _load_jsonl(_labels_path(root))
    by_id = {
        str(row["example_id"]): row
        for row in labs
        if row.get("example_id") is not None
    }
    aligned_f: List[Dict[str, Any]] = []
    aligned_l: List[Dict[str, Any]] = []
    for feat in feats:
        eid = feat.get("example_id")
        if eid is None:
            continue
        lab = by_id.get(str(eid))
        if lab is None:
            continue
        aligned_f.append(feat)
        aligned_l.append(lab)
    return aligned_f, aligned_l


def write_manifest(root: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(root)
    path = _manifest_path(root)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")


def read_manifest(root: str) -> Optional[Dict[str, Any]]:
    path = _manifest_path(root)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be an object: {path}")
    return data

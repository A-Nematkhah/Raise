"""
Surrogate dataset I/O: jsonl features/labels + manifest.

Default root: ``raise_env/domains/crowdnav/data/surrogate_dataset/``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

LABEL_SCHEMA_VERSION = "1"

_FEATURES_NAME = "features.jsonl"
_LABELS_NAME = "labels.jsonl"
_MANIFEST_NAME = "manifest.json"
_APPEND_LOCK = threading.RLock()


def _ensure_dir(root: str) -> None:
    os.makedirs(root, exist_ok=True)


def _features_path(root: str) -> str:
    return os.path.join(root, _FEATURES_NAME)


def _labels_path(root: str) -> str:
    return os.path.join(root, _LABELS_NAME)


def _manifest_path(root: str) -> str:
    return os.path.join(root, _MANIFEST_NAME)


def _json_safe(value: Any) -> Any:
    """Non-finite floats → None (readers already treat None as missing)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _jsonl_line(payload: Dict[str, Any]) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n"


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
                # Partial row from a crash mid-append; its example has no
                # complete pair, so it is simply relabeled.
                logger.warning("Skipping unreadable JSONL row %s:%d (%s)", path, line_no, exc)
                continue
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def _append_line(path: str, line: str) -> None:
    # A crash can leave the file without a trailing newline; start a fresh
    # line so the new row is not glued onto the partial one.
    needs_newline = False
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            needs_newline = fh.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        fh.write(line)


def _ids_in(path: str) -> Set[str]:
    return {
        str(row["example_id"])
        for row in _load_jsonl(path)
        if row.get("example_id") is not None
    }


def existing_example_ids(root: str) -> Set[str]:
    """Return ``example_id`` values with both a features and a labels row.

    A features row whose labels row never landed (crash between the two
    appends) does not count, so that example is relabeled.
    """
    return _ids_in(_features_path(root)) & _ids_in(_labels_path(root))


def claim_example_id(known: Set[str], example_id: str) -> bool:
    """
    Atomically claim ``example_id`` in ``known`` (thread-safe).

    Returns True if this caller owns the id (first claim); False if already seen.
    """
    eid = str(example_id).strip()
    with _APPEND_LOCK:
        if eid in known:
            return False
        known.add(eid)
        return True


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
    # Serialize both before touching disk so a bad value cannot orphan a row.
    feat_line = _jsonl_line(feat)
    lab_line = _jsonl_line(lab)
    with _APPEND_LOCK:
        _append_line(_features_path(root), feat_line)
        _append_line(_labels_path(root), lab_line)


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
    seen: Set[str] = set()
    for feat in feats:
        eid = feat.get("example_id")
        if eid is None or str(eid) in seen:
            continue
        lab = by_id.get(str(eid))
        if lab is None:
            continue
        seen.add(str(eid))
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

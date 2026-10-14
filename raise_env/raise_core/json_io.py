"""Crash-safe JSON writes shared by run-state checkpoints."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

_REPLACE_RETRIES = 5
_REPLACE_BACKOFF_S = 0.05


def utc_now_z() -> str:
    """UTC timestamp like ``2026-10-09T12:34:56Z`` (seconds precision)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_json_dict(path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON object from ``path``; None if missing or not a dict."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def atomic_write_json(
    path: str,
    payload: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
    default: Optional[Callable[[Any], Any]] = None,
) -> str:
    """Write ``payload`` to ``path`` via temp file + ``os.replace``.

    A crash mid-write leaves the previous file intact instead of truncated JSON.
    """
    text = json.dumps(
        payload,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        default=default,
    )
    return atomic_write_text(path, text + "\n")


def atomic_write_text(path: str, text: str) -> str:
    """Replace ``path`` with ``text`` atomically (temp file + fsync + ``os.replace``)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        # Windows: os.replace raises PermissionError while another process
        # (viewer, antivirus) briefly holds the target open.
        for attempt in range(_REPLACE_RETRIES):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == _REPLACE_RETRIES - 1:
                    raise
                time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


__all__ = ["atomic_write_json", "atomic_write_text", "read_json_dict", "utc_now_z"]

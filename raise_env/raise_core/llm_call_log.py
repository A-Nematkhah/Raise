"""
Per-request LLM call log: ``llm_calls.jsonl`` + rendered prompt snapshots.

Request records carry call type, subject candidates, card sections, history
ids, token counts and truncation steps; validation records link each
resulting candidate back to its ``call_id``. Prompts hold no credentials
(keys live only inside the provider clients), so snapshots are written verbatim.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

CALLS_FILE = "llm_calls.jsonl"
PROMPTS_DIR = "llm_prompts"


class LLMCallLog:
    def __init__(self, directory: Optional[str]) -> None:
        self.directory = directory
        self._n = 0

    def next_call_id(self, call_type: str) -> str:
        self._n += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{stamp}_{self._n:04d}_{call_type}"

    def _append(self, record: Dict[str, Any]) -> None:
        if not self.directory:
            return
        os.makedirs(self.directory, exist_ok=True)
        with open(os.path.join(self.directory, CALLS_FILE), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def write_request(self, record: Dict[str, Any], messages: Sequence[str]) -> Optional[str]:
        """Log one request; returns the prompt snapshot path (None when not persisted)."""
        if not self.directory:
            return None
        snap_dir = os.path.join(self.directory, PROMPTS_DIR)
        os.makedirs(snap_dir, exist_ok=True)
        snap = os.path.join(snap_dir, f"{record['call_id']}.txt")
        roles = ("client_system", "user") if len(messages) == 2 else tuple(
            f"message_{i}" for i in range(len(messages))
        )
        with open(snap, "w", encoding="utf-8") as fh:
            for role, text in zip(roles, messages):
                fh.write(f"===== {role} =====\n{text}\n")
        self._append(
            {
                "event": "request",
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                **record,
                "prompt_snapshot": os.path.relpath(snap, self.directory),
            }
        )
        return snap

    def write_validation(
        self,
        *,
        call_id: str,
        candidate_id: str,
        accepted: bool,
        validation_error: Optional[str],
    ) -> None:
        self._append(
            {
                "event": "validation",
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "call_id": call_id,
                "candidate_id": candidate_id,
                "validator_passed": bool(accepted),
                "validation_error": (validation_error or None) and str(validation_error)[:300],
            }
        )


__all__ = ["CALLS_FILE", "PROMPTS_DIR", "LLMCallLog"]

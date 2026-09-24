"""
DS-RNN Table 1 baseline helpers (Option A train paths / Option B skip).

Option A: PPO-trained ``srnn`` checkpoints under
  ``trained_models/ds_rnn_no_rand/`` and ``trained_models/ds_rnn_rand/``.
Option B: explicit ``not reproduced locally — see paper Table 1`` row —
  never fabricate or interpolate metric values.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

# Canonical message for Option B (must stay stable for tests / JSON consumers).
DSRNN_NOT_REPRODUCED_MSG = "not reproduced locally — see paper Table 1"

DEFAULT_DSRNN_NO_RAND_DIR = "trained_models/ds_rnn_no_rand"
DEFAULT_DSRNN_RAND_DIR = "trained_models/ds_rnn_rand"


def dsrnn_placeholder(*, reason: Optional[str] = None) -> Dict[str, Any]:
    """
    Table 1 DS-RNN entry when checkpoints are skipped or absent.

    Contains no SR/CR/… numeric fields — callers must not invent baselines.
    """
    return {
        "method": "DS-RNN",
        "skipped": True,
        "not_reproduced": True,
        "reason": reason or DSRNN_NOT_REPRODUCED_MSG,
    }


def format_dsrnn_table1_row(
    entry: Dict[str, Any],
    *,
    randomize: bool,
) -> str:
    """One-line Table 1 printer for evaluated or placeholder DS-RNN rows."""
    if entry.get("skipped") or entry.get("not_reproduced"):
        reason = entry.get("reason") or DSRNN_NOT_REPRODUCED_MSG
        return f"{'DS-RNN':16s} rand={randomize}  [{reason}]"
    fmt = entry.get("format") or {}
    if fmt:
        return (
            f"{'DS-RNN':16s} rand={randomize}  "
            f"SR={fmt.get('SR', '?')} CR={fmt.get('CR', '?')} TR={fmt.get('TR', '?')} "
            f"NT={fmt.get('NT', '?')} PL={fmt.get('PL', '?')} "
            f"ITR={fmt.get('ITR', '?')} SD={fmt.get('SD', '?')}"
        )
    return f"{'DS-RNN':16s} rand={randomize}  (see JSON)"


def latest_checkpoint(model_dir: str) -> Optional[str]:
    """Return the lexicographically last ``*.pt`` under ``model_dir/checkpoints``."""
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    return pts[-1] if pts else None


def resolve_checkpoint(model_dir: str, ckpt: str) -> Optional[str]:
    """
    Resolve checkpoint filename.

    ``ckpt`` of ``auto`` / empty → latest; otherwise require the named file.
    """
    if not os.path.isdir(model_dir):
        return None
    key = (ckpt or "auto").strip().lower()
    if key in ("auto", "latest", ""):
        return latest_checkpoint(model_dir)
    path = os.path.join(model_dir, "checkpoints", ckpt)
    return ckpt if os.path.isfile(path) else None


def model_dir_ready(model_dir: str, ckpt: str = "auto") -> bool:
    return resolve_checkpoint(model_dir, ckpt) is not None


def resolve_dsrnn_dirs(
    *,
    skip_dsrnn: bool,
    ds_rnn_dir: Optional[str] = None,
    ds_rnn_no_rand_dir: Optional[str] = None,
    ds_rnn_rand_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, Optional[str]]]:
    """
    Decide Option A dirs vs Option B skip.

    Returns ``(mode, dirs)`` where mode is ``"skip"`` | ``"evaluate"`` and
    dirs maps ``no_rand`` / ``rand`` → path or None.
    """
    no_rand = ds_rnn_no_rand_dir or ds_rnn_dir or DEFAULT_DSRNN_NO_RAND_DIR
    rand = ds_rnn_rand_dir or DEFAULT_DSRNN_RAND_DIR
    # If only --ds-rnn-dir pointed at the rand tree, keep no_rand default.
    if ds_rnn_dir and os.path.basename(ds_rnn_dir.rstrip("/\\")) == "ds_rnn_rand":
        rand = ds_rnn_dir
        if ds_rnn_no_rand_dir is None:
            sibling = os.path.join(
                os.path.dirname(ds_rnn_dir.rstrip("/\\")), "ds_rnn_no_rand"
            )
            no_rand = sibling

    if skip_dsrnn:
        return "skip", {"no_rand": no_rand, "rand": rand}
    return "evaluate", {"no_rand": no_rand, "rand": rand}

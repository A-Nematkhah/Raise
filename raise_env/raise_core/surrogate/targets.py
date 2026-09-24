"""
Surrogate regression targets (CrowdNav vs highway).

CrowdNav stays on SR/CR/TR (thesis lock). Highway adds continuous Stage II
fields so the surrogate can rank cruise vs crawl, not only survival.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Locked CrowdNav / default contract (PLAN.md).
DEFAULT_TARGET_KEYS: Tuple[str, ...] = ("SR", "CR", "TR")

# Highway multi-target: survival + throughput (Phase 2).
HIGHWAY_TARGET_KEYS: Tuple[str, ...] = (
    "SR",
    "CR",
    "TR",
    "mean_speed",
    "mean_progress",
    "soft_success",
)


def target_keys_for_domain(domain: Optional[str]) -> Tuple[str, ...]:
    key = str(domain or "crowdnav").strip().lower() or "crowdnav"
    if key == "highway":
        return HIGHWAY_TARGET_KEYS
    return DEFAULT_TARGET_KEYS


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def normalize_label_row(
    labels: Mapping[str, Any],
    *,
    target_keys: Sequence[str] = DEFAULT_TARGET_KEYS,
) -> Dict[str, Any]:
    """
    Ensure every fit target is a finite float.

    Aliases: mean_speed ← ITR (unless ITR is a bogus 0 with large PL),
    mean_progress ← PL. Missing soft_success → 0.
    """
    out = dict(labels)
    pl = _finite(out.get("PL", out.get("pl", out.get("mean_progress", 0.0))))
    nt = _finite(out.get("NT", out.get("nt", 0.0)))
    itr = _finite(out.get("ITR", out.get("itr", 0.0)))
    existing_speed = out.get("mean_speed")
    speed_ok = math.isfinite(_finite(existing_speed, float("nan")))
    if not speed_ok:
        # Legacy rows: ITR=0 with PL≈800 means missing highway speed, not idle.
        if itr <= 1e-9 and pl > 50.0 and nt > 1e-6:
            out["mean_speed"] = float(pl / nt)
        else:
            out["mean_speed"] = itr
    if "mean_progress" not in out or not math.isfinite(
        _finite(out.get("mean_progress"), float("nan"))
    ):
        out["mean_progress"] = pl
    if "soft_success" not in out or not math.isfinite(
        _finite(out.get("soft_success"), float("nan"))
    ):
        out["soft_success"] = 0.0
    for key in target_keys:
        if key not in out or not math.isfinite(_finite(out.get(key), float("nan"))):
            out[key] = _finite(out.get(key, 0.0))
    return out


def filter_ok_examples(
    features: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    target_keys: Sequence[str],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Drop ``ok=False`` rows and normalize label aliases (paired)."""
    feats_out: list[Dict[str, Any]] = []
    labs_out: list[Dict[str, Any]] = []
    for feat, lab in zip(features, labels):
        if lab.get("ok") is False:
            continue
        feats_out.append(dict(feat))
        labs_out.append(normalize_label_row(lab, target_keys=target_keys))
    return feats_out, labs_out


def normalize_labels(
    labels: Sequence[Mapping[str, Any]],
    *,
    target_keys: Sequence[str],
    drop_failed: bool = True,
) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for row in labels:
        if drop_failed and row.get("ok") is False:
            continue
        out.append(normalize_label_row(row, target_keys=target_keys))
    return out


def quality_from_y_hat(y_hat: Optional[Mapping[str, Any]]) -> float:
    """
    Predicted elite quality for gate / AL.

    Rich highway y_hat → ``highway_navigation_scalar``; else SR−CR−0.5·TR.
    """
    if not y_hat:
        return float("-inf")
    rich = any(
        k in y_hat
        for k in ("mean_speed", "mean_progress", "soft_success", "selection_scalar")
    )
    if rich:
        from domains.highway.metrics import highway_navigation_scalar

        payload = dict(y_hat)
        if "PL" not in payload and "mean_progress" in payload:
            payload["PL"] = payload["mean_progress"]
        if "mean_speed" not in payload and "ITR" in payload:
            payload["mean_speed"] = payload["ITR"]
        return float(highway_navigation_scalar(payload))
    from raise_core.selection import navigation_scalar

    return float(
        navigation_scalar(
            _finite(y_hat.get("SR", y_hat.get("sr")), 0.0),
            _finite(y_hat.get("CR", y_hat.get("cr")), 0.0),
            _finite(y_hat.get("TR", y_hat.get("tr")), 0.0),
        )
    )

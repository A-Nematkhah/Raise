"""
Checkpoint / resume + wall-clock cost logging for paper-scale runs.

Keys are ``(seed, stage, round, candidate_id)``. A paper-scale job can be
interrupted mid-Stage-II/III and continue without re-training finished cells.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CheckpointKey:
    seed: int
    stage: str  # stage1 | stage2 | stage3
    round: int  # generation (stage1) or refinement round (stage2/3)
    candidate_id: str

    def path_parts(self) -> Tuple[str, str, str, str]:
        return (
            f"seed_{int(self.seed):04d}",
            str(self.stage),
            f"round_{int(self.round):03d}",
            f"{self.candidate_id}.json",
        )


@dataclass
class CostEvent:
    seed: int
    stage: str
    round: int
    candidate_id: str
    wall_seconds: float
    device: str
    resumed: bool = False
    gpu_hours: float = 0.0
    timestamp_utc: str = field(default_factory=_utc_now)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class CostLogger:
    """Append-only cost log → ``results/paper_scale/cost_log.json``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.events: List[Dict[str, Any]] = []
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    payload = json.load(f)
                self.events = list(payload.get("events") or [])
            except (OSError, json.JSONDecodeError):
                self.events = []

    def record(self, event: CostEvent) -> None:
        if event.device.startswith("cuda") and event.gpu_hours <= 0.0:
            event.gpu_hours = float(event.wall_seconds) / 3600.0
        self.events.append(event.to_dict())
        self.flush()

    def totals(self) -> Dict[str, Any]:
        by_stage: Dict[str, float] = {}
        gpu_by_stage: Dict[str, float] = {}
        for e in self.events:
            st = str(e.get("stage", "unknown"))
            by_stage[st] = by_stage.get(st, 0.0) + float(e.get("wall_seconds") or 0.0)
            gpu_by_stage[st] = gpu_by_stage.get(st, 0.0) + float(e.get("gpu_hours") or 0.0)
        return {
            "wall_seconds_by_stage": by_stage,
            "gpu_hours_by_stage": gpu_by_stage,
            "wall_seconds_total": float(sum(by_stage.values())),
            "gpu_hours_total": float(sum(gpu_by_stage.values())),
            "n_events": len(self.events),
            "note": (
                "gpu_hours approximates wall_seconds/3600 on a single CUDA device "
                "(compare against paper Table 9 cost claims with that caveat)."
            ),
        }

    def flush(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        payload = {
            "updated_utc": _utc_now(),
            "events": self.events,
            "totals": self.totals(),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)


class CheckpointStore:
    """
    Disk store keyed by ``(seed, stage, round, candidate_id)``.

    Layout::
        <root>/seed_0425/stage2/round_000/cand_0003.json
    """

    def __init__(
        self,
        root: str,
        *,
        cost_logger: Optional[CostLogger] = None,
        device: str = "cpu",
    ) -> None:
        self.root = root
        self.cost_logger = cost_logger
        self.device = device
        os.makedirs(self.root, exist_ok=True)

    def _path(self, key: CheckpointKey) -> str:
        parts = key.path_parts()
        return os.path.join(self.root, *parts)

    def has(self, key: CheckpointKey) -> bool:
        return os.path.isfile(self._path(key))

    def load(self, key: CheckpointKey) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save(
        self,
        key: CheckpointKey,
        payload: Dict[str, Any],
        *,
        wall_seconds: float,
        resumed: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        body = {
            "key": {
                "seed": key.seed,
                "stage": key.stage,
                "round": key.round,
                "candidate_id": key.candidate_id,
            },
            "saved_utc": _utc_now(),
            "wall_seconds": float(wall_seconds),
            "resumed": bool(resumed),
            "payload": payload,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

        if self.cost_logger is not None:
            self.cost_logger.record(
                CostEvent(
                    seed=key.seed,
                    stage=key.stage,
                    round=key.round,
                    candidate_id=key.candidate_id,
                    wall_seconds=float(wall_seconds),
                    device=self.device,
                    resumed=bool(resumed),
                    extra=dict(extra or {}),
                )
            )
        return path

    def mark_stage_done(self, seed: int, stage: str, summary: Dict[str, Any]) -> str:
        path = os.path.join(self.root, f"seed_{int(seed):04d}", f"{stage}_done.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        body = {"seed": seed, "stage": stage, "saved_utc": _utc_now(), **summary}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(body, f, indent=2, sort_keys=True)
            f.write("\n")
        return path

    def stage_done(self, seed: int, stage: str) -> bool:
        path = os.path.join(self.root, f"seed_{int(seed):04d}", f"{stage}_done.json")
        return os.path.isfile(path)

    def load_stage_done(self, seed: int, stage: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.root, f"seed_{int(seed):04d}", f"{stage}_done.json")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def seed_complete(self, seed: int) -> bool:
        return self.stage_done(seed, "stage3")


class Timer:
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        return float(time.perf_counter() - self._t0)

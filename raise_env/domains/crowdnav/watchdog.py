"""
Stall watchdog for CrowdNav PPO labels.

A reward that passed the smoke test and Score1 can still hang on a state the
RL policy reaches later (e.g. ``for i in range(10**9)`` behind a rare branch).
:class:`StallWatchdog` watches a per-thread heartbeat that
:class:`RewardInjectedCrowdNavEnv` bumps on every step; when it goes silent for
``stall_seconds`` it dumps all thread stacks and raises :class:`RewardStallError`
asynchronously in the training thread, so the label fails instead of the run
hanging. Works for training on the main thread or a label worker thread.

Limitation: async exceptions are delivered between bytecodes, so a single C
call that never returns (e.g. a huge ``int ** int``) cannot be interrupted.
"""

from __future__ import annotations

import ctypes
import faulthandler
import logging
import sys
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_STALL_SECONDS = 300.0

_local = threading.local()


class RewardStallError(Exception):
    """Raised in the training thread when no env step happened for too long."""


def beat() -> None:
    """Heartbeat for the watchdog guarding the current thread (no-op if none)."""
    dog = getattr(_local, "dog", None)
    if dog is not None:
        dog.last_beat = time.monotonic()


def stall_seconds_from_config(config: Any) -> float:
    raw = getattr(config, "stall_seconds", None)
    if raw is None:
        return DEFAULT_STALL_SECONDS
    return max(0.0, float(raw))


def _set_async_exc(thread_id: int, exc_type: Optional[type]) -> int:
    return int(
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id),
            ctypes.py_object(exc_type) if exc_type is not None else None,
        )
    )


class StallWatchdog:
    """
    Context manager guarding the calling thread.

    On exit after a stall, re-raises the interrupted work's exception as
    ``RuntimeError`` (label failure). ``stall_seconds <= 0`` disables it.
    """

    def __init__(self, stall_seconds: float, *, label: str = "") -> None:
        self.stall_seconds = float(stall_seconds)
        self.label = label
        self.fired = False
        self.last_beat = time.monotonic()
        self._lock = threading.Lock()
        self._armed = False
        self._stop = threading.Event()
        self._thread_id: Optional[int] = None
        self._monitor: Optional[threading.Thread] = None
        self._prev: Optional["StallWatchdog"] = None

    def __enter__(self) -> "StallWatchdog":
        if self.stall_seconds <= 0.0:
            return self
        self._thread_id = threading.get_ident()
        self.last_beat = time.monotonic()
        self._prev = getattr(_local, "dog", None)
        _local.dog = self
        self._armed = True
        self._monitor = threading.Thread(
            target=self._watch, name=f"stall-watchdog-{self.label}", daemon=True
        )
        self._monitor.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.stall_seconds <= 0.0:
            return False
        self._stop.set()
        with self._lock:
            self._armed = False
            if self.fired and self._thread_id is not None:
                # Drop a pending stall exception that has not been delivered yet.
                _set_async_exc(self._thread_id, None)
        if self._monitor is not None:
            self._monitor.join(timeout=5.0)
        _local.dog = self._prev
        if self.fired and exc_type is not None:
            raise RuntimeError(
                f"PPO label {self.label or '<candidate>'} stalled: no env step for "
                f"{self.stall_seconds:.0f}s (reward hang?)"
            ) from exc
        return False

    def _watch(self) -> None:
        poll = min(5.0, max(0.05, self.stall_seconds / 10.0))
        while not self._stop.wait(poll):
            idle = time.monotonic() - self.last_beat
            if idle < self.stall_seconds:
                continue
            with self._lock:
                if not self._armed or self._thread_id is None:
                    return
                self.fired = True
                logger.error(
                    "CrowdNav PPO label %s stalled for %.0fs; interrupting training thread",
                    self.label or "<candidate>",
                    idle,
                )
                try:
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except Exception:  # noqa: BLE001 — stderr may be captured / fd-less
                    pass
                _set_async_exc(self._thread_id, RewardStallError)
            return

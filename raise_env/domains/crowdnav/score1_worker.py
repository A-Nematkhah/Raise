"""
CrowdNav Stage I Score1 in a persistent, killable child process.

The AST policy bans ``while`` but cannot bound ``for i in range(10**9)``, so a
reward that passes the smoke test can still stall Score1 over the dataset.
The worker loads the dataset once, receives the candidate as a
pickled :class:`SandboxedReward` (recompiled from source in the child), and is
terminated and lazily respawned when a candidate exceeds ``timeout_seconds``.
That candidate gets a rejected :class:`Score1Result` (``score1_timeout``).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import sys
import threading
import weakref
from dataclasses import replace
from typing import Any, Callable, Optional, Sequence, Tuple

from domains.crowdnav.settings import (
    DEFAULT_ENV_CONFIG_PATH,
    CrowdNavSettings,
    get_settings,
)
from domains.crowdnav.score1 import score_crowdnav_dataset
from domains.crowdnav.stage1 import (
    CrowdNavTrajectoryRecord,
    load_crowdnav_dataset,
    make_score_fn as stage1_make_score_fn,
    resolve_dataset_path,
    with_settings,
)
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.scoring import Score1Result

logger = logging.getLogger(__name__)

DEFAULT_SCORE1_TIMEOUT_SECONDS = 60.0
# Spawn + numpy import + dataset load; generous for cold Windows starts.
_STARTUP_TIMEOUT_SECONDS = 120.0


def _sanitize_child_argv() -> None:
    """Drop Raise CLI flags so CrowdNav++ ``get_args()`` cannot SystemExit the child.

    Windows ``spawn`` re-imports modules with the parent's ``sys.argv``. Upstream
    ``crowd_nav.configs.config.Config`` calls ``get_args()`` at class body time.
    """
    prog = sys.argv[0] if sys.argv else "crowdnav-score1"
    sys.argv = [prog]


def settings_for_score1_child(settings: CrowdNavSettings) -> CrowdNavSettings:
    """Strip ``env_config_obj`` so spawn does not pickle/import upstream ``Config``.

    Pickling a live ``Config`` forces the child to import
    ``crowd_nav.configs.config``, which parses ``sys.argv`` and dies on Raise flags
    (``--config``, ``--output-dir``, …).
    """
    if settings.env_config_obj is None:
        return settings
    path = settings.env_config_path or DEFAULT_ENV_CONFIG_PATH
    return replace(settings, env_config_obj=None, env_config_path=path)


def _worker_main(conn: Any, dataset_path: str, settings: CrowdNavSettings) -> None:
    """Child entry (top-level for spawn): score candidates until told to stop.

    The child starts with default settings, so the parent's are passed in.
    """
    _sanitize_child_argv()
    try:
        dataset = with_settings(load_crowdnav_dataset(dataset_path), settings)
    except BaseException as exc:  # noqa: BLE001
        conn.send(("init_err", repr(exc)))
        return
    conn.send(("ready", len(dataset)))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        reward_fn, candidate_id = msg
        try:
            result = score_crowdnav_dataset(reward_fn, dataset, candidate_id=candidate_id)
            conn.send(("ok", result))
        except BaseException as exc:  # noqa: BLE001
            try:
                conn.send(("err", exc))
            except Exception:  # noqa: BLE001 — unpicklable exception
                wrapper = RewardSandboxError if isinstance(exc, RewardSandboxError) else RuntimeError
                conn.send(("err", wrapper(f"{type(exc).__name__}: {exc}")))


def _stop_process(proc: Any, conn: Any) -> None:
    try:
        conn.send(None)
    except Exception:  # noqa: BLE001
        pass
    proc.join(timeout=2.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def rejected_result(reason: str) -> Score1Result:
    return Score1Result(
        score=float("-inf"),
        n_nan_frames=0,
        n_frames=0,
        rejected=True,
        reject_reason=reason,
        error=reason,
    )


class Score1TimeoutWorker:
    """Callable ``(reward_fn, *, candidate_id) -> Score1Result`` with a hard timeout."""

    def __init__(
        self,
        dataset_path: str,
        dataset: Sequence[CrowdNavTrajectoryRecord],
        *,
        timeout_seconds: float = DEFAULT_SCORE1_TIMEOUT_SECONDS,
        settings: Optional[CrowdNavSettings] = None,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        self.dataset_path = str(dataset_path)
        self.dataset = dataset
        self.settings = settings or get_settings()
        self.timeout_seconds = float(timeout_seconds)
        self.restarts = 0
        self._lock = threading.Lock()
        self._proc: Optional[Any] = None
        self._conn: Optional[Any] = None
        self._finalizer: Optional[weakref.finalize] = None

    def __call__(self, reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        if not self._can_offload(reward_fn):
            return score_crowdnav_dataset(reward_fn, self.dataset, candidate_id=candidate_id)
        with self._lock:
            return self._score_in_child(reward_fn, candidate_id)

    def close(self) -> None:
        with self._lock:
            self._discard()

    @staticmethod
    def _can_offload(reward_fn: Any) -> bool:
        # Daemonic processes may not spawn children; callables without source
        # (test doubles) cannot be rebuilt in the child.
        if mp.current_process().daemon or not getattr(reward_fn, "_source_code", None):
            return False
        try:
            pickle.dumps(reward_fn)
        except Exception:  # noqa: BLE001
            return False
        return True

    def _score_in_child(self, reward_fn: Any, candidate_id: str) -> Score1Result:
        self._ensure_started()
        if self._conn is None:
            raise RuntimeError("Score1 worker connection is not open")
        try:
            self._conn.send((reward_fn, candidate_id))
            ready = self._conn.poll(self.timeout_seconds)
        except (BrokenPipeError, EOFError, OSError):
            return self._fail("score1_worker_crashed", candidate_id)
        if not ready:
            return self._fail("score1_timeout", candidate_id)
        try:
            status, payload = self._conn.recv()
        except (EOFError, OSError):
            return self._fail("score1_worker_crashed", candidate_id)
        if status == "err":
            raise payload
        return payload

    def _fail(self, reason: str, candidate_id: str) -> Score1Result:
        logger.warning(
            "CrowdNav Score1 for %s failed (%s, timeout=%.0fs); restarting worker",
            candidate_id or "<candidate>",
            reason,
            self.timeout_seconds,
        )
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
        self._discard()
        self.restarts += 1
        return rejected_result(reason)

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        if self._proc is not None:
            self._discard()
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        child_settings = settings_for_score1_child(self.settings)
        proc = ctx.Process(
            target=_worker_main,
            args=(child_conn, self.dataset_path, child_settings),
            daemon=True,
            name="crowdnav-score1",
        )
        proc.start()
        child_conn.close()
        if not parent_conn.poll(_STARTUP_TIMEOUT_SECONDS):
            _stop_process(proc, parent_conn)
            raise RuntimeError("CrowdNav Score1 worker did not start in time")
        status, payload = parent_conn.recv()
        if status != "ready":
            _stop_process(proc, parent_conn)
            raise RuntimeError(f"CrowdNav Score1 worker failed to load dataset: {payload}")
        self._proc, self._conn = proc, parent_conn
        self._finalizer = weakref.finalize(self, _stop_process, proc, parent_conn)

    def _discard(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self._proc = None
        self._conn = None


def make_score_fn(
    *,
    mode: str = "dataset",
    dataset_path: Optional[str] = None,
    timeout_seconds: float = DEFAULT_SCORE1_TIMEOUT_SECONDS,
) -> Tuple[Callable[..., Any], Optional[Any]]:
    """Pack entry: :func:`stage1.make_score_fn`, with dataset Score1 in the timeout worker."""
    settings = get_settings()
    score_fn, dataset = stage1_make_score_fn(
        mode=mode, dataset_path=dataset_path, settings=settings
    )
    if dataset is None:
        return score_fn, dataset
    worker = Score1TimeoutWorker(
        os.path.abspath(resolve_dataset_path(dataset_path)),
        dataset,
        timeout_seconds=timeout_seconds,
        settings=settings,
    )
    return worker, dataset

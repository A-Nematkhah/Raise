"""
Live terminal feedback for RAISE runs.

This module is about what a user sees *while a run is in progress*.
It does not write JSON artifacts or change algorithms. Progress bars use
``tqdm`` when available and degrade to quiet counters when stdout/stderr
is not a TTY (CI / redirected logs).
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

_VERBOSE = False


def set_verbose(enabled: bool) -> None:
    """Enable opt-in DEBUG-noise that is normally suppressed."""
    global _VERBOSE
    _VERBOSE = bool(enabled)


def ensure_utf8_stdio() -> None:
    """Write stdout/stderr as UTF-8 even when redirected (Windows defaults to cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def is_interactive() -> bool:
    """True when stderr looks like a live terminal (not a pipe / CI capture)."""
    if os.environ.get("RAISE_FORCE_TQDM", "").strip() in ("1", "true", "yes"):
        return True
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # noqa: BLE001
        return False


def configure_run_logging(*, verbose: bool = False) -> None:
    """
    Readable default logs: INFO for RAISE, quiet HTTP/SDK noise.

    ``--verbose`` turns on DEBUG for RAISE packages only — not httpx/groq
    full request dumps (those made 1h runs unreadable).
    """
    root_level = logging.INFO
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    set_verbose(bool(verbose))
    pkg_level = logging.DEBUG if verbose else logging.INFO
    for name in ("domains", "raise_core"):
        logging.getLogger(name).setLevel(pkg_level)

    # Third-party chatter (Groq/httpx dump entire prompts at DEBUG).
    for name in (
        "httpx",
        "httpcore",
        "groq",
        "openai",
        "urllib3",
        "asyncio",
        "matplotlib",
        "PIL",
        "filelock",
        "stable_baselines3",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message="Box bound precision lowered by casting to float32",
    )
    warnings.filterwarnings(
        "ignore",
        message="An input array is constant; the correlation coefficient is not defined",
    )
    warnings.filterwarnings(
        "ignore",
        message="You are trying to run PPO on the GPU",
        category=UserWarning,
    )


def status(message: str, *, stage: Optional[str] = None) -> None:
    """Short consistent status line (INFO)."""
    prefix = f"[{stage}] " if stage else ""
    logger.info("%s%s", prefix, message)


def warn(message: str, *, stage: Optional[str] = None) -> None:
    prefix = f"[{stage}] " if stage else ""
    logger.warning("%s%s", prefix, message)


def fail(message: str, *, stage: Optional[str] = None) -> None:
    """One-line failure summary that stands out above a traceback."""
    prefix = f"[{stage}] " if stage else ""
    logger.error("FAILED %s%s", prefix, message)


def debug(message: str, *, stage: Optional[str] = None) -> None:
    if not _VERBOSE:
        logger.debug("%s", message)
        return
    prefix = f"[{stage}] " if stage else ""
    logger.info("%s%s", prefix, message)


def banner(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info("%s", title)
    logger.info(bar)


def format_seconds(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    m, rem = divmod(s, 60.0)
    if m < 60.0:
        return f"{int(m)}m{rem:04.1f}s"
    h, rem_m = divmod(m, 60.0)
    return f"{int(h)}h{int(rem_m):02d}m"


@contextmanager
def timed(label: str, *, stage: Optional[str] = None) -> Iterator[None]:
    """Print start + elapsed finish for operations expected to take >~5s."""
    status(f"{label} ...", stage=stage)
    t0 = time.perf_counter()
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        fail(f"{label} crashed after {format_seconds(elapsed)}: {exc}", stage=stage)
        raise
    else:
        elapsed = time.perf_counter() - t0
        status(f"{label} done in {format_seconds(elapsed)}", stage=stage)


class _PlainProgress:
    """Minimal fallback when tqdm is unavailable or non-interactive."""

    def __init__(self, total: Optional[int], desc: str, *, log_every: int = 0) -> None:
        self.total = total
        self.desc = desc
        self.n = 0
        self._log_every = log_every
        if total and total > 0 and log_every <= 0:
            self._log_every = max(1, total // 10)
        if desc:
            status(f"{desc} (0/{total})" if total else f"{desc} ...")

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.total and self._log_every and (
            self.n % self._log_every == 0 or self.n >= self.total
        ):
            pct = 100.0 * self.n / max(1, self.total)
            status(f"{self.desc} {self.n}/{self.total} ({pct:.0f}%)")

    def close(self) -> None:
        if self.total and self.n < self.total:
            status(f"{self.desc} {self.n}/{self.total} (stopped early)")
        elif self.desc and not self.total:
            status(f"{self.desc} finished ({self.n} steps)")

    def __enter__(self) -> "_PlainProgress":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def progress(
    iterable: Optional[Iterable] = None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    leave: bool = True,
    unit: str = "it",
    mininterval: float = 0.5,
) -> Any:
    """
    tqdm progress bar when interactive; otherwise a quiet periodic logger.

    Can wrap an iterable ``progress(range(n), desc=...)`` or be used as a
    manual bar ``with progress(total=n, desc=...) as p: p.update(1)``.
    """
    disable = not is_interactive()
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        if iterable is not None:
            items = list(iterable) if total is None else iterable
            n_total = total if total is not None else (
                len(items) if hasattr(items, "__len__") else None
            )
            bar = _PlainProgress(n_total, desc)

            def _gen():
                for item in items:
                    yield item
                    bar.update(1)
                bar.close()

            return _gen()
        return _PlainProgress(total, desc)

    kwargs = dict(
        total=total,
        desc=desc,
        leave=leave,
        unit=unit,
        mininterval=mininterval,
        disable=disable,
        file=sys.stderr,
        dynamic_ncols=True,
    )
    if iterable is not None:
        return tqdm(iterable, **kwargs)
    return tqdm(**kwargs)


def final_run_summary(
    *,
    output_dir: str,
    wall_seconds: float,
    best_stage1: Optional[Any] = None,
    best_stage2: Optional[Any] = None,
    best_stage3: Optional[Any] = None,
) -> None:
    banner("RAISE - finished")
    status(f"Total wall-clock: {format_seconds(wall_seconds)}")
    for label, cand in (
        ("Stage I best", best_stage1),
        ("Stage II best", best_stage2),
        ("Stage III best", best_stage3),
    ):
        if cand is None:
            status(f"{label}: (none)")
            continue
        score = getattr(cand, "score", None)
        md = getattr(cand, "metadata", None) or {}
        last = md.get("last_metrics") or {}
        extra = ""
        if last:
            extra = (
                f"  SR={float(last.get('SR', 0)):.3f} "
                f"CR={float(last.get('CR', 0)):.3f} "
                f"TR={float(last.get('TR', 0)):.3f}"
            )
        elif score is not None:
            extra = f"  Score1={_fmt_score(score)}"
        status(f"{label}: {getattr(cand, 'candidate_id', '?')}{extra}")
    status(f"Artifacts: {os.path.abspath(output_dir)}")
    status(
        "Key files: closed_loop/REPORT.txt, closed_loop/epochs.jsonl, "
        "manifest.json, best_stage3.json, final_candidate.json"
    )


def _fmt_score(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == float("-inf"):
        return "-inf"
    return f"{v:.4f}"

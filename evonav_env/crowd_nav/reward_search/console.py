"""
Live terminal feedback for EvoNav runs.

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
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Optional, Sequence

logger = logging.getLogger(__name__)

_VERBOSE = False


def set_verbose(enabled: bool) -> None:
    """Enable opt-in DEBUG-noise that is normally suppressed."""
    global _VERBOSE
    _VERBOSE = bool(enabled)


def is_verbose() -> bool:
    return _VERBOSE


def is_interactive() -> bool:
    """True when stderr looks like a live terminal (not a pipe / CI capture)."""
    if os.environ.get("EVONAV_FORCE_TQDM", "").strip() in ("1", "true", "yes"):
        return True
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # noqa: BLE001
        return False


def configure_run_logging(*, verbose: bool = False) -> None:
    """
    Readable default logs: INFO for EvoNav, quiet HTTP/SDK noise.

    ``--verbose`` turns on DEBUG for ``crowd_nav`` / ``crowd_sim`` only —
    not httpx/groq full request dumps (those made 1h runs unreadable).
    """
    import logging
    import warnings

    root_level = logging.INFO
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    set_verbose(bool(verbose))
    pkg_level = logging.DEBUG if verbose else logging.INFO
    for name in ("crowd_nav", "crowd_sim", "rl"):
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


def stage1_gen_summary(
    generation: int,
    total_gens: int,
    ranked: Sequence[Any],
    *,
    global_best: Optional[Any] = None,
) -> None:
    """Print a one-screen Stage I generation result."""
    if not ranked:
        status(f"Gen {generation}/{total_gens} - empty population", stage="Stage I")
        return
    best = ranked[0]
    worst = ranked[-1]
    best_s = getattr(best, "score", None)
    worst_s = getattr(worst, "score", None)
    gb = global_best or best
    gb_s = getattr(gb, "score", None)
    status(
        f"Gen {generation}/{total_gens} summary - "
        f"best={getattr(best, 'candidate_id', '?')} "
        f"score={_fmt_score(best_s)} | "
        f"worst={getattr(worst, 'candidate_id', '?')} "
        f"score={_fmt_score(worst_s)} | "
        f"global_best={getattr(gb, 'candidate_id', '?')} "
        f"score={_fmt_score(gb_s)}",
        stage="Stage I",
    )


def stage_round_summary(
    stage: str,
    round_index: int,
    total_rounds: int,
    records: Sequence[Any],
) -> None:
    """Print Stage II/III round metrics (SR/CR/TR) for scanability."""
    if not records:
        status(f"Round {round_index + 1}/{total_rounds} - no records", stage=stage)
        return
    lines = [
        f"Round {round_index + 1}/{total_rounds} metrics:",
        f"  {'candidate':<16} {'SR':>6} {'CR':>6} {'TR':>6} {'scalar':>8} refined",
    ]
    best_id = None
    best_scalar = float("-inf")
    for rec in records:
        m = getattr(rec, "metrics", None)
        if m is None:
            continue
        sr = float(getattr(m, "sr", 0.0))
        cr = float(getattr(m, "cr", 0.0))
        tr = float(getattr(m, "tr", 0.0))
        scalar = sr - cr - 0.5 * tr
        cid = str(getattr(rec, "candidate_id", "?"))
        refined = "yes" if getattr(rec, "refined", False) else (
            "kept" if getattr(rec, "kept_previous", False) else "-"
        )
        lines.append(
            f"  {cid:<16} {sr:6.3f} {cr:6.3f} {tr:6.3f} {scalar:8.3f} {refined}"
        )
        if scalar > best_scalar:
            best_scalar = scalar
            best_id = cid
    if best_id is not None:
        lines.append(
            f"  best this round: {best_id} (SR-CR-0.5TR={best_scalar:.3f})"
        )
    for line in lines:
        status(line, stage=stage)


def final_run_summary(
    *,
    output_dir: str,
    wall_seconds: float,
    best_stage1: Optional[Any] = None,
    best_stage2: Optional[Any] = None,
    best_stage3: Optional[Any] = None,
    closed_loop: bool = False,
) -> None:
    banner("EvoNav Algorithm 1 - finished")
    status(f"Total wall-clock: {format_seconds(wall_seconds)}")
    if closed_loop:
        status("mode: closed_loop_v1 (innovation)")
        try:
            from crowd_nav.reward_search.closed_loop.report import (
                build_closed_loop_report,
            )

            for line in build_closed_loop_report(output_dir).strip().splitlines():
                status(line, stage="closed-loop")
        except Exception as exc:  # noqa: BLE001
            warn(f"closed-loop report unavailable: {exc}", stage="pipeline")
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
    if closed_loop:
        status(
            "Key files: closed_loop/REPORT.txt, closed_loop/epochs.jsonl, "
            "manifest.json, best_stage3.json"
        )
    else:
        status(
            "Key files: manifest.json, final_candidate.json, "
            "best_stage1.json, best_stage2.json, best_stage3.json"
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

"""
CPU parallelism defaults for Stage II/III RealPolicyTrainer.

Kept separate from Config/argparse so Stage2Config / Stage3Config can share
one policy: prefer true multi-process rollouts up to a soft cap, leave one
core free for the learner / OS.
"""

from __future__ import annotations

import os
from typing import Optional


def default_num_processes(*, cap: int = 16) -> int:
    """
    Reasonable parallel env count for this machine.

    ``min(cap, max(1, cpu_count - 1))`` — never returns 0.
    """
    n = os.cpu_count() or 1
    if n <= 1:
        return 1
    return max(1, min(int(cap), int(n) - 1))


def default_num_mini_batch(num_processes: int) -> int:
    """PPO/A2C mini-batches must be <= num_processes (storage assert)."""
    n = max(1, int(num_processes))
    return max(1, min(2, n))


def resolve_num_processes(value: Optional[int], *, cap: int = 16) -> int:
    """``None`` or ``<= 0`` → auto; otherwise clamp to at least 1."""
    if value is None or int(value) <= 0:
        return default_num_processes(cap=cap)
    return max(1, int(value))

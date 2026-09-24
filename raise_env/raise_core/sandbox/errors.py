"""Errors raised by the reward sandbox (validation or runtime)."""

from __future__ import annotations


class RewardSandboxError(RuntimeError):
    """Candidate code failed a sandbox check or a sandboxed compute() call."""

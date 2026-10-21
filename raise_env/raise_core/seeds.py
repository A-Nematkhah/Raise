"""Eval and PPO seed offsets (single source of truth)."""

from __future__ import annotations

# Closed-loop generation uses ``seed + g * 1000``; these stay clear of realistic G.
STAGE3_EVAL_SEED_OFFSET: int = 10_000_000
FINAL_EVAL_SEED_OFFSET: int = 20_000_000

# Training-scenario eval and stochastic policy sampling (relative to eval seed base).
TRAIN_EVAL_SEED_OFFSET: int = 10_003
HOLDOUT_SEED_OFFSET: int = 50_003

__all__ = [
    "FINAL_EVAL_SEED_OFFSET",
    "HOLDOUT_SEED_OFFSET",
    "STAGE3_EVAL_SEED_OFFSET",
    "TRAIN_EVAL_SEED_OFFSET",
]

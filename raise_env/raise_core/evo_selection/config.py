"""Configuration for optional LLM-based candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class SelectionConfig:
    mode: str = "legacy"
    top_k: int = 2
    n_votes: int = 3
    temperature: float = 0.1
    p_restart: float = 0.1
    guards: bool = False
    guard_success_threshold: float = 0.9
    guard_min_path_length: float = 0.1
    reflection_note_cap: int = 3
    seed: int = 425

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in ("legacy", "llm_rank"):
            raise ValueError("selection.mode must be 'legacy' or 'llm_rank'")
        if int(self.top_k) < 1:
            raise ValueError("selection.top_k must be >= 1")
        if int(self.n_votes) < 1:
            raise ValueError("selection.n_votes must be >= 1")
        if not math.isfinite(float(self.temperature)) or float(self.temperature) < 0:
            raise ValueError("selection.temperature must be finite and >= 0")
        if not math.isfinite(float(self.p_restart)) or not 0 <= float(self.p_restart) <= 1:
            raise ValueError("selection.p_restart must be in [0, 1]")
        if not 0 <= float(self.guard_success_threshold) <= 1:
            raise ValueError("selection.guard_success_threshold must be in [0, 1]")
        if float(self.guard_min_path_length) < 0:
            raise ValueError("selection.guard_min_path_length must be >= 0")
        if int(self.reflection_note_cap) < 0:
            raise ValueError("selection.reflection_note_cap must be >= 0")

        self.top_k = int(self.top_k)
        self.n_votes = int(self.n_votes)
        self.temperature = float(self.temperature)
        self.p_restart = float(self.p_restart)
        self.guard_success_threshold = float(self.guard_success_threshold)
        self.guard_min_path_length = float(self.guard_min_path_length)
        self.reflection_note_cap = int(self.reflection_note_cap)
        self.seed = int(self.seed)

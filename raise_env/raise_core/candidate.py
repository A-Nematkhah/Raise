"""The evolvable reward individual shared by every RAISE stage."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from raise_core.sandbox.runtime import RewardFunction

# Stage II results that belong to the genome (and its saved policy), so an
# identical genome can take them over instead of being trained again.
EVALUATION_KEYS = (
    "fitness",
    "selection_scalar",
    "last_metrics",
    "proxy_feedback",
    "checkpoint_path",
    "eval_generation",
    "fitness_history",
    "train_steps",
    "train_seed",
    "ppo_n_steps",
    "training_dynamics",
    "eval_seed_base",
    "eval_episodes",
    "eval_deterministic",
)


def evaluation_fields(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    md = metadata or {}
    return {k: md[k] for k in EVALUATION_KEYS if md.get(k) is not None}


def genome_fingerprint(code: str) -> str:
    """Hash of the code's AST without docstrings (comments/layout do not count)."""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        canonical = "\n".join(line.rstrip() for line in str(code or "").strip().splitlines())
    else:
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (
                isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
        canonical = ast.dump(tree, include_attributes=False)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


@dataclass
class RewardCandidate:
    """One evolvable reward individual."""

    candidate_id: str
    code: str
    reward_fn: Optional[RewardFunction] = None
    score: Optional[float] = None
    valid: bool = False
    origin: str = "unknown"  # initial | mutation | crossover | random
    parent_ids: tuple = ()
    validation_error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        return self.valid and self.reward_fn is not None

    def as_reward_function(self) -> RewardFunction:
        """Return the validated reward callable (Stage I Score1 entry point)."""
        if self.reward_fn is None:
            raise ValueError(f"Candidate {self.candidate_id} has no reward_fn")
        return self.reward_fn

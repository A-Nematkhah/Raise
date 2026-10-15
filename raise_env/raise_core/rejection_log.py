"""Categorize sandbox validation failures for Gen0 rejection logs."""

from __future__ import annotations

from typing import Optional

_FORBIDDEN_CONSTRUCT_MARKERS = (
    "comprehensions are forbidden",
    "generator expressions are forbidden",
    "while loops are forbidden",
    "class definitions are forbidden",
    "async functions are forbidden",
    "async with is forbidden",
    "await is forbidden",
    "yield is forbidden",
    "yield from is forbidden",
    "with statements are forbidden",
    "global is forbidden",
    "nonlocal is forbidden",
)


def categorize_validation_error(error: Optional[str]) -> str:
    """Map a RewardSandboxError message to a coarse rejection bucket.

    Order matters: a runtime error message can quote arbitrary text (e.g. the
    ``timeout`` field), so the specific sandbox phrases are matched first.
    """
    if not error:
        return "unknown"
    text = error.lower()
    if text.startswith("duplicate genome"):
        return "duplicate"
    if text.startswith("duplicate behaviour") or text.startswith("duplicate behavior"):
        return "duplicate_behavior"
    if "syntax error" in text:
        return "syntax"
    if "source code is empty" in text or "no source code" in text:
        return "empty_code"
    if "import is forbidden" in text or "from-import is forbidden" in text:
        return "forbidden_import"
    if "not on the domain allowlist" in text or (
        "math." in text and "is not allowed" in text
    ):
        return "not_on_allowlist"
    if any(marker in text for marker in _FORBIDDEN_CONSTRUCT_MARKERS):
        return "forbidden_construct"
    if "mutable top-level values" in text:
        return "mutable_global"
    if "is forbidden" in text:
        return "forbidden_name"
    if "execution exceeded timeout of" in text:
        return "timeout"
    if "reward too slow" in text:
        return "too_slow"
    if "reward magnitude too large" in text:
        return "reward_too_large"
    if "non-finite" in text or "must return a finite" in text:
        return "non_finite_output"
    if (
        "expected exactly one" in text
        or "must be named" in text
        or "must take exactly" in text
        or "compute_reward must not use" in text
        or "wrong signature" in text
        or "unsupported top-level" in text
    ):
        return "wrong_signature"
    if "top-level" in text:
        return "top_level_code"
    if "exec failed" in text:
        return "exec_failed"
    if "runtime error" in text:
        return "runtime_error"
    if "exceeds max_code_length" in text:
        return "code_too_long"
    return "other"

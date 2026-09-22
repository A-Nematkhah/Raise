"""Categorize sandbox validation failures for Gen0 rejection logs."""

from __future__ import annotations

from typing import Optional


def categorize_validation_error(error: Optional[str]) -> str:
    """Map a RewardSandboxError message to a coarse rejection bucket."""
    if not error:
        return "unknown"
    text = error.lower()
    if "syntax error" in text:
        return "syntax"
    if "source code is empty" in text or "no source code" in text:
        return "empty_code"
    if "import is forbidden" in text or "from-import is forbidden" in text:
        return "forbidden_import"
    if "is forbidden" in text:
        return "forbidden_name"
    if "timeout" in text:
        return "timeout"
    if "non-finite" in text or "must return a finite" in text:
        return "non_finite_output"
    if (
        "expected exactly one" in text
        or "must be named" in text
        or "must take exactly" in text
        or "wrong signature" in text
        or "unsupported top-level" in text
    ):
        return "wrong_signature"
    if "exec failed" in text:
        return "exec_failed"
    if "runtime error" in text:
        return "runtime_error"
    if "exceeds max_code_length" in text:
        return "code_too_long"
    return "other"

"""LLM provider error types shared by the clients and the Groq key pool."""

from __future__ import annotations


class CompletionTruncatedError(RuntimeError):
    """The completion stopped at ``max_tokens`` (or strongly appears to have)."""


class EmptyCompletionError(RuntimeError):
    """Provider returned HTTP success with empty/missing message content."""


class GroqRateLimitError(RuntimeError):
    """Every Groq key stayed rate-limited beyond the wait budget."""


__all__ = ["CompletionTruncatedError", "EmptyCompletionError", "GroqRateLimitError"]

"""
Input-token budget for LLM requests: counting, fitting, validation.

Every Stage I LLM request is measured on its complete rendered input (client
system message + RAISE system prompt + user prompt) before it is sent:

- Exact count with OpenAI's o200k tokenizers (``tiktoken``) for gpt-oss
  providers (Groq / vLLM), whose models use that vocabulary.
- Other providers (e.g. Qwen on Ollama) use the o200k count as a proxy
  tokenizer times ``PROXY_TOKENIZER_MARGIN``.
- Without ``tiktoken`` (or its vocabulary file), a word/punctuation piece
  estimator times ``HEURISTIC_MARGIN``, never below ``chars / 3``. Unscaled,
  it reads 1.38-1.61x the o200k count on the RAISE prompts and reward code.

Chat-template tokens are added per message and once per request
(gpt-oss on Groq/vLLM prepends a dated harmony system header).
"""

from __future__ import annotations

import functools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import tiktoken
except ImportError:  # optional dependency: heuristic fallback below
    tiktoken = None

PER_MESSAGE_OVERHEAD = 8
TEMPLATE_OVERHEAD = 64
PROXY_TOKENIZER_MARGIN = 1.15
HEURISTIC_MARGIN = 1.1
_O200K_PROVIDERS = frozenset({"groq", "vllm"})
_ENCODING_NAMES = ("o200k_harmony", "o200k_base")
_PIECE_RE = re.compile(r"[A-Za-z]+|\d|\s+|[^\sA-Za-z\d]")


class PromptBudgetError(RuntimeError):
    """The smallest rendering of a request still exceeds the hard input limit."""

    def __init__(self, message: str, fit: "BudgetFit") -> None:
        super().__init__(message)
        self.fit = fit


@functools.lru_cache(maxsize=1)
def _o200k_encoding() -> Any:
    if tiktoken is None:
        return None
    for name in _ENCODING_NAMES:
        try:
            return tiktoken.get_encoding(name)
        except Exception:  # noqa: BLE001 — unknown name or vocabulary download failed
            continue
    return None


def heuristic_token_count(text: str) -> int:
    """Conservative BPE-free estimate: letters per 4 chars, digits/punct per char."""
    n = 0
    for piece in _PIECE_RE.findall(text or ""):
        head = piece[0]
        if head.isalpha():
            n += math.ceil(len(piece) / 4)
        elif head.isspace():
            n += 1 if ("\n" in piece or len(piece) > 1) else 0
        else:
            n += 1
    return max(n, math.ceil(len(text or "") / 3))


@dataclass(frozen=True)
class TokenCounter:
    """Counts tokens of rendered prompt text (``method`` names the tokenizer)."""

    method: str
    margin: float = 1.0
    _encode: Optional[Callable[[str], int]] = field(default=None, repr=False, compare=False)

    def count(self, text: str) -> int:
        raw = self._encode(text) if self._encode is not None else heuristic_token_count(text)
        return int(math.ceil(raw * self.margin))

    def count_request(self, messages: Sequence[str]) -> int:
        """Tokens of a full chat request: messages + template overhead."""
        return (
            sum(self.count(m) for m in messages)
            + PER_MESSAGE_OVERHEAD * len(messages)
            + TEMPLATE_OVERHEAD
        )


def make_token_counter(provider: str = "") -> TokenCounter:
    enc = _o200k_encoding()
    if enc is None:
        return TokenCounter(method=f"heuristic x{HEURISTIC_MARGIN}", margin=HEURISTIC_MARGIN)

    def _encode(text: str) -> int:
        return len(enc.encode(text or "", disallowed_special=()))

    if str(provider).strip().lower() in _O200K_PROVIDERS:
        return TokenCounter(method=f"tiktoken:{enc.name}", margin=1.0, _encode=_encode)
    return TokenCounter(
        method=f"tiktoken:{enc.name} x{PROXY_TOKENIZER_MARGIN} (proxy tokenizer)",
        margin=PROXY_TOKENIZER_MARGIN,
        _encode=_encode,
    )


@dataclass
class BudgetFit:
    """Outcome of fitting one request into the input budget."""

    messages: List[str]
    tokens: int
    level: int
    steps: List[str]
    info: Dict[str, Any]
    within_target: bool


def fit_to_budget(
    render: Callable[[int], Optional[Tuple[List[str], str, Dict[str, Any]]]],
    counter: TokenCounter,
    *,
    max_input_tokens: int,
    target_input_tokens: int,
) -> BudgetFit:
    """
    Render levels 0, 1, 2, … until the request fits ``target_input_tokens``.

    ``render(level)`` returns ``(messages, step_name, info)`` for that level, or
    None when no smaller rendering exists. The first rendering within target
    wins; otherwise the smallest rendering within ``max_input_tokens``.
    ``steps`` lists only the levels that changed the request.
    Raises ``PromptBudgetError`` when nothing fits the hard limit.
    """
    steps: List[str] = []
    best: Optional[BudgetFit] = None
    previous: Optional[List[str]] = None
    level = 0
    while True:
        rendered = render(level)
        if rendered is None:
            break
        messages, step, info = rendered
        if previous is not None and list(messages) != previous:
            steps.append(step)  # only steps that actually removed content
        previous = list(messages)
        tokens = counter.count_request(messages)
        if best is None or tokens < best.tokens:
            best = BudgetFit(
                messages=list(messages),
                tokens=tokens,
                level=level,
                steps=list(steps),
                info=dict(info),
                within_target=tokens <= int(target_input_tokens),
            )
        if tokens <= int(target_input_tokens):
            return best
        level += 1
    if best is None:
        raise ValueError("render(0) must produce a request")
    if best.tokens > int(max_input_tokens):
        raise PromptBudgetError(
            f"LLM request needs {best.tokens} input tokens ({counter.method}) after "
            f"{len(steps)} truncation step(s); hard limit is {int(max_input_tokens)}",
            best,
        )
    return best


__all__ = [
    "BudgetFit",
    "PromptBudgetError",
    "TokenCounter",
    "fit_to_budget",
    "heuristic_token_count",
    "make_token_counter",
]

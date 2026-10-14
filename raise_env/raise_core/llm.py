"""
LLM client for reward-code generation.

Thin provider wrappers (Groq / local vLLM / Ollama OpenAI-compatible / scripted /
seed). Text in, raw completion strings out — no domain logic in the client.

Primary Stage I API:
    complete(prompt, max_tokens=...) -> str
    generate(prompt, n, max_tokens=...) -> list[str]
"""

from __future__ import annotations

import ast
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

from raise_core.key_manager import (
    GroqKeyManager,
    _is_non_retryable_client_error,
    finish_reason,
    get_key_manager,
    groq_key_pool_available,
    resolve_single_groq_api_key,
)
from raise_core.llm_errors import (
    CompletionTruncatedError,
    EmptyCompletionError,
    GroqRateLimitError,
)

# Shared by Groq / vLLM defaults and feedback-card token-count tests.
DEFAULT_CLIENT_SYSTEM = (
    "You are a reward-function designer. Put the code in exactly one "
    "```python fenced block; any explanation must come before the fence."
)


class LLMClient(ABC):
    """Minimal text-completion interface."""

    provider: str = ""
    # Filled by providers after each ``complete`` (usage / finish_reason / latency).
    last_usage: Optional[Dict[str, Any]] = None

    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, n: int = 1, *, max_tokens: Optional[int] = None) -> List[str]:
        """Return ``n`` raw completion strings for the same prompt."""
        if n <= 0:
            raise ValueError("n must be positive")
        return [self.complete(prompt, max_tokens=max_tokens) for _ in range(n)]


def is_recoverable_llm_error(exc: BaseException) -> bool:
    """True for soft LLM failures that Stage I should retry / soft-fail on."""
    if isinstance(
        exc, (CompletionTruncatedError, EmptyCompletionError, GroqRateLimitError)
    ):
        return True
    msg = str(exc).lower()
    return any(
        needle in msg
        for needle in (
            "empty completion",
            "truncated",
            "no closing code fence",
        )
    )


def _usage_from_response(
    response: Any, *, latency_s: float, attempts: int = 1
) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "finish_reason": finish_reason(response),
        "latency_s": round(float(latency_s), 3),
        "attempts": int(attempts),
    }


def _gpt_oss_model(model: str) -> bool:
    return "gpt-oss" in str(model or "").lower()


class ScriptedLLMClient(LLMClient):
    """Deterministic stand-in: returns pre-canned completions in order."""

    provider = "scripted"

    def __init__(self, completions: Sequence[str]) -> None:
        if not completions:
            raise ValueError("ScriptedLLMClient requires at least one completion.")
        self._completions = tuple(completions)
        self._index = 0

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        if self._index >= len(self._completions):
            raise IndexError(
                f"ScriptedLLMClient has no remaining completions "
                f"({len(self._completions)} provided)."
            )
        completion = self._completions[self._index]
        self._index += 1
        return completion

    @property
    def remaining(self) -> int:
        return max(0, len(self._completions) - self._index)


class GroqLLMClient(LLMClient):
    """
    Live Groq chat completion.

    Auth (first match wins):
      1. ``api_key=`` constructor argument (single key)
      2. ``raise_env/groq_keys.json`` key pool via :mod:`key_manager`
         (rate-limit rotation; copy from ``groq_keys.json.example``)
      3. ``GROQ_API_KEY`` environment variable (single key)

    A single key runs through the same :class:`GroqKeyManager` retry/wait logic
    as a one-key pool. Requires optional dependency: ``pip install groq``.
    """

    provider = "groq"
    DEFAULT_MODEL = "openai/gpt-oss-120b"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.4,
        max_tokens: int = 1500,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_key_pool: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt or DEFAULT_CLIENT_SYSTEM
        self.last_usage: Optional[Dict[str, Any]] = None
        explicit = (api_key or "").strip() or None
        if explicit is None and use_key_pool and groq_key_pool_available():
            self._key_manager = get_key_manager()
            return
        key = explicit or resolve_single_groq_api_key()
        if not key:
            raise RuntimeError(
                "GroqLLMClient requires GROQ_API_KEY, api_key=..., or "
                "raise_env/groq_keys.json (see groq_keys.json.example)"
            )
        self._key_manager = GroqKeyManager(keys=[key])

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": int(max_tokens) if max_tokens is not None else self.max_tokens,
        }
        if self.reasoning_effort is not None and _gpt_oss_model(self.model):
            create_kwargs["reasoning_effort"] = self.reasoning_effort
        started = time.monotonic()
        try:
            response = self._key_manager.chat_completion(**create_kwargs)
        except ImportError as exc:
            raise RuntimeError(
                "GroqLLMClient requires the 'groq' package. Install with: pip install groq"
            ) from exc
        self.last_usage = _usage_from_response(
            response,
            latency_s=time.monotonic() - started,
            attempts=getattr(self._key_manager, "last_attempts", 1) or 1,
        )
        reason = self.last_usage.get("finish_reason")
        content = response.choices[0].message.content
        if reason == "length":
            raise CompletionTruncatedError(
                "Groq finish_reason=length; completion hit max_tokens "
                "(including any reasoning budget)."
            )
        if not content or not str(content).strip():
            raise EmptyCompletionError("Groq returned an empty completion.")
        return str(content)


class VLLMLLMClient(LLMClient):
    """
    Local gpt-oss-120B (or any chat model) served by vLLM's OpenAI-compatible API.

    Default base URL: ``http://127.0.0.1:8000/v1`` (override with ``base_url``
    or ``VLLM_BASE_URL``). Uses the ``openai`` Python SDK when available,
    otherwise urllib JSON.
    """

    provider = "vllm"
    DEFAULT_MODEL = "gpt-oss-120b"
    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        temperature: float = 0.4,
        max_tokens: int = 1500,
        system_prompt: Optional[str] = None,
        max_attempts: int = 3,
        retry_delay_seconds: float = 3.0,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or self.DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system_prompt = system_prompt or DEFAULT_CLIENT_SYSTEM
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.last_usage: Optional[Dict[str, Any]] = None

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._complete_once(prompt, max_tokens=max_tokens, attempt=attempt)
            except (CompletionTruncatedError, EmptyCompletionError):
                raise
            except Exception as exc:  # noqa: BLE001
                if _is_non_retryable_client_error(exc):
                    raise
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"vLLM LLM failed after {self.max_attempts} attempts. Last error: {last_error}"
        ) from last_error

    def _complete_once(
        self, prompt: str, *, max_tokens: Optional[int] = None, attempt: int = 1
    ) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        token_limit = int(max_tokens) if max_tokens is not None else self.max_tokens
        started = time.monotonic()
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=token_limit,
                **self._request_options(),
            )
            self.last_usage = _usage_from_response(
                response, latency_s=time.monotonic() - started, attempts=attempt
            )
            return self._content_or_raise(response.choices[0], provider="vLLM")
        except ImportError:
            return self._complete_urllib(messages, max_tokens=token_limit, attempt=attempt)

    def _request_options(self) -> dict:
        """Return provider-specific OpenAI-compatible request options."""
        if self.reasoning_effort is not None:
            return {"reasoning_effort": self.reasoning_effort}
        return {}

    @staticmethod
    def _extract_message_text(message) -> str:
        """Read the assistant ``content`` field only (never fall back to reasoning)."""
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content or "")

    def _content_or_raise(self, choice: Any, *, provider: str) -> str:
        reason = getattr(choice, "finish_reason", None)
        if isinstance(choice, dict):
            reason = choice.get("finish_reason", reason)
            message = choice.get("message")
        else:
            message = getattr(choice, "message", None)
        content = self._extract_message_text(message)
        if str(reason) == "length":
            raise CompletionTruncatedError(
                f"{provider} finish_reason=length; completion hit max_tokens."
            )
        if not content.strip():
            raise EmptyCompletionError(f"{provider} returned an empty completion.")
        return content

    def _complete_urllib(
        self, messages: list, *, max_tokens: int, attempt: int = 1
    ) -> str:
        import json
        import urllib.error
        import urllib.request

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self._request_options())
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        self.last_usage = {
            "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
            "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
            "latency_s": round(time.monotonic() - started, 3),
            "attempts": int(attempt),
        }
        return self._content_or_raise(body["choices"][0], provider="vLLM")


class OllamaLLMClient(VLLMLLMClient):
    """
    Local Ollama server (OpenAI-compatible endpoint). No API key required.

    The default qwen3.5:4b model is a reasoning model, so max_tokens leaves
    generous headroom for thinking tokens before the code fence.
    """

    provider = "ollama"
    DEFAULT_MODEL = "qwen3.5:4b"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, **kwargs):
        kwargs.setdefault("model", self.DEFAULT_MODEL)
        if kwargs.get("base_url") is None:
            kwargs["base_url"] = os.environ.get("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        kwargs.setdefault("api_key", "ollama")
        kwargs.setdefault("max_tokens", 4096)
        super().__init__(**kwargs)

    def _request_options(self) -> dict:
        # Qwen reasoning can consume the entire completion budget before code.
        return {"reasoning_effort": "none"}


class SeedVariantLLMClient(LLMClient):
    """
    Emits valid domain ``compute_reward`` variants (no API key).

    ``variant_fn(n)`` comes from the domain prompts module (``seed_variant``);
    without one the client returns the domain seed itself.

    Used for dry-runs / ``--llm seed`` (guarded by ``--allow-seed-llm`` for
    real runs) so the pipeline executes without an LLM.
    """

    provider = "seed"

    def __init__(
        self,
        base_code: Optional[str] = None,
        *,
        variant_fn: Optional[Callable[[int], str]] = None,
    ) -> None:
        self._base = (base_code or "").strip()
        self._variant_fn = variant_fn
        self._n = 0

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        self._n += 1
        if self._variant_fn is not None:
            return self._variant_fn(self._n)
        return f"```python\n{self._base}\n```\n"


def make_llm_client(
    provider: str = "scripted",
    *,
    completions: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
    base_code: Optional[str] = None,
    variant_fn: Optional[Callable[[int], str]] = None,
    **kwargs,
) -> LLMClient:
    """
    Factory: ``provider`` in {``scripted``, ``seed``, ``groq``, ``vllm``, ``ollama``}.

    For ``scripted``, pass ``completions=...``.
    For ``seed``, returns ``SeedVariantLLMClient`` (no API key).
    Optional ``base_code`` / ``variant_fn`` (domain D.5 seed and its
    ``seed_variant``) drive the seed-variant client.
    """
    name = str(provider).strip().lower()
    if name in ("seed", "seed-variant", "d5"):
        return SeedVariantLLMClient(base_code=base_code, variant_fn=variant_fn)
    if name == "scripted":
        if not completions:
            raise ValueError("make_llm_client(scripted) requires completions=...")
        return ScriptedLLMClient(completions)
    if name == "groq":
        return GroqLLMClient(model=model or GroqLLMClient.DEFAULT_MODEL, **kwargs)
    if name in ("vllm", "local", "gpt-oss", "gpt-oss-120b"):
        return VLLMLLMClient(model=model or VLLMLLMClient.DEFAULT_MODEL, **kwargs)
    if name == "ollama":
        return OllamaLLMClient(model=model or OllamaLLMClient.DEFAULT_MODEL, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


# Above the sandbox's max_code_length; anything longer is rejected later anyway.
_MAX_PARSE_CHARS = 50_000


def _safe_parse(text: str) -> Optional[ast.Module]:
    """``ast.parse`` that never raises on hostile LLM output (None instead)."""
    if len(text) > _MAX_PARSE_CHARS:
        return None
    try:
        return ast.parse(text, mode="exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


_FENCE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_all_python_blocks(completion: str) -> List[str]:
    """Return every fenced code block in an LLM reply (may be empty)."""
    text = (completion or "").strip()
    blocks = [m.group(1).strip() for m in _FENCE_BLOCK_RE.finditer(text) if m.group(1).strip()]
    return blocks


def extract_python_code(completion: str, *, max_tokens: Optional[int] = None) -> str:
    """
    Pull Python source from an LLM reply.

    Prefers fenced blocks (```python ... ```); strips leading/trailing prose.
    If multiple blocks exist, returns the first non-empty block.
    """
    text = (completion or "").strip()
    # Fallback when the provider does not expose finish_reason (chars ≈ tokens
    # for dense code; prefer finish_reason=="length" in the client).
    opening = re.search(r"```(?:python\b)?", text, re.IGNORECASE)
    if (
        max_tokens is not None
        and len(text) >= 0.95 * int(max_tokens)
        and opening is not None
        and "```" not in text[opening.end() :]
    ):
        raise CompletionTruncatedError(
            "no closing code fence found; completion may have been truncated mid-reasoning by max_tokens"
        )
    blocks = extract_all_python_blocks(text)
    if blocks:
        return blocks[0]
    if text.startswith("```"):
        # Unclosed fence — strip opening tag and return remainder.
        rest = re.sub(r"^```(?:python)?\s*\n?", "", text, count=1, flags=re.IGNORECASE)
        return rest.strip()
    return text


def split_reward_function_sources(
    completion: str,
    *,
    func_name: str = "compute_reward",
    max_tokens: Optional[int] = None,
) -> List[str]:
    """
    Split a multi-candidate LLM reply into separate reward function sources.

    Used for Appendix D.1 batch generation (one call, several ``def`` bodies).
    """
    blocks = extract_all_python_blocks(completion)
    if not blocks:
        blocks = [extract_python_code(completion, max_tokens=max_tokens)]
    pieces: List[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        tree = _safe_parse(block)
        if tree is None:
            pieces.append(normalize_to_compute_reward(block, func_name))
            continue
        func_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if not func_nodes:
            pieces.append(normalize_to_compute_reward(block, func_name))
            continue
        if len(func_nodes) == 1:
            pieces.append(normalize_to_compute_reward(block, func_name))
            continue
        for node in func_nodes:
            segment = ast.get_source_segment(block, node)
            if segment:
                pieces.append(normalize_to_compute_reward(segment, func_name))
    return pieces


_DEF_RE = re.compile(
    r"^def\s+(cal_reward|seed_reward_func|compute_reward_v\d+|[\w]+_v2)\s*\(",
    re.MULTILINE,
)


def normalize_to_compute_reward(code: str, target_name: str = "compute_reward") -> str:
    """
    Ensure the top-level function is named ``compute_reward(state, memory)``.

    Renames common paper aliases (cal_reward, *_v2, seed_reward_func) and
    upgrades a single-arg ``(state)`` signature to ``(state, memory)`` so
    older completions still validate under the episode-memory contract.
    Also strips redundant ``[0]``/``[1]``/``[-1]`` on known scalar state
    fields (LLMs often treat ``state.robot.px`` like a batched tensor).
    """
    text = code.strip()
    match = re.search(r"^def\s+(\w+)\s*\(", text, re.MULTILINE)
    if not match:
        return strip_redundant_scalar_subscripts(text)
    name = match.group(1)
    if name != target_name:
        text = re.sub(
            rf"^def\s+{re.escape(name)}\s*\(",
            f"def {target_name}(",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    sig = re.search(
        rf"^def\s+{re.escape(target_name)}\s*\(([^)]*)\)",
        text,
        re.MULTILINE,
    )
    if not sig:
        return strip_redundant_scalar_subscripts(text)
    raw_args = [a.strip() for a in sig.group(1).split(",") if a.strip()]
    arg_names = [a.split(":")[0].split("=")[0].strip() for a in raw_args]
    if arg_names == ["state"]:
        text = re.sub(
            rf"^def\s+{re.escape(target_name)}\s*\([^)]*\)",
            f"def {target_name}(state, memory)",
            text,
            count=1,
            flags=re.MULTILINE,
        )
    return strip_redundant_scalar_subscripts(text)


# Scalar CrowdNavRewardState / RobotState / HumanState fields — never arrays.
# ``action`` is deliberately absent (a tuple in continuous mode).
_SCALAR_FIELD_ATTRS = frozenset(
    {
        "px",
        "py",
        "vx",
        "vy",
        "gx",
        "gy",
        "theta",
        "radius",
        "v_pref",
        "collision",
        "reached_goal",
        "timeout",
        "min_separation",
        "discomfort_dist",
        "goal_distance",
        "progress",
        "time_step",
        "global_time",
        "time_limit",
    }
)


def _is_trivial_int_index(slice_node: ast.AST) -> bool:
    """True for ``[0]``, ``[1]``, ``[-1]`` (common LLM tensor-habits)."""
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
        return slice_node.value in (0, 1, -1)
    if (
        isinstance(slice_node, ast.UnaryOp)
        and isinstance(slice_node.op, ast.USub)
        and isinstance(slice_node.operand, ast.Constant)
        and isinstance(slice_node.operand.value, int)
        and slice_node.operand.value == 1
    ):
        return True
    return False


class _StripScalarSubscripts(ast.NodeTransformer):
    """Rewrite ``state.robot.px[0]`` → ``state.robot.px`` (and nested repeats)."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Subscript)
        if not isinstance(node.value, ast.Attribute):
            return node
        if node.value.attr not in _SCALAR_FIELD_ATTRS:
            return node
        if not _is_trivial_int_index(node.slice):
            return node
        return node.value


def strip_redundant_scalar_subscripts(code: str) -> str:
    """
    Drop ``[0]``/``[1]``/``[-1]`` on known float/bool state fields.

    Does **not** touch ``state.action[...]`` (a tuple in continuous mode).
    On parse failure, or when nothing changes, returns ``code`` unchanged so
    LLM comments are preserved (``ast.unparse`` would strip them).
    """
    text = (code or "").strip()
    if not text:
        return text
    tree = _safe_parse(text)
    if tree is None:
        return text
    try:
        before = ast.dump(tree)
        new_tree = _StripScalarSubscripts().visit(tree)
        if ast.dump(new_tree) == before:
            return text
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree)
    except Exception:  # noqa: BLE001 — keep original if transform/unparse fails (incl. RecursionError)
        return text

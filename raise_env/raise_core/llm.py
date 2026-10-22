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
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

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
from raise_core.source_normalize import (  # noqa: F401 — re-export
    normalize_to_compute_reward,
    normalize_to_compute_reward_with_report,
    strip_redundant_scalar_subscripts,
)

# Shared by Groq / vLLM defaults and feedback-card token-count tests.
DEFAULT_CLIENT_SYSTEM = (
    "You are a reward-function designer. Put the code in exactly one "
    "```python fenced block; any explanation must come before the fence."
)

# Groq free/on_demand often enforces tokens-per-minute == per-request cap
# (prompt + max_tokens). Measured org limit: 8000. Soft cap keeps headroom for
# tokenizer mismatch (heuristic vs Groq billable). Override after a TPM upgrade.
_GROQ_ORG_TOKEN_LIMIT = int(os.environ.get("GROQ_ORG_TOKEN_LIMIT", "8000"))
_GROQ_REQUEST_TOKEN_CAP = min(
    int(os.environ.get("GROQ_REQUEST_TOKEN_CAP", "7500")),
    max(512, int(_GROQ_ORG_TOKEN_LIMIT)),
)
# Inflate local prompt estimates so under-counting cannot push Requested > cap.
_GROQ_PROMPT_TOKEN_MARGIN = float(os.environ.get("GROQ_PROMPT_TOKEN_MARGIN", "1.25"))
# Minimum completion budget we still bother sending (else refuse the request).
_GROQ_MIN_COMPLETION_TOKENS = 256


def estimate_groq_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    """Conservative prompt-token estimate (counter × safety margin)."""
    import math

    from raise_core.token_budget import make_token_counter

    counter = make_token_counter("groq")
    raw = counter.count_request([system_prompt, user_prompt])
    margin = max(1.0, float(_GROQ_PROMPT_TOKEN_MARGIN))
    return int(math.ceil(raw * margin))


def groq_margined_input_budget(max_completion_tokens: int) -> int:
    """Max *margined* prompt tokens so prompt+completion stays under the soft cap."""
    cap = max(512, int(_GROQ_REQUEST_TOKEN_CAP))
    out = max(1, int(max_completion_tokens))
    return max(512, cap - out)


def groq_raw_input_budget(max_completion_tokens: int) -> int:
    """Max *raw* (unmargined) prompt tokens for ``fit_to_budget`` counters.

    ``fit_to_budget`` / Stage I use the plain token counter. Clamp/estimate use
    ``raw × margin``. So the raw hard limit must be ``floor(room / margin)`` or
    a "fitted" prompt can still leave no safe completion room after margin.
    """
    import math

    room = groq_margined_input_budget(max_completion_tokens)
    margin = max(1.0, float(_GROQ_PROMPT_TOKEN_MARGIN))
    return max(512, int(math.floor(room / margin)))


def groq_sizing_completion_tokens(
    desired_raw_input_tokens: int, requested_max_tokens: int
) -> int:
    """Completion tokens to *reserve while sizing input*, not the send budget.

    Prefer keeping ``desired_raw_input_tokens`` under the soft cap; shrink the
    reserved completion first. ``GroqLLMClient.complete`` still clamps
    ``max_tokens`` at send time to whatever room remains after the real prompt.

    Without this, a large ``llm.max_tokens`` (e.g. 3500) starves crossover
    cards: raw input room drops to 3200 while truncated prompts need ~3400+.
    """
    import math

    desired = max(512, int(desired_raw_input_tokens))
    requested = max(256, int(requested_max_tokens))
    margin = max(1.0, float(_GROQ_PROMPT_TOKEN_MARGIN))
    margined = int(math.ceil(desired * margin))
    room_for_completion = int(_GROQ_REQUEST_TOKEN_CAP) - margined
    if room_for_completion < 256:
        return 256
    return max(256, min(requested, room_for_completion))


def _clamp_groq_request_budget(
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> int:
    """Shrink ``max_tokens`` so prompt+output stay under the soft TPM request cap.

    Last line of defense for every Groq ``complete``: never request more than
    ``_GROQ_REQUEST_TOKEN_CAP`` (≤ org limit) by our margined estimate.
    """
    from raise_core.token_budget import BudgetFit, PromptBudgetError

    prompt_tokens = estimate_groq_prompt_tokens(system_prompt, user_prompt)
    cap = max(512, int(_GROQ_REQUEST_TOKEN_CAP))
    out = max(1, int(max_tokens))
    if prompt_tokens + out <= cap:
        return out
    shrunk = cap - prompt_tokens
    if shrunk < _GROQ_MIN_COMPLETION_TOKENS:
        raise PromptBudgetError(
            f"Groq request prompt is ~{prompt_tokens} tokens (with safety margin); "
            f"soft cap {cap} (org ≤{_GROQ_ORG_TOKEN_LIMIT}) leaves no room for a "
            f"completion (need >={_GROQ_MIN_COMPLETION_TOKENS}). "
            f"Shrink selection.max_input_tokens or raise GROQ_REQUEST_TOKEN_CAP "
            f"after a TPM upgrade.",
            BudgetFit(
                messages=[system_prompt, user_prompt],
                tokens=prompt_tokens,
                level=0,
                steps=["groq_tpm_cap"],
                info={"cap": cap, "org_limit": int(_GROQ_ORG_TOKEN_LIMIT)},
                within_target=False,
            ),
        )
    if shrunk < out:
        logger.warning(
            "Groq max_tokens clamped %d -> %d (prompt≈%d margined, cap=%d, org≤%d)",
            out,
            shrunk,
            prompt_tokens,
            cap,
            int(_GROQ_ORG_TOKEN_LIMIT),
        )
    return int(shrunk)


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
    response: Any,
    *,
    latency_s: float,
    attempts: int = 1,
    attempt_errors: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    out: Dict[str, Any] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "finish_reason": finish_reason(response),
        "latency_s": round(float(latency_s), 3),
        "attempts": int(attempts),
    }
    if attempt_errors:
        out["attempt_errors"] = list(attempt_errors)
    return out


def _gpt_oss_model(model: str) -> bool:
    return "gpt-oss" in str(model or "").lower()


def _looks_like_unsupported_seed(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "seed" in msg and any(
        needle in msg
        for needle in (
            "unexpected",
            "unknown",
            "unsupported",
            "invalid",
            "not supported",
            "extra",
        )
    )


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
        # gpt-oss reasoning tokens count against max_tokens. Default leaves
        # headroom under Groq on_demand TPM (often 8000 = prompt + max_tokens).
        # Override with llm.max_tokens / --set llm.max_tokens=…
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_key_pool: bool = True,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.seed = int(seed) if seed is not None else None
        self.system_prompt = system_prompt or DEFAULT_CLIENT_SYSTEM
        self.last_usage: Optional[Dict[str, Any]] = None
        self._seed_unsupported_logged = False
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
        out_budget = int(max_tokens) if max_tokens is not None else int(self.max_tokens)
        out_budget = _clamp_groq_request_budget(
            system_prompt=str(self.system_prompt or ""),
            user_prompt=str(prompt or ""),
            max_tokens=out_budget,
        )
        create_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": out_budget,
        }
        if self.reasoning_effort is not None and _gpt_oss_model(self.model):
            create_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            create_kwargs["seed"] = self.seed
        started = time.monotonic()
        try:
            response = self._chat_with_optional_seed(create_kwargs)
        except ImportError as exc:
            raise RuntimeError(
                "GroqLLMClient requires the 'groq' package. Install with: pip install groq"
            ) from exc
        self.last_usage = _usage_from_response(
            response,
            latency_s=time.monotonic() - started,
            attempts=getattr(self._key_manager, "last_attempts", 1) or 1,
            attempt_errors=getattr(self._key_manager, "last_errors", None) or None,
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

    def _chat_with_optional_seed(self, create_kwargs: Dict[str, Any]) -> Any:
        try:
            return self._key_manager.chat_completion(**create_kwargs)
        except Exception as exc:  # noqa: BLE001
            if "seed" not in create_kwargs or not _looks_like_unsupported_seed(exc):
                raise
            if not self._seed_unsupported_logged:
                logger.warning(
                    "Groq create() rejected seed=%s; retrying without seed",
                    create_kwargs.get("seed"),
                )
                self._seed_unsupported_logged = True
            create_kwargs = dict(create_kwargs)
            create_kwargs.pop("seed", None)
            self.seed = None
            return self._key_manager.chat_completion(**create_kwargs)


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
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or self.DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.seed = int(seed) if seed is not None else None
        self.system_prompt = system_prompt or DEFAULT_CLIENT_SYSTEM
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.last_usage: Optional[Dict[str, Any]] = None
        self._seed_unsupported_logged = False

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
            opts = self._request_options()
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=token_limit,
                    **opts,
                )
            except Exception as exc:  # noqa: BLE001
                if "seed" in opts and _looks_like_unsupported_seed(exc):
                    if not self._seed_unsupported_logged:
                        logger.warning(
                            "vLLM create() rejected seed=%s; retrying without seed",
                            opts.get("seed"),
                        )
                        self._seed_unsupported_logged = True
                    self.seed = None
                    opts = self._request_options()
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=token_limit,
                        **opts,
                    )
                else:
                    raise
            self.last_usage = _usage_from_response(
                response, latency_s=time.monotonic() - started, attempts=attempt
            )
            return self._content_or_raise(response.choices[0], provider="vLLM")
        except ImportError:
            return self._complete_urllib(messages, max_tokens=token_limit, attempt=attempt)

    def _request_options(self) -> dict:
        """Return provider-specific OpenAI-compatible request options."""
        opts: Dict[str, Any] = {}
        if self.reasoning_effort is not None:
            opts["reasoning_effort"] = self.reasoning_effort
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts

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
        opts = super()._request_options()
        opts["reasoning_effort"] = "none"
        return opts


# raise_core/llm.py → raise_env/ (same place as groq_keys.json)
_RAISE_ENV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_XAI_KEY_PATH = os.path.join(_RAISE_ENV_ROOT, "xai_key")
DEFAULT_XAI_KEYS_JSON = os.path.join(_RAISE_ENV_ROOT, "xai_keys.json")


def resolve_xai_api_key(explicit: Optional[str] = None) -> str:
    """Resolve xAI API key from arg, env, ``xai_key``, or ``xai_keys.json``."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    env = (os.environ.get("XAI_API_KEY") or os.environ.get("XAI_KEY") or "").strip()
    if env:
        return env
    key_path = os.environ.get("XAI_KEY_PATH", DEFAULT_XAI_KEY_PATH)
    if os.path.isfile(key_path):
        text = open(key_path, encoding="utf-8").read().strip()
        # Allow a bare key, or "XAI_API_KEY=..." / "key: ..."
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                line = line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.lower().startswith("xai-") or line.startswith("xai"):
                return line
            if len(line) > 20:
                return line
        if text and not text.startswith("#"):
            return text.splitlines()[0].strip()
    json_path = os.environ.get("XAI_KEYS_PATH", DEFAULT_XAI_KEYS_JSON)
    if os.path.isfile(json_path):
        data = json.loads(open(json_path, encoding="utf-8").read())
        keys = data.get("keys") or data.get("api_keys") or []
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            k = str(k).strip()
            if k and not k.startswith("xai_REPLACE") and "YOUR" not in k.upper():
                return k
    raise RuntimeError(
        "xAI API key not found. Create raise_env/xai_key (one line: your key), "
        "or set XAI_API_KEY, or copy xai_key.example → xai_key."
    )


class XaiLLMClient(VLLMLLMClient):
    """xAI Grok via the OpenAI-compatible API (``https://api.x.ai/v1``)."""

    provider = "xai"
    DEFAULT_MODEL = "grok-4.20-0309-reasoning"
    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(self, **kwargs):
        kwargs.setdefault("model", self.DEFAULT_MODEL)
        if kwargs.get("base_url") is None:
            kwargs["base_url"] = os.environ.get("XAI_BASE_URL", self.DEFAULT_BASE_URL)
        api_key = kwargs.pop("api_key", None)
        kwargs["api_key"] = resolve_xai_api_key(api_key)
        kwargs.setdefault("max_tokens", 2000)
        super().__init__(**kwargs)


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
    Factory: ``provider`` in
    {``scripted``, ``seed``, ``groq``, ``xai``, ``vllm``, ``ollama``}.

    For ``scripted``, pass ``completions=...``.
    For ``seed``, returns ``SeedVariantLLMClient`` (no API key).
    For ``xai``, reads ``raise_env/xai_key`` or ``XAI_API_KEY``.
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
    if name in ("xai", "grok"):
        return XaiLLMClient(model=model or XaiLLMClient.DEFAULT_MODEL, **kwargs)
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


"""
LLM client for reward-code generation.

Thin provider wrappers (Groq / local vLLM OpenAI-compatible / scripted).
Interface mirrors mobile_robot_env.rewards.generators: no crowd-nav logic
inside the client — only text in, raw completion strings out.

Primary API for EvoNav Stage I:
    generate(prompt: str, n: int) -> list[str]
"""

from __future__ import annotations

import ast
import os
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

class LLMClient(ABC):
    """Minimal text-completion interface (sister-project compatible)."""

    # Used by produce_final_ranking to decide LLM vs lex fallback.
    provider: str = ""

    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

    def generate(self, prompt: str, n: int = 1, *, max_tokens: Optional[int] = None) -> List[str]:
        """Return ``n`` raw completion strings for the same prompt."""
        if n <= 0:
            raise ValueError("n must be positive")
        return [self.complete(prompt, max_tokens=max_tokens) for _ in range(n)]


class CompletionTruncatedError(RuntimeError):
    """Raised when a fenced completion strongly appears truncated."""


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
    Live Groq chat completion (same design as mobile_robot_env GroqLLMClient).

    Auth (first match wins):
      1. ``api_key=`` constructor argument (single key)
      2. ``GROQ_API_KEY`` environment variable (single key)
      3. ``evonav_env/groq_keys.json`` key pool via :mod:`key_manager`
         (rate-limit rotation; copy from ``groq_keys.json.example``)

    Requires optional dependency: ``pip install groq``.
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
        max_attempts: int = 3,
        retry_delay_seconds: float = 3.0,
        use_key_pool: bool = True,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a reward-function designer. Reply with one Python code block only."
        )
        if int(max_attempts) < 1:
            raise ValueError("max_attempts must be >= 1")
        if float(retry_delay_seconds) < 0.0:
            raise ValueError("retry_delay_seconds must be >= 0")
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)

        self._key_manager = None
        explicit = (api_key or "").strip() or None
        if explicit:
            self.api_key = explicit
        else:
            from crowd_nav.reward_search.key_manager import (
                groq_key_pool_available,
                resolve_single_groq_api_key,
            )

            # Prefer multi-key pool (rotation + pacing) when groq_keys.json exists.
            if use_key_pool and groq_key_pool_available():
                from crowd_nav.reward_search.key_manager import get_key_manager

                self._key_manager = get_key_manager()
                self.api_key = None
            else:
                env_key = resolve_single_groq_api_key()
                self.api_key = env_key
                if not self.api_key:
                    raise RuntimeError(
                        "GroqLLMClient requires GROQ_API_KEY, api_key=..., or "
                        "evonav_env/groq_keys.json (see groq_keys.json.example)"
                    )

    @staticmethod
    def _is_non_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        if status == 404 or "model_not_found" in text or "does not exist" in text:
            return True
        if status == 400 and "model" in text:
            return True
        return False

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        if not self.api_key and self._key_manager is None:
            raise RuntimeError(
                "GroqLLMClient requires GROQ_API_KEY, api_key=..., or "
                "evonav_env/groq_keys.json (see groq_keys.json.example)"
            )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens) if max_tokens is not None else self.max_tokens,
        }

        if self._key_manager is not None:
            # Key pool owns 429 rotation + transient network retries.
            try:
                response = self._key_manager.chat_completion(**create_kwargs)
            except ImportError as exc:
                raise RuntimeError(
                    "GroqLLMClient requires the 'groq' package. "
                    "Install with: pip install groq"
                ) from exc
            content = response.choices[0].message.content
            if not content or not str(content).strip():
                raise RuntimeError("Groq returned an empty completion.")
            return str(content)

        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError(
                "GroqLLMClient requires the 'groq' package. "
                "Install with: pip install groq"
            ) from exc

        from crowd_nav.reward_search.key_manager import (
            DEFAULT_REQUEST_TIMEOUT,
            _is_rate_limit_error,
            _is_transient_error,
            get_groq_pacer,
        )

        pacer = get_groq_pacer()
        # Disable SDK auto-retry; we pace explicitly to avoid 429 storms.
        client = Groq(
            api_key=self.api_key,
            max_retries=0,
            timeout=DEFAULT_REQUEST_TIMEOUT,
        )
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            pacer.wait_turn()
            try:
                response = client.chat.completions.create(**create_kwargs)
                pacer.observe_response(response)
                content = response.choices[0].message.content
                if not content or not str(content).strip():
                    raise RuntimeError("Groq returned an empty completion.")
                return str(content)
            except Exception as exc:  # noqa: BLE001 — retry transient errors
                last_error = exc
                if _is_rate_limit_error(exc):
                    retry_after = getattr(exc, "retry_after", None)
                    if retry_after is not None:
                        try:
                            pacer._hold_until = max(
                                pacer._hold_until, time.time() + float(retry_after)
                            )
                        except (TypeError, ValueError):
                            pass
                if self._is_non_retryable(exc):
                    raise RuntimeError(
                        f"Groq LLM non-retryable error for model={self.model!r}: {exc}"
                    ) from exc
                if not _is_transient_error(exc) and not _is_rate_limit_error(exc):
                    raise
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"Groq LLM failed after {self.max_attempts} attempts. Last error: {last_error}"
        ) from last_error


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
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or self.DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a reward-function designer. Reply with one Python code block only."
        )
        self.max_attempts = int(max_attempts)
        self.retry_delay_seconds = float(retry_delay_seconds)

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._complete_once(prompt, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"vLLM LLM failed after {self.max_attempts} attempts. Last error: {last_error}"
        ) from last_error

    def _complete_once(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        token_limit = int(max_tokens) if max_tokens is not None else self.max_tokens
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
            content = self._extract_message_text(response.choices[0].message)
            if not content or not str(content).strip():
                raise RuntimeError("vLLM returned an empty completion.")
            return str(content)
        except ImportError:
            return self._complete_urllib(messages, max_tokens=token_limit)

    def _request_options(self) -> dict:
        """Return provider-specific OpenAI-compatible request options."""
        return {}

    @staticmethod
    def _extract_message_text(message) -> str:
        """Read normal and reasoning-model message shapes from SDK responses."""
        if isinstance(message, dict):
            content = message.get("content")
            reasoning = message.get("reasoning")
        else:
            content = getattr(message, "content", None)
            reasoning = getattr(message, "reasoning", None)
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content or reasoning or "")

    def _complete_urllib(self, messages: list, *, max_tokens: int) -> str:
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
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        content = self._extract_message_text(body["choices"][0]["message"])
        if not content or not str(content).strip():
            raise RuntimeError("vLLM returned an empty completion.")
        return str(content)


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
    Always emits a valid ``compute_reward`` variant of the D.5 seed.

    Used for dry-runs / ``--llm seed`` so Algorithm 1 can execute without
    an API key or a huge scripted completion list.
    """

    provider = "seed"

    def __init__(self, base_code: Optional[str] = None) -> None:
        from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION

        self._base = (base_code or D5_SEED_FUNCTION).strip()
        self._n = 0

    def complete(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        self._n += 1
        pot = 2.0 + 0.05 * ((self._n % 20) - 10)
        # Tiny localized tweak so Stage I/II/III see distinct codes.
        code = (
            "```python\n"
            "def compute_reward(state, memory):\n"
            "    success_reward = 10.0\n"
            "    collision_penalty = -20.0\n"
            f"    pot_factor = {pot:.4f}\n"
            "    if state.reaching_goal:\n"
            "        return float(success_reward)\n"
            "    if state.collision:\n"
            "        return float(collision_penalty)\n"
            "    dist = ((state.robot.px - state.robot.gx) ** 2 + "
            "(state.robot.py - state.robot.gy) ** 2) ** 0.5\n"
            "    prev = memory.get('prev_dist')\n"
            "    if prev is None:\n"
            "        memory['prev_dist'] = dist\n"
            "        return float(0.0)\n"
            "    progress = float(prev) - dist\n"
            "    memory['prev_dist'] = dist\n"
            "    return float(pot_factor * progress)\n"
            "```\n"
        )
        return code


def make_llm_client(
    provider: str = "scripted",
    *,
    completions: Optional[Sequence[str]] = None,
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Factory: ``provider`` in {``scripted``, ``seed``, ``groq``, ``vllm``, ``ollama``}.

    For ``scripted``, pass ``completions=...``.
    For ``seed``, returns ``SeedVariantLLMClient`` (no API key).
    """
    name = str(provider).strip().lower()
    if name in ("seed", "seed-variant", "d5"):
        return SeedVariantLLMClient()
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
    opening = re.search(r"```python\b", text, re.IGNORECASE)
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
    import ast

    blocks = extract_all_python_blocks(completion)
    if not blocks:
        blocks = [extract_python_code(completion, max_tokens=max_tokens)]
    pieces: List[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            tree = ast.parse(block)
        except SyntaxError:
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
    Also strips redundant ``[0]``/``[1]``/``[-1]`` on known scalar RewardState
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


# Scalar RewardState / HumanObservable / RobotRewardState fields — never arrays.
_SCALAR_FIELD_ATTRS = frozenset(
    {
        "px",
        "py",
        "vx",
        "vy",
        "radius",
        "gx",
        "gy",
        "v_pref",
        "dmin",
        "discomfort_dist",
        "time_step",
        "global_time",
        "time_limit",
        "collision",
        "reaching_goal",
        "timeout",
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
    Drop ``[0]``/``[1]``/``[-1]`` on known float/bool RewardState fields.

    Does **not** touch ``state.action[...]`` (action may be a tuple/ActionXY).
    On parse failure, returns ``code`` unchanged.
    """
    text = (code or "").strip()
    if not text:
        return text
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError:
        return text
    new_tree = _StripScalarSubscripts().visit(tree)
    ast.fix_missing_locations(new_tree)
    try:
        return ast.unparse(new_tree)
    except Exception:  # noqa: BLE001 — keep original if unparse fails
        return text

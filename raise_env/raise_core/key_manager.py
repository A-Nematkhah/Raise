"""
Groq API key pool (rate-limit rotation).

Loads keys from ``raise_env/groq_keys.json`` (or ``GROQ_KEYS_PATH``).
Falls back to a single ``GROQ_API_KEY`` env var when the JSON file is absent.

Copy ``groq_keys.json.example`` → ``groq_keys.json`` and fill in your keys.
Never commit ``groq_keys.json``.

Usage::

    from raise_core.key_manager import get_key_manager

    manager = get_key_manager()
    response = manager.chat_completion(
        model="openai/gpt-oss-120b",
        messages=[...],
        max_tokens=1500,
        temperature=0.4,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

COOLDOWN_SECONDS = 60
DEFAULT_MIN_REQUEST_INTERVAL = float(os.environ.get("GROQ_MIN_REQUEST_INTERVAL", "3.0"))
# SSL / proxy handshakes to api.groq.com can stall; default SDK timeout is too tight.
DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT_SECONDS", "120"))
# Outer attempts across the key pool for 429 + transient network failures.
DEFAULT_MAX_ATTEMPTS = int(os.environ.get("GROQ_MAX_ATTEMPTS", "8"))
DEFAULT_TRANSIENT_BACKOFF = float(os.environ.get("GROQ_TRANSIENT_BACKOFF", "2.0"))
_PLACEHOLDER_PREFIXES = ("gsk_REPLACE", "gsk_your_", "YOUR_KEY")

# raise_core/key_manager.py → raise_env/ (where groq_keys.json lives)
_RAISE_ENV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_GROQ_KEYS_PATH = os.path.join(_RAISE_ENV_ROOT, "groq_keys.json")


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


def _is_transient_error(exc: Exception) -> bool:
    """
    Network / gateway flakes that should be retried (not auth / bad-request).

    Includes httpx/httpcore ConnectTimeout through proxies, Groq APITimeoutError,
    and typical 5xx / 408 responses.
    """
    if _is_rate_limit_error(exc):
        return True
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 425, 500, 502, 503, 504):
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    needles = (
        "timeout",
        "timed out",
        "connecterror",
        "connection reset",
        "connection aborted",
        "connection refused",
        "temporarily unavailable",
        "handshake",
        "remoteprotocolerror",
        "readerror",
        "writeerror",
        "network",
    )
    if any(n in name for n in ("timeout", "connect", "network", "unavailable")):
        return True
    return any(n in text for n in needles)


def _is_non_retryable_client_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status in (401, 403):
        return True
    if status == 404 or "model_not_found" in text or "does not exist" in text:
        return True
    if status == 400 and "model" in text:
        return True
    return False


def _is_placeholder_key(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return True
    return any(k.startswith(p) for p in _PLACEHOLDER_PREFIXES)


def _response_headers(response: Any) -> dict:
    """Best-effort extraction of HTTP headers from a Groq SDK response."""
    raw = getattr(response, "_response", None) or getattr(response, "response", None)
    if raw is None:
        return {}
    headers = getattr(raw, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k).lower(): v for k, v in dict(headers).items()}
    except Exception:  # noqa: BLE001
        return {}


class GroqRateLimitPacer:
    """
    Proactive spacing between Groq requests using min-interval + response headers.

    Observes ``retry-after`` and ``x-ratelimit-remaining-*`` when present so Gen0
    batching does not spend most of its wall time in reactive 429 backoff.
    """

    def __init__(self, *, min_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL) -> None:
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._last_request_at = 0.0
        self._hold_until = 0.0

    def wait_turn(self) -> None:
        now = time.time()
        target = max(self._last_request_at + self.min_interval_seconds, self._hold_until)
        if now < target:
            delay = target - now
            logger.info("Groq pacing: sleeping %.2fs before request", delay)
            time.sleep(delay)
        self._last_request_at = time.time()

    def observe_response(self, response: Any) -> None:
        headers = _response_headers(response)
        if not headers:
            return
        retry_after = headers.get("retry-after") or headers.get("retry_after")
        if retry_after is not None:
            try:
                self._hold_until = max(self._hold_until, time.time() + float(retry_after))
            except (TypeError, ValueError):
                pass
        for key in (
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-remaining-tokens-day",
        ):
            remaining = headers.get(key)
            if remaining is None:
                continue
            try:
                if int(remaining) <= 1:
                    self._hold_until = max(
                        self._hold_until,
                        time.time() + self.min_interval_seconds * 2.0,
                    )
            except (TypeError, ValueError):
                continue


_pacer_singleton: Optional[GroqRateLimitPacer] = None


def get_groq_pacer() -> GroqRateLimitPacer:
    global _pacer_singleton
    if _pacer_singleton is None:
        _pacer_singleton = GroqRateLimitPacer()
    return _pacer_singleton


def resolve_groq_keys_path() -> str:
    return os.environ.get("GROQ_KEYS_PATH", DEFAULT_GROQ_KEYS_PATH)


def load_groq_keys(keys_path: Optional[str] = None) -> List[str]:
    """
    Load Groq keys from JSON ``{"keys": ["gsk_...", ...]}``.

    Returns an empty list if the file is missing (caller may fall back to env).
    """
    path = keys_path or resolve_groq_keys_path()
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("keys", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path}: 'keys' must be a JSON array")
    return [str(k).strip() for k in raw if k and not _is_placeholder_key(str(k))]


def resolve_single_groq_api_key() -> Optional[str]:
    """``GROQ_API_KEY`` env var, if set and non-empty."""
    key = (os.environ.get("GROQ_API_KEY") or "").strip()
    return key or None


class GroqKeyManager:
    """Rotate through a pool of Groq keys on HTTP 429 / rate-limit errors."""

    def __init__(self, keys_path: Optional[str] = None, *, keys: Optional[List[str]] = None):
        self.keys_path = keys_path or resolve_groq_keys_path()
        if keys is not None:
            self.keys = [k for k in keys if k and not _is_placeholder_key(k)]
        else:
            self.keys = load_groq_keys(self.keys_path)
        if not self.keys:
            raise RuntimeError(
                f"No Groq API keys available. Set GROQ_API_KEY or create "
                f"{self.keys_path} from groq_keys.json.example"
            )
        self.index = 0
        self.cooldown_until = {key: 0.0 for key in self.keys}

    def _current_key(self) -> str:
        return self.keys[self.index]

    def _advance(self) -> None:
        self.index = (self.index + 1) % len(self.keys)

    def _mark_rate_limited(self, key: str) -> None:
        self.cooldown_until[key] = time.time() + COOLDOWN_SECONDS
        logger.warning(
            "Groq key ...%s rate-limited; cooling down %ds",
            key[-6:],
            COOLDOWN_SECONDS,
        )

    def _select_available_key(self) -> str:
        now = time.time()
        for _ in range(len(self.keys)):
            key = self._current_key()
            if self.cooldown_until[key] <= now:
                return key
            self._advance()
        return self._current_key()

    def get_client(self) -> Tuple[Any, str]:
        from groq import Groq

        key = self._select_available_key()
        # max_retries=0: we own pacing / key rotation / transient backoff.
        return (
            Groq(
                api_key=key,
                max_retries=0,
                timeout=DEFAULT_REQUEST_TIMEOUT,
            ),
            key,
        )

    def chat_completion(self, **kwargs: Any) -> Any:
        """
        Same signature as ``client.chat.completions.create(**kwargs)``.

        Retries on HTTP 429 (rotate key + cooldown) and on transient network
        errors (ConnectTimeout / APITimeout / 5xx), including proxy SSL stalls.
        Non-retryable client errors (401/403/bad model) propagate immediately.
        """
        pacer = get_groq_pacer()
        last_exception: Optional[Exception] = None
        max_attempts = max(DEFAULT_MAX_ATTEMPTS, len(self.keys) * 2)
        for attempt in range(1, max_attempts + 1):
            pacer.wait_turn()
            client, key = self.get_client()
            try:
                response = client.chat.completions.create(**kwargs)
                pacer.observe_response(response)
                return response
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                if _is_non_retryable_client_error(exc):
                    raise
                if _is_rate_limit_error(exc):
                    retry_after = getattr(exc, "retry_after", None)
                    if retry_after is not None:
                        try:
                            pacer._hold_until = max(
                                pacer._hold_until, time.time() + float(retry_after)
                            )
                        except (TypeError, ValueError):
                            pass
                    self._mark_rate_limited(key)
                    self._advance()
                    continue
                if _is_transient_error(exc):
                    delay = min(
                        60.0,
                        DEFAULT_TRANSIENT_BACKOFF * (2.0 ** (attempt - 1)),
                    )
                    logger.warning(
                        "Groq transient error on key ...%s "
                        "(attempt %d/%d): %s; retrying in %.1fs",
                        key[-6:],
                        attempt,
                        max_attempts,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
                    # Another key may avoid a sticky broken proxy path.
                    self._advance()
                    continue
                raise
        raise RuntimeError(
            f"All {len(self.keys)} Groq API keys are rate-limited or failing "
            f"after {max_attempts} attempts. Last error: {last_exception}"
        )


_manager_singleton: Optional[GroqKeyManager] = None


def get_key_manager(*, keys_path: Optional[str] = None, force_reload: bool = False) -> GroqKeyManager:
    """Lazily build one ``GroqKeyManager`` per process."""
    global _manager_singleton
    if force_reload or _manager_singleton is None:
        _manager_singleton = GroqKeyManager(keys_path=keys_path)
    return _manager_singleton


def groq_key_pool_available(keys_path: Optional[str] = None) -> bool:
    """True if ``groq_keys.json`` exists and has at least one real key."""
    try:
        return len(load_groq_keys(keys_path)) > 0
    except (OSError, ValueError, json.JSONDecodeError):
        return False

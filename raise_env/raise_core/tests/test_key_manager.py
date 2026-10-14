"""Tests for Groq key pool (no live API calls)."""

from __future__ import annotations

import json
import logging
import types
from unittest.mock import MagicMock, patch

import pytest

from raise_core import key_manager as km
from raise_core.key_manager import (
    GroqKeyManager,
    GroqRateLimitPacer,
    _is_non_retryable_client_error,
    _is_rate_limit_error,
    _is_transient_error,
    _retry_after_seconds,
    load_groq_keys,
)
from raise_core.llm import GroqLLMClient, is_recoverable_llm_error
from raise_core.llm_errors import EmptyCompletionError, GroqRateLimitError

KEY_A = "gsk_test_key_aaaaaaaaaaaaaaaa"
KEY_B = "gsk_test_key_bbbbbbbbbbbbbbbb"


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(float(seconds))
        self.now += float(seconds)


@pytest.fixture
def clock(monkeypatch):
    fake = _FakeClock()
    monkeypatch.setattr(km, "time", types.SimpleNamespace(time=fake.time, sleep=fake.sleep))
    monkeypatch.setattr(km, "_pacer_singleton", GroqRateLimitPacer(min_interval_seconds=0.0))
    return fake


class _RateLimited(Exception):
    status_code = 429

    def __init__(self, retry_after=None):
        super().__init__("Error code: 429 - rate limit reached")
        headers = {} if retry_after is None else {"retry-after": str(retry_after)}
        self.response = types.SimpleNamespace(headers=headers)


def _reply(content="ok", finish="stop"):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content), finish_reason=finish)]
    )


def _pool_with(manager, side_effect):
    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect
    manager.get_client = lambda: (client, manager._select_available_key())
    return client


def test_is_rate_limit_error():
    class E429(Exception):
        status_code = 429

    assert _is_rate_limit_error(E429("x"))
    assert _is_rate_limit_error(RuntimeError("HTTP 429 too many"))


def test_is_transient_timeout_and_connect():
    class APITimeoutError(Exception):
        pass

    class ConnectTimeout(Exception):
        pass

    assert _is_transient_error(APITimeoutError("Request timed out."))
    assert _is_transient_error(ConnectTimeout("handshake operation timed out"))
    assert _is_transient_error(RuntimeError("Connection reset by peer"))

    class E502(Exception):
        status_code = 502

    assert _is_transient_error(E502("bad gateway"))


def test_non_retryable_auth():
    class E401(Exception):
        status_code = 401

    assert _is_non_retryable_client_error(E401("unauthorized"))
    assert not _is_rate_limit_error(E401("unauthorized"))


def test_load_keys_filters_placeholders(tmp_path):
    path = tmp_path / "groq_keys.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    "gsk_REPLACE_ME",
                    "gsk_real_key_abcdefghijklmnop",
                    "",
                ]
            }
        ),
        encoding="utf-8",
    )
    keys = load_groq_keys(str(path))
    assert keys == ["gsk_real_key_abcdefghijklmnop"]


def test_manager_requires_real_keys(tmp_path):
    path = tmp_path / "groq_keys.json"
    path.write_text(json.dumps({"keys": ["gsk_REPLACE"]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="No Groq API keys"):
        GroqKeyManager(keys_path=str(path))


def test_manager_accepts_inline_keys():
    m = GroqKeyManager(keys=["gsk_test_key_aaaaaaaaaaaaaaaa"])
    assert m.keys == ["gsk_test_key_aaaaaaaaaaaaaaaa"]


def test_chat_completion_retries_timeout_then_succeeds():
    manager = GroqKeyManager(
        keys=["gsk_test_key_aaaaaaaaaaaaaaaa", "gsk_test_key_bbbbbbbbbbbbbbbb"]
    )

    class APITimeoutError(Exception):
        pass

    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="ok"))]

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        APITimeoutError("Request timed out."),
        ok,
    ]

    with patch.object(manager, "get_client", return_value=(client, manager.keys[0])):
        with patch("raise_core.key_manager.time.sleep"):
            with patch(
                "raise_core.key_manager.DEFAULT_MAX_ATTEMPTS",
                4,
            ):
                out = manager.chat_completion(model="m", messages=[])
    assert out is ok
    assert client.chat.completions.create.call_count == 2


def test_chat_completion_raises_immediately_on_401():
    manager = GroqKeyManager(keys=["gsk_test_key_aaaaaaaaaaaaaaaa"])

    class AuthError(Exception):
        status_code = 401

    client = MagicMock()
    client.chat.completions.create.side_effect = AuthError("nope")

    with patch.object(manager, "get_client", return_value=(client, manager.keys[0])):
        with pytest.raises(AuthError):
            manager.chat_completion(model="m", messages=[])
    assert client.chat.completions.create.call_count == 1


def test_chat_completion_retries_empty_content_then_succeeds():
    manager = GroqKeyManager(
        keys=["gsk_test_key_aaaaaaaaaaaaaaaa", "gsk_test_key_bbbbbbbbbbbbbbbb"]
    )
    empty = MagicMock()
    empty.choices = [MagicMock(message=MagicMock(content=""))]
    ok = MagicMock()
    ok.choices = [MagicMock(message=MagicMock(content="def compute_reward(state, memory):\n    return 0.0\n"))]

    client = MagicMock()
    client.chat.completions.create.side_effect = [empty, ok]

    with patch.object(manager, "get_client", return_value=(client, manager.keys[0])):
        with patch("raise_core.key_manager.time.sleep"):
            with patch("raise_core.key_manager.DEFAULT_MAX_ATTEMPTS", 4):
                out = manager.chat_completion(model="m", messages=[])
    assert out is ok
    assert client.chat.completions.create.call_count == 2


def test_empty_completion_is_transient():
    assert _is_transient_error(RuntimeError("Groq returned an empty completion."))


def test_retry_after_is_read_from_sdk_response_headers():
    assert _retry_after_seconds(_RateLimited(retry_after=7)) == 7.0
    assert _retry_after_seconds(_RateLimited()) is None


def test_rate_limit_cools_key_for_retry_after_then_rotates(clock):
    manager = GroqKeyManager(keys=[KEY_A, KEY_B])
    client = _pool_with(manager, [_RateLimited(retry_after=12), _reply()])
    assert manager.chat_completion(model="m", messages=[]).choices[0].message.content == "ok"
    assert manager.cooldown_until[KEY_A] == pytest.approx(clock.now + 12.0)
    assert client.chat.completions.create.call_count == 2
    assert manager.last_attempts == 2


def test_waits_for_cooldown_when_every_key_is_rate_limited(clock):
    manager = GroqKeyManager(keys=[KEY_A, KEY_B])
    _pool_with(manager, [_RateLimited(retry_after=20), _RateLimited(retry_after=30), _reply()])
    start = clock.now
    assert manager.chat_completion(model="m", messages=[]).choices[0].message.content == "ok"
    assert clock.slept == [pytest.approx(20.0)]
    assert clock.now - start == pytest.approx(20.0)


def test_rate_limit_beyond_wait_budget_stops_the_run(clock):
    manager = GroqKeyManager(keys=[KEY_A], max_wait_seconds=180)
    _pool_with(manager, [_RateLimited(retry_after=100), _RateLimited(retry_after=100), _reply()])
    with pytest.raises(GroqRateLimitError, match="after waiting up to 180s") as info:
        manager.chat_completion(model="m", messages=[])
    assert clock.slept == [pytest.approx(100.0)]
    # Soft-fail the Stage I slot instead of killing a multi-hour run.
    assert is_recoverable_llm_error(info.value)


def test_default_rate_limit_wait_budget_is_ten_minutes():
    assert km.DEFAULT_MAX_WAIT_SECONDS == 600.0
    assert GroqKeyManager(keys=[KEY_A]).max_wait_seconds == 600.0


def test_retry_after_header_absent_uses_default_cooldown(clock):
    manager = GroqKeyManager(keys=[KEY_A])
    _pool_with(manager, [_RateLimited(), _reply()])
    manager.chat_completion(model="m", messages=[])
    assert clock.slept == [pytest.approx(float(km.COOLDOWN_SECONDS))]


def test_logs_name_keys_by_position_not_by_characters(clock, caplog):
    manager = GroqKeyManager(keys=[KEY_A, KEY_B])
    _pool_with(manager, [_RateLimited(retry_after=5), _reply()])
    with caplog.at_level(logging.WARNING, logger=km.logger.name):
        manager.chat_completion(model="m", messages=[])
    assert "#1/2" in caplog.text
    assert KEY_A[-6:] not in caplog.text


def test_repeated_empty_replies_raise_empty_completion_error(clock):
    manager = GroqKeyManager(keys=[KEY_A])
    _pool_with(manager, [_reply(content="")] * 20)
    with patch.object(km, "DEFAULT_MAX_ATTEMPTS", 3):
        with pytest.raises(EmptyCompletionError) as info:
            manager.chat_completion(model="m", messages=[])
    assert is_recoverable_llm_error(info.value)


def test_single_api_key_uses_the_same_wait_logic(clock, monkeypatch):
    llm = GroqLLMClient(api_key=KEY_A)
    assert isinstance(llm._key_manager, GroqKeyManager)
    assert llm._key_manager.keys == [KEY_A]
    _pool_with(llm._key_manager, [_RateLimited(retry_after=9), _reply("```python\nx\n```")])
    assert llm.complete("p").startswith("```python")
    assert clock.slept == [pytest.approx(9.0)]

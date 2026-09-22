"""Tests for Groq key pool (no live API calls)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from crowd_nav.reward_search.key_manager import (
    GroqKeyManager,
    _is_non_retryable_client_error,
    _is_rate_limit_error,
    _is_transient_error,
    load_groq_keys,
)


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
        with patch("crowd_nav.reward_search.key_manager.time.sleep"):
            with patch(
                "crowd_nav.reward_search.key_manager.DEFAULT_MAX_ATTEMPTS",
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

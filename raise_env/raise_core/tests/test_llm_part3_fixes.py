"""Part 3 LLM client / logging fixes: truncation, usage, reasoning_effort, vLLM."""

from __future__ import annotations

import json
import sys
import types

import pytest

from raise_core.llm import (
    DEFAULT_CLIENT_SYSTEM,
    GroqLLMClient,
    VLLMLLMClient,
    is_recoverable_llm_error,
    make_llm_client,
)
from raise_core.llm_call_log import LLMCallLog
from raise_core.llm_errors import CompletionTruncatedError, EmptyCompletionError, GroqRateLimitError
from raise_core.pipeline import RaiseRunConfig


def test_default_client_system_allows_prose_before_fence():
    assert "only" not in DEFAULT_CLIENT_SYSTEM.lower().split("block")[0]
    assert "before the fence" in DEFAULT_CLIENT_SYSTEM


def test_groq_rate_limit_error_is_recoverable():
    assert is_recoverable_llm_error(GroqRateLimitError("waited out"))


def test_groq_finish_reason_length_raises_truncated(monkeypatch):
    llm = GroqLLMClient(api_key="gsk_test_aaaaaaaaaaaaaaaa")

    class _Msg:
        content = "```python\ndef compute_reward(state, memory):\n    return 0.0, {}\n"

    class _Choice:
        message = _Msg()
        finish_reason = "length"

    class _Usage:
        prompt_tokens = 100
        completion_tokens = 1500

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    def _chat(**_kwargs):
        return _Resp()

    monkeypatch.setattr(llm._key_manager, "chat_completion", _chat)
    llm._key_manager.last_attempts = 2
    with pytest.raises(CompletionTruncatedError, match="finish_reason=length"):
        llm.complete("prompt")
    assert llm.last_usage["finish_reason"] == "length"
    assert llm.last_usage["attempts"] == 2
    assert llm.last_usage["prompt_tokens"] == 100


def test_groq_passes_reasoning_effort_for_gpt_oss(monkeypatch):
    llm = GroqLLMClient(
        api_key="gsk_test_aaaaaaaaaaaaaaaa",
        reasoning_effort="high",
        model="openai/gpt-oss-120b",
    )
    captured = {}

    class _Msg:
        content = "ok"

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Resp:
        choices = [_Choice()]
        usage = None

    def _chat(**kwargs):
        captured.update(kwargs)
        return _Resp()

    monkeypatch.setattr(llm._key_manager, "chat_completion", _chat)
    assert llm.complete("p") == "ok"
    assert captured["reasoning_effort"] == "high"


def test_vllm_does_not_retry_permanent_errors(monkeypatch):
    calls = {"n": 0}

    class _Exc(Exception):
        status_code = 400

        def __str__(self):
            return "model not found"

    class FakeCompletions:
        def create(self, **kwargs):
            calls["n"] += 1
            raise _Exc()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    with pytest.raises(_Exc):
        VLLMLLMClient(max_attempts=3, retry_delay_seconds=0).complete("p")
    assert calls["n"] == 1


def test_vllm_empty_content_ignores_reasoning_field(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        finish_reason="stop",
                        message=types.SimpleNamespace(
                            content="",
                            reasoning="def compute_reward(state, memory):\n    return 1.0, {}",
                        ),
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    with pytest.raises(EmptyCompletionError):
        VLLMLLMClient(max_attempts=1).complete("p")


def test_llm_call_log_writes_response(tmp_path):
    log = LLMCallLog(str(tmp_path))
    log.write_response(
        call_id="c1",
        status="ok",
        usage={"prompt_tokens": 10, "completion_tokens": 20, "finish_reason": "stop"},
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["event"] == "response"
    assert rows[0]["prompt_tokens"] == 10
    assert rows[0]["finish_reason"] == "stop"


def test_make_llm_client_forwards_max_tokens_and_reasoning():
    client = make_llm_client(
        "groq",
        api_key="gsk_test_aaaaaaaaaaaaaaaa",
        max_tokens=2000,
        reasoning_effort="low",
    )
    assert isinstance(client, GroqLLMClient)
    assert client.max_tokens == 2000
    assert client.reasoning_effort == "low"


def test_raise_run_config_carries_llm_budget_fields():
    cfg = RaiseRunConfig(llm_max_tokens=2048, llm_reasoning_effort="medium")
    assert cfg.llm_max_tokens == 2048
    assert cfg.llm_reasoning_effort == "medium"

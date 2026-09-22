"""Mocked tests for the local Ollama OpenAI-compatible provider."""

from __future__ import annotations

import sys
import types

import pytest

from crowd_nav.reward_search.llm import (
    CompletionTruncatedError,
    OllamaLLMClient,
    extract_python_code,
    make_llm_client,
)


def _mock_openai(monkeypatch, completion: str) -> None:
    class FakeCompletions:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=completion))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))


def test_ollama_defaults_and_factory(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example/v1")
    client = OllamaLLMClient()
    assert client.model == "qwen3.5:4b"
    assert client.base_url == "http://localhost:11434/v1"
    assert client.max_tokens == 4096
    assert isinstance(make_llm_client("ollama"), OllamaLLMClient)


def test_ollama_base_url_isolated_from_vllm(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.example/v1/")
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.example/v1")
    assert OllamaLLMClient().base_url == "http://ollama.example/v1"


def test_truncated_ollama_completion_has_specific_error(monkeypatch):
    raw = "<think>reasoning</think>\n```python\n" + ("x" * 3900)
    _mock_openai(monkeypatch, raw)
    completion = OllamaLLMClient(max_attempts=1).complete("prompt")
    with pytest.raises(CompletionTruncatedError, match="no closing code fence found"):
        extract_python_code(completion, max_tokens=4096)


def test_ollama_complete_extracts_code_after_thinking_preamble(monkeypatch):
    raw = "<think>reasoning</think>\n```python\ndef compute_reward(state):\n    return 1.0\n```"
    _mock_openai(monkeypatch, raw)
    completion = OllamaLLMClient(max_attempts=1).complete("prompt")
    assert extract_python_code(completion) == "def compute_reward(state):\n    return 1.0"


def test_ollama_disables_reasoning_and_accepts_reasoning_field(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="",
                            reasoning="```python\ndef compute_reward(state):\n    return 2.0\n```",
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    completion = OllamaLLMClient(max_attempts=1).complete("prompt")

    assert captured["reasoning_effort"] == "none"
    assert "def compute_reward" in completion
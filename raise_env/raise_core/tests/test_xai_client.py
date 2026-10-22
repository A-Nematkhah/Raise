"""xAI key resolution and client factory (no network)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from raise_core.llm import XaiLLMClient, make_llm_client, resolve_xai_api_key


def test_resolve_xai_key_from_file(tmp_path: Path, monkeypatch):
    key_file = tmp_path / "xai_key"
    key_file.write_text("# comment\nxai-testkey1234567890abcdef\n", encoding="utf-8")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_KEY", raising=False)
    monkeypatch.setenv("XAI_KEY_PATH", str(key_file))
    assert resolve_xai_api_key() == "xai-testkey1234567890abcdef"


def test_resolve_xai_key_from_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XAI_API_KEY", "xai-from-env-abcdefghijklmnopqrst")
    monkeypatch.setenv("XAI_KEY_PATH", str(tmp_path / "missing"))
    assert resolve_xai_api_key().startswith("xai-from-env")


def test_make_llm_client_xai(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "xai_key"
    key_file.write_text("xai-factorykey1234567890abcdef\n", encoding="utf-8")
    monkeypatch.setenv("XAI_KEY_PATH", str(key_file))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    client = make_llm_client("xai", model="grok-4.20-0309-reasoning", max_tokens=512)
    assert isinstance(client, XaiLLMClient)
    assert client.provider == "xai"
    assert client.model == "grok-4.20-0309-reasoning"
    assert client.api_key.startswith("xai-factory")
    assert "api.x.ai" in client.base_url


def test_resolve_xai_key_missing_raises(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("XAI_KEY", raising=False)
    monkeypatch.setenv("XAI_KEY_PATH", str(tmp_path / "nope"))
    monkeypatch.setenv("XAI_KEYS_PATH", str(tmp_path / "nope.json"))
    with pytest.raises(RuntimeError, match="xAI API key not found"):
        resolve_xai_api_key()

"""Tests for LLM code extraction and Gen0 batch splitting."""

from __future__ import annotations

from crowd_nav.reward_search.llm import (
    extract_all_python_blocks,
    extract_python_code,
    split_reward_function_sources,
)
from crowd_nav.reward_search.prompts import format_d1_initial_batch


def test_extract_python_code_strips_fences_and_prose():
    raw = (
        "Sure! Here is the function:\n"
        "```python\n"
        "def compute_reward(state):\n"
        "    return float(1.0)\n"
        "```\n"
        "Hope this helps."
    )
    code = extract_python_code(raw)
    assert "def compute_reward" in code
    assert "Hope" not in code


def test_split_reward_function_sources_multiple_defs():
    raw = (
        "```python\n"
        "def compute_reward_v1(state):\n"
        "    return float(1.0)\n\n"
        "def compute_reward_v2(state):\n"
        "    return float(2.0)\n"
        "```"
    )
    parts = split_reward_function_sources(raw)
    assert len(parts) == 2
    assert all("def compute_reward(state, memory)" in p for p in parts)


def test_format_d1_initial_batch_asks_for_n_functions():
    prompt = format_d1_initial_batch(8)
    assert "8 diverse" in prompt
    assert "compute_reward_v8" in prompt


def test_extract_all_python_blocks_multiple_fences():
    raw = "```python\nx=1\n```\n\n```python\ndef f():\n pass\n```"
    blocks = extract_all_python_blocks(raw)
    assert len(blocks) == 2

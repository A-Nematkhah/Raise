"""Guards so Groq on_demand TPM (prompt + max_tokens) cannot exceed the org cap."""

from __future__ import annotations

import math

import pytest

from raise_core.llm import (
    _GROQ_ORG_TOKEN_LIMIT,
    _GROQ_PROMPT_TOKEN_MARGIN,
    _GROQ_REQUEST_TOKEN_CAP,
    _clamp_groq_request_budget,
    estimate_groq_prompt_tokens,
    groq_margined_input_budget,
    groq_raw_input_budget,
)
from raise_core.token_budget import PromptBudgetError


def test_soft_cap_never_exceeds_org_limit():
    assert int(_GROQ_REQUEST_TOKEN_CAP) <= int(_GROQ_ORG_TOKEN_LIMIT)


def test_clamp_keeps_prompt_plus_completion_under_cap():
    system = "You are a reward-function designer."
    # Large but under the margined cap so clamp can shrink max_tokens.
    user = ("metric line SR CR TR NT PL ITR SD " * 20 + "\n") * 8
    out = _clamp_groq_request_budget(
        system_prompt=system, user_prompt=user, max_tokens=2000
    )
    prompt = estimate_groq_prompt_tokens(system, user)
    assert prompt < int(_GROQ_REQUEST_TOKEN_CAP)
    assert prompt + out <= int(_GROQ_REQUEST_TOKEN_CAP)
    assert prompt + out <= int(_GROQ_ORG_TOKEN_LIMIT)
    assert out <= 2000


def test_clamp_rejects_prompt_alone_over_cap():
    system = "sys"
    user = ("abcd " * 8000)
    with pytest.raises(PromptBudgetError):
        _clamp_groq_request_budget(
            system_prompt=system, user_prompt=user, max_tokens=2000
        )


def test_raw_input_budget_leaves_room_after_margin():
    """Stage I raw fit limit × margin + completion ≤ soft cap."""
    for completion in (512, 1800, 2000):
        raw = groq_raw_input_budget(completion)
        margined_room = groq_margined_input_budget(completion)
        # Worst case: a prompt that uses the full raw budget.
        worst_margined = int(math.ceil(raw * float(_GROQ_PROMPT_TOKEN_MARGIN)))
        assert worst_margined <= margined_room
        assert worst_margined + completion <= int(_GROQ_REQUEST_TOKEN_CAP)
        assert worst_margined + completion <= int(_GROQ_ORG_TOKEN_LIMIT)


def test_batch_max_tokens_never_reserves_7000():
    class _E:
        llm = type("L", (), {"max_tokens": 2000})()

        def _batch_max_tokens(self, population_size: int) -> int:
            from raise_core.explore import StageIEvolver

            return StageIEvolver._batch_max_tokens(self, population_size)

    assert _E()._batch_max_tokens(8) <= 2000
    assert _E()._batch_max_tokens(8) < 7000


def test_sizing_completion_prefers_crossover_input_room():
    """llm.max_tokens=3500 must not shrink raw input room to 3200 for cards."""
    from raise_core.llm import groq_sizing_completion_tokens

    # crowdnav_run failure mode: desired input 3800, requested completion 3500.
    sizing = groq_sizing_completion_tokens(3800, 3500)
    raw = groq_raw_input_budget(sizing)
    assert raw >= 3800
    assert sizing < 3500
    # Send-time clamp still keeps margined prompt + completion under soft cap.
    assert int(math.ceil(3800 * float(_GROQ_PROMPT_TOKEN_MARGIN))) + sizing <= int(
        _GROQ_REQUEST_TOKEN_CAP
    )

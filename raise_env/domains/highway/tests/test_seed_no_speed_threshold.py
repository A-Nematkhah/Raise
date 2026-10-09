"""Guard: highway seed must not pin state.speed to a fixed numeric threshold."""

from __future__ import annotations

import re

from domains.highway.prompts import D5_SEED_FUNCTION, _SELECTION_OBJECTIVE


def test_d5_seed_has_no_traffic_speed_anchor():
    assert "traffic_speed" not in D5_SEED_FUNCTION
    assert "V_TARGET" not in D5_SEED_FUNCTION
    assert "lag_penalty" not in D5_SEED_FUNCTION


def test_d5_seed_no_numeric_speed_comparison():
    pat = re.compile(
        r"state\.speed\s*(?:<|>|<=|>=|==)\s*\d+(?:\.\d+)?|"
        r"\d+(?:\.\d+)?\s*(?:<|>|<=|>=|==)\s*state\.speed"
    )
    assert pat.search(D5_SEED_FUNCTION) is None, D5_SEED_FUNCTION


def test_selection_objective_has_no_soft_success_or_speed_target():
    assert "soft_success" not in _SELECTION_OBJECTIVE
    assert "traffic-matching" not in _SELECTION_OBJECTIVE

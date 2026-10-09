"""Rejection buckets are computed from real validator messages."""

from __future__ import annotations

import pytest

from domains.highway.reward_checks import make_highway_validator
from raise_core.rejection_log import categorize_validation_error

_FN = "def compute_reward(state, memory):\n"


@pytest.mark.parametrize(
    "code, expected",
    [
        ("def compute_reward(state, memory)\n    return 0.0\n", "syntax"),
        ("import os\n" + _FN + "    return 0.0\n", "forbidden_import"),
        (_FN + "    return float(state.lane_position)\n", "not_on_allowlist"),
        (_FN + "    f = math.factorial\n    return 0.0\n", "not_on_allowlist"),
        (_FN + "    xs = [v.x for v in state.others]\n    return 0.0\n", "forbidden_construct"),
        (_FN + "    while True:\n        pass\n    return 0.0\n", "forbidden_construct"),
        ("HIST = []\n" + _FN + "    return 0.0\n", "mutable_global"),
        (_FN + "    return float(eval('1'))\n", "forbidden_name"),
        ("X = max(1, 2)\n" + _FN + "    return 0.0\n", "top_level_code"),
        (_FN + "    return 1e15\n", "reward_too_large"),
        (_FN + "    return float('nan')\n", "non_finite_output"),
        ("def compute_reward(state, memory, extra):\n    return 0.0\n", "wrong_signature"),
        ("def compute_reward(state, memory=None):\n    return 0.0\n", "wrong_signature"),
        (_FN + "    return float(state.others[99].x)\n", "runtime_error"),
    ],
)
def test_validator_messages_map_to_buckets(code: str, expected: str):
    reward_fn, err = make_highway_validator().try_validate(code)
    assert reward_fn is None
    assert categorize_validation_error(err) == expected, err


def test_runtime_error_quoting_timeout_field_is_not_a_timeout():
    msg = "runtime error during smoke test: TypeError: bad operand for 'timeout'"
    assert categorize_validation_error(msg) == "runtime_error"
    assert (
        categorize_validation_error("execution exceeded timeout of 5.0s") == "timeout"
    )

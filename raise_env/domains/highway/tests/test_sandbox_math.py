"""The pre-injected ``math`` module: safe functions pass, everything else is rejected."""

from __future__ import annotations

import math

import pytest

from domains.highway.prompts import format_d1_initial
from domains.highway.reward_checks import make_highway_validator
from domains.highway.state import default_smoke_states


@pytest.mark.parametrize(
    "expr",
    [
        "math.sqrt(1.0 + s)",
        "math.exp(-s / 10.0)",
        "math.tanh(s / 30.0)",
        "math.log1p(s)",
        "math.hypot(state.ego.vx, state.ego.vy)",
        "math.pi * 0.0 + s",
    ],
)
def test_safe_math_functions_are_accepted(expr: str):
    code = (
        "def compute_reward(state, memory):\n"
        "    s = float(state.speed)\n"
        f"    return float({expr})\n"
    )
    reward_fn, err = make_highway_validator().try_validate(code)
    assert reward_fn is not None, err
    assert math.isfinite(reward_fn.compute(default_smoke_states()[0]))


@pytest.mark.parametrize("attr", ["factorial", "gamma", "__dict__", "pow"])
def test_other_math_attributes_are_rejected(attr: str):
    code = (
        "def compute_reward(state, memory):\n"
        f"    f = math.{attr}\n"
        "    return 0.0\n"
    )
    reward_fn, err = make_highway_validator().try_validate(code)
    assert reward_fn is None
    assert "math" in (err or "") or "forbidden" in (err or "")


def test_math_constant_allowed_at_module_level():
    code = (
        "HALF_PI = math.pi / 2.0\n"
        "def compute_reward(state, memory):\n"
        "    return float(math.cos(state.ego.heading) * HALF_PI)\n"
    )
    reward_fn, err = make_highway_validator().try_validate(code)
    assert reward_fn is not None, err


def test_import_error_message_points_to_injected_math():
    code = "import math\ndef compute_reward(state, memory):\n    return 0.0\n"
    _, err = make_highway_validator().try_validate(code)
    assert err is not None and "math.sqrt" in err and "** 0.5" not in err


def test_prompt_lists_allowed_math_functions():
    prompt = format_d1_initial()
    assert "math.tanh" in prompt and "never write `import math`" in prompt

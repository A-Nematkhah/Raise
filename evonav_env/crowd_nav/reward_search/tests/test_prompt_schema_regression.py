"""Regression: rendered Stage I/II/III prompts must keep schema + sandbox facts."""

from __future__ import annotations

import pytest

from crowd_nav.reward_search.prompts import (
    D3_SYSTEM_PROMPT,
    format_d1_initial,
    format_d1_initial_batch,
    format_d2_crossover,
    format_d2_mutation,
)
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.sandbox.config import FORBIDDEN_NAME_IDS
from crowd_nav.reward_search.sandbox.errors import RewardSandboxError
from crowd_nav.reward_search.sandbox.runtime import default_smoke_states


_SCHEMA_SNIPPETS = (
    "state.robot.px",
    "state.robot.py",
    "state.robot.vx",
    "state.robot.vy",
    "state.robot.radius",
    "state.robot.gx",
    "state.robot.gy",
    "state.robot.v_pref",
    "state.humans",
    "state.dmin",
    "state.discomfort_dist",
    "state.collision",
    "state.reaching_goal",
    "state.timeout",
)

_MATH_SNIPPETS = (
    "math module is not importable",
    "** 0.5",
    "getattr, hasattr, or __import__",
    "state.robot.px[0]",
)


def _assert_contains(text: str, needles: tuple[str, ...], *, label: str) -> None:
    missing = [n for n in needles if n not in text]
    assert not missing, f"{label} missing substrings: {missing}"


def test_d1_prompt_contains_schema_and_math_rules():
    text = format_d1_initial(include_seed=False, include_external_knowledge=False)
    _assert_contains(text, _SCHEMA_SNIPPETS, label="D1")
    _assert_contains(text, _MATH_SNIPPETS, label="D1")
    assert "NO state.history" in text or "There is NO state.history" in text


def test_d1_batch_prompt_contains_schema_and_math_rules():
    text = format_d1_initial_batch(4, include_seed=False, include_external_knowledge=False)
    _assert_contains(text, _SCHEMA_SNIPPETS, label="D1 batch")
    _assert_contains(text, _MATH_SNIPPETS, label="D1 batch")


def test_d2_prompts_contain_schema_and_math_rules():
    cross = format_d2_crossover("codeA", "codeB", "note")
    mut = format_d2_mutation("elitist", "weakness")
    for label, text in (("D2 crossover", cross), ("D2 mutation", mut)):
        _assert_contains(text, _SCHEMA_SNIPPETS, label=label)
        assert "math module is not importable" in text
        assert "** 0.5" in text
        assert "__import__" in text
        assert "getattr" in text


def test_d3_system_prompt_forbids_math_import_and_reflection():
    assert "math module is not importable" in D3_SYSTEM_PROMPT
    assert "** 0.5" in D3_SYSTEM_PROMPT
    assert "__import__" in D3_SYSTEM_PROMPT
    assert "getattr" in D3_SYSTEM_PROMPT
    assert "state.robot" in D3_SYSTEM_PROMPT and "state.dmin" in D3_SYSTEM_PROMPT


def test_d3_repair_prompt_forbids_math_import():
    from crowd_nav.reward_search.prompts import format_d3_repair

    text = format_d3_repair(
        bad_code="def compute_reward(state, memory):\n    return None\n",
        validation_error="got NoneType",
    )
    assert "math module is not importable" in text
    assert "import math" in text  # mentioned as forbidden
    assert "Optional `import math`" not in text
    assert "getattr" in text and "__import__" in text
    assert "** 0.5" in text


def test_dunder_import_is_explicitly_forbidden_at_ast():
    assert "__import__" in FORBIDDEN_NAME_IDS
    code = (
        "def compute_reward(state, memory):\n"
        "    m = __import__('math')\n"
        "    return float(m.sqrt(1.0))\n"
    )
    with pytest.raises(RewardSandboxError) as exc:
        RewardValidator().validate_code(code)
    assert "forbidden" in str(exc.value).lower()
    assert "__import__" in str(exc.value)


def test_nested_import_math_rejected_at_ast_not_runtime():
    """Regression for run_5h_easy: nested import math → clear AST reject."""
    code = (
        "def compute_reward(state, memory):\n"
        "    import math\n"
        "    return float(math.sqrt(1.0))\n"
    )
    fn, err = RewardValidator().try_validate(code)
    assert fn is None
    assert err is not None
    assert "import is forbidden" in err.lower()
    assert "__import__ not found" not in err.lower()


def test_math_usable_without_import_via_injection():
    code = (
        "def compute_reward(state, memory):\n"
        "    return float(math.sqrt(4.0))\n"
    )
    reward = RewardValidator().validate_code(code)
    assert reward.compute(default_smoke_states()[0]) == 2.0

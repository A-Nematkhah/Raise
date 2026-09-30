"""Sandbox AST must reject list/set/dict comps and generator expressions."""

from __future__ import annotations

import pytest

from domains.crowdnav.prompts import D5_SEED_FUNCTION as CN_SEED
from domains.highway.prompts import D5_SEED_FUNCTION as HW_SEED
from raise_core.domains import load_domain, make_validator_for_domain
from raise_core.sandbox.errors import RewardSandboxError
from raise_core.sandbox.validator import RewardValidator


@pytest.mark.parametrize(
    "kind,body",
    [
        (
            "list comprehension",
            "return float(sum([float(v.x) for v in state.others]))",
        ),
        (
            "set comprehension",
            "return float(len({float(v.x) for v in state.others}))",
        ),
        (
            "dict comprehension",
            "d = {str(i): float(i) for i in range(3)}\n    return float(d.get('0', 0.0))",
        ),
        (
            "generator expression",
            "return float(sum((float(v.x) for v in state.others)))",
        ),
    ],
)
def test_highway_rejects_comprehensions(kind: str, body: str):
    pack = load_domain("highway")
    v = make_validator_for_domain(pack)
    code = f"def compute_reward(state, memory):\n    {body}\n"
    with pytest.raises(RewardSandboxError) as ei:
        v.validate_code(code)
    msg = str(ei.value).lower()
    assert "comprehension" in msg or "generator expression" in msg, msg
    assert kind.split()[0] in msg or "generator" in msg


@pytest.mark.parametrize(
    "kind,body",
    [
        (
            "list comprehension",
            "return float(sum([float(h.px) for h in state.humans]))",
        ),
        (
            "set comprehension",
            "return float(len({float(h.px) for h in state.humans}))",
        ),
        (
            "dict comprehension",
            "d = {str(i): float(i) for i in range(3)}\n    return float(d.get('0', 0.0))",
        ),
        (
            "generator expression",
            "return float(sum((float(h.px) for h in state.humans)))",
        ),
    ],
)
def test_crowdnav_rejects_comprehensions(kind: str, body: str):
    pack = load_domain("crowdnav")
    v = make_validator_for_domain(pack)
    code = f"def compute_reward(state, memory):\n    {body}\n"
    with pytest.raises(RewardSandboxError) as ei:
        v.validate_code(code)
    msg = str(ei.value).lower()
    assert "comprehension" in msg or "generator expression" in msg, msg


def test_explicit_for_loop_still_ok_both_domains():
    hw = make_validator_for_domain(load_domain("highway"))
    hw_ok = """
def compute_reward(state, memory):
    nearest = 50.0
    for v in state.others:
        d = (float(v.x) ** 2 + float(v.y) ** 2) ** 0.5
        if d < nearest:
            nearest = d
    return float(state.progress + 0.01 * nearest)
"""
    assert hw.validate_code(hw_ok) is not None

    cn = make_validator_for_domain(load_domain("crowdnav"))
    cn_ok = """
def compute_reward(state, memory):
    nearest = 50.0
    for h in state.humans:
        d = ((float(h.px) - float(state.robot.px)) ** 2
             + (float(h.py) - float(state.robot.py)) ** 2) ** 0.5
        if d < nearest:
            nearest = d
    return float(state.dmin + 0.01 * nearest)
"""
    assert cn.validate_code(cn_ok) is not None


def test_domain_seeds_still_validate():
    assert make_validator_for_domain(load_domain("highway")).validate_code(HW_SEED)
    assert make_validator_for_domain(load_domain("crowdnav")).validate_code(CN_SEED)
    # Default CrowdNav-style validator (no domain pack) also rejects comps.
    with pytest.raises(RewardSandboxError):
        RewardValidator().validate_code(
            "def compute_reward(state, memory):\n"
            "    return float(sum([1.0 for _ in range(3)]))\n"
        )

"""Highway AST field allowlist + seed diversity."""

from __future__ import annotations

import pytest

from domains.highway.prompts import D5_SEED_FUNCTION
from domains.highway.sandbox_fields import (
    HIGHWAY_FORBIDDEN_HALLUCINATIONS,
    highway_sandbox_config,
)
from raise_core.domains import load_domain, make_validator_for_domain
from raise_core.llm import SeedVariantLLMClient, extract_python_code
from raise_core.sandbox.errors import RewardSandboxError


def test_highway_validator_accepts_seed():
    pack = load_domain("highway")
    v = make_validator_for_domain(pack)
    assert pack.sandbox_config is not None
    assert pack.sandbox_config.allowed_attributes is not None
    reward = v.validate_code(D5_SEED_FUNCTION)
    assert reward is not None


def test_highway_ast_rejects_lane_position():
    pack = load_domain("highway")
    v = make_validator_for_domain(pack)
    bad = """
def compute_reward(state, memory):
    if state.collision:
        return -20.0
    return float(state.progress + state.lane_position)
"""
    with pytest.raises(RewardSandboxError) as ei:
        v.validate_code(bad)
    assert "lane_position" in str(ei.value)


def test_highway_ast_rejects_distance_and_robot():
    pack = load_domain("highway")
    v = make_validator_for_domain(pack)
    for bad in (
        """
def compute_reward(state, memory):
    return float(state.distance)
""",
        """
def compute_reward(state, memory):
    return float(state.robot.x)
""",
    ):
        with pytest.raises(RewardSandboxError):
            v.validate_code(bad)


def test_highway_allows_clearance_via_others():
    pack = load_domain("highway")
    v = make_validator_for_domain(pack)
    ok = """
def compute_reward(state, memory):
    if state.collision:
        return -20.0
    nearest = 50.0
    for v in state.others:
        d = (float(v.x) ** 2 + float(v.y) ** 2) ** 0.5
        if d < nearest:
            nearest = d
    return float(state.progress + 0.1 * nearest)
"""
    assert v.validate_code(ok) is not None


def test_seed_variant_structural_diversity():
    client = SeedVariantLLMClient(base_code=D5_SEED_FUNCTION)
    pack = load_domain("highway")
    v = make_validator_for_domain(pack)
    codes = []
    for _ in range(10):
        raw = client.complete("gen")
        code = extract_python_code(raw)
        codes.append(code)
        v.validate_code(code)
    # At least two distinct structural families among first 5 kinds.
    assert len(set(codes)) >= 4
    joined = "\n".join(codes)
    assert "state.others" in joined or "lane_index" in joined or "crawl_pen" in joined


def test_forbidden_list_covers_common_hallucinations():
    for name in ("lane_position", "distance", "robot", "humans"):
        assert name in HIGHWAY_FORBIDDEN_HALLUCINATIONS


def test_highway_sandbox_config_factory():
    cfg = highway_sandbox_config()
    assert "ego" in cfg.allowed_attributes
    assert "lane_index" in cfg.allowed_attributes
    assert "lane_position" not in cfg.allowed_attributes

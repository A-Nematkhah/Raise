"""Phase 3: highway AST field allowlist + repair / seed diversity."""

from __future__ import annotations

import pytest

from domains.highway.prompts import D5_SEED_FUNCTION, format_d3_repair
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


def test_crowdnav_validator_unchanged_no_allowlist():
    pack = load_domain("crowdnav")
    v = make_validator_for_domain(pack)
    assert getattr(pack, "sandbox_config", None) is None
    # CrowdNav still allows state.robot.* (no highway allowlist).
    from domains.crowdnav.prompts import D5_SEED_FUNCTION as CN_SEED

    assert v.validate_code(CN_SEED) is not None


def test_repair_prompt_mentions_hallucinated_field():
    text = format_d3_repair(
        bad_code="def compute_reward(state, memory):\n    return state.lane_position\n",
        validation_error="attribute 'lane_position' is not on the domain allowlist",
    )
    assert "lane_position" in text
    assert "lane_index" in text
    assert "MANDATORY FIELD FIXES" in text


def test_seed_variant_structural_diversity():
    from domains.highway.prompts import D5_SEED_FUNCTION as HW_SEED

    client = SeedVariantLLMClient(base_code=HW_SEED)
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

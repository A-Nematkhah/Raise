"""Prompt / fixture wording matches what the wrapper and fitness code actually do."""

from __future__ import annotations

from domains.highway import prompts
from domains.highway.action_config import ActionModeSettings, action_space_prompt_block
from domains.highway.stage1 import smoke_trajectories


def test_timeout_documented_as_survival():
    assert "survived" in prompts.D1_USER_PROMPT


def test_selection_objective_describes_holdout_fitness():
    text = prompts._SELECTION_OBJECTIVE
    assert "holdout fitness" in text and "higher is better" in text
    assert "survival rate minus collision rate" in text
    assert "Pareto" not in text


def test_action_block_documents_action_type_per_mode():
    meta = action_space_prompt_block(ActionModeSettings(mode="meta_default"))
    assert "0=LANE_LEFT" in meta and "4=SLOWER" in meta
    cont = action_space_prompt_block(
        ActionModeSettings(mode="continuous", continuous_lateral=False)
    )
    assert "state.action[0]" in cont and "state.action[1]" not in cont


def test_smoke_fixture_never_marks_failures_as_timeout():
    for traj in smoke_trajectories():
        for state in traj.states:
            assert not (state.timeout and (state.off_road or state.collision))
        if traj.label == "timeout":
            assert traj.states[-1].off_road

"""min_separation_post timing and potential_factor_for / with_kinematics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from domains.crowdnav.settings import CrowdNavSettings
from domains.crowdnav.state import (
    CrowdNavRewardState,
    HumanState,
    RobotState,
    boundary_min_separation,
    potential_factor_for,
    with_kinematics,
)
from domains.crowdnav.tests.test_env_wrapper import _env


def test_potential_factor_for_kinematics():
    assert potential_factor_for("unicycle") == 3.0
    assert potential_factor_for("holonomic") == 2.0
    assert potential_factor_for("HOLONOMIC") == 2.0


def test_min_separation_post_matches_state_geometry():
    env = _env(human_num=2)
    try:
        env.reset(options={"case": 0})
        env.step(np.array([0.2, 0.1], dtype=np.float32))
        state = env.last_state
        assert state is not None
        assert state.min_separation_post is not None
        recomputed = boundary_min_separation(state.robot, state.humans)
        assert state.min_separation_post == pytest.approx(recomputed)
        # Pre-step dmin may differ from post-step gap after motion.
        assert isinstance(state.min_separation, float)
    finally:
        env.close()


def test_state_from_dict_defaults_min_separation_post():
    from domains.crowdnav.stage1 import _state_from_dict, _state_to_dict

    env = _env(human_num=1)
    try:
        env.reset(options={"case": 1})
        env.step(np.array([0.0, 0.3], dtype=np.float32))
        state = env.last_state
        assert state is not None
        payload = _state_to_dict(state)
        assert "min_separation_post" in payload
        legacy = dict(payload)
        legacy.pop("min_separation_post")
        loaded = _state_from_dict(legacy)
        # Legacy rows recompute post-step gap from stored robot/humans geometry.
        assert loaded.min_separation_post == pytest.approx(
            boundary_min_separation(loaded.robot, loaded.humans)
        )
    finally:
        env.close()


def test_with_kinematics_uses_goal_bearing_not_always_pi_over_2():
    settings = CrowdNavSettings(kinematics="unicycle")
    robot = RobotState(
        px=0.0,
        py=0.0,
        vx=0.0,
        vy=0.0,
        gx=4.0,
        gy=0.0,
        theta=float("nan"),
        radius=0.3,
        v_pref=1.0,
        kinematics="holonomic",
    )
    st = CrowdNavRewardState(
        robot=robot,
        humans=(),
        action=(1.0, 0.0),
        collision=False,
        reached_goal=False,
        timeout=False,
        min_separation=1.0,
        discomfort_dist=0.25,
        goal_distance=4.0,
        progress=0.0,
        time_step=0.25,
        global_time=0.25,
        time_limit=25.0,
        potential_factor=2.0,
        min_separation_post=1.0,
    )
    out = with_kinematics((st,), settings)
    assert len(out) == 1
    assert out[0].potential_factor == 3.0
    # First heading comes from atan2(gy-py, gx-px) when theta is not finite.
    assert out[0].robot.theta == pytest.approx(0.0, abs=1e-6)


def test_with_kinematics_prefers_recorded_theta():
    settings = CrowdNavSettings(kinematics="unicycle")
    theta0 = math.pi / 4.0
    robot = RobotState(
        px=0.0,
        py=0.0,
        vx=0.0,
        vy=0.0,
        gx=0.0,
        gy=4.0,
        theta=theta0,
        radius=0.3,
        v_pref=1.0,
        kinematics="holonomic",
    )
    st = CrowdNavRewardState(
        robot=robot,
        humans=(
            HumanState(px=1.0, py=1.0, vx=0.0, vy=0.0, radius=0.3, visible=True),
        ),
        action=(0.0, 0.0),
        collision=False,
        reached_goal=False,
        timeout=False,
        min_separation=1.0,
        discomfort_dist=0.25,
        goal_distance=4.0,
        progress=0.0,
        time_step=0.25,
        global_time=0.25,
        time_limit=25.0,
        min_separation_post=1.0,
    )
    out = with_kinematics((st,), settings)
    # Zero motion keeps prior heading (initialized from recorded theta).
    assert out[0].robot.theta == pytest.approx(theta0)

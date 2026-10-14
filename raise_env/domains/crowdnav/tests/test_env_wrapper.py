"""RewardInjectedCrowdNavEnv against the unmodified upstream CrowdSim."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import raise_paths  # noqa: F401
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import JointState

from domains.crowdnav.env_wrapper import (
    HUMAN_FEATURES,
    SELF_FEATURES,
    RewardInjectedCrowdNavEnv,
    eval_cases,
    scenario_case,
)
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.settings import (
    DEFAULT_ENV_CONFIG_PATH,
    CrowdNavSettings,
    continuous_to_velocity,
    discrete_action_table,
)

HOLONOMIC = CrowdNavSettings()
INFO_KEYS = {
    "raise_outcome", "raise_collision", "raise_reached_goal", "raise_timeout", "raise_danger",
    "raise_danger_min_dist", "raise_min_separation", "raise_native_reward", "raise_progress",
    "raise_speed", "raise_goal_distance", "raise_global_time", "raise_phase", "raise_case",
    "raise_reward_components",
}


def _env(reward_fn=None, *, settings=HOLONOMIC, **kw) -> RewardInjectedCrowdNavEnv:
    return RewardInjectedCrowdNavEnv(reward_fn, settings=settings, **kw)


def _toward_goal(env, frac: float = 1.0) -> ActionXY:
    r = env.sim.robot
    dx, dy = r.gx - r.px, r.gy - r.py
    d = math.hypot(dx, dy) or 1.0
    return ActionXY(frac * env.v_pref * dx / d, frac * env.v_pref * dy / d)


def _orca(env):
    policy = ORCA()
    policy.time_step = env.time_step
    return lambda: policy.predict(
        JointState(env.sim.robot.get_full_state(), [h.get_observable_state() for h in env.sim.humans])
    )


def _episode(env, case, controller, phase="val"):
    env.reset(options={"case": case, "phase": phase})
    steps = []
    while True:
        _obs, reward, terminated, truncated, info = env.step_upstream(controller())
        steps.append((reward, terminated, truncated, info, env.last_state))
        if terminated or truncated:
            return steps


def _first_episode_with(env, outcome, controller_fn, cases=range(40)):
    for case in cases:
        steps = _episode(env, case, controller_fn(env))
        if steps[-1][3]["raise_outcome"] == outcome:
            return case, steps
    pytest.fail(f"no {outcome} episode in val cases {cases}")


# ------------------------------------------------------------------ init / reset


def test_reset_returns_valid_observation_and_info():
    env = _env(phase="val")
    obs, info = env.reset(seed=3)
    assert env.human_num == 5 and env.time_step == 0.25 and env.time_limit == 25.0
    assert obs.shape == (SELF_FEATURES + HUMAN_FEATURES * 5,) == env.observation_space.shape
    assert obs.dtype == np.float32 and np.all(np.isfinite(obs))
    assert env.observation_space.contains(obs)
    assert info == {"raise_phase": "val", "raise_case": 3}
    # Upstream circle crossing: robot at (0, -r) heading for (0, r).
    r = env.sim.robot
    assert (r.px, r.py, r.gx, r.gy) == (0, -4, 0, 4)
    assert obs[0] == pytest.approx(8.0) and obs[1] == pytest.approx(1.0)


def test_reset_is_reproducible_and_leaves_global_rng_alone():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a, b = _env(phase="val"), _env(phase="val")
    obs_a, _ = a.reset(seed=11)
    obs_b, _ = b.reset(seed=11)
    assert np.array_equal(np.random.get_state()[1], before)
    np.testing.assert_array_equal(obs_a, obs_b)
    for _ in range(10):
        oa = a.step(5)[0]
        ob = b.step(5)[0]
        np.testing.assert_array_equal(oa, ob)
    # Different cases give different human layouts.
    assert not np.array_equal(a.reset(seed=11)[0], a.reset(seed=12)[0])


def test_phase_split_and_case_mapping():
    env = _env(phase="train")
    _, info = env.reset(seed=5)
    assert info["raise_phase"] == "train"
    assert info["raise_case"] == scenario_case(5, "train", env.sim.case_size["train"])
    _, info = env.reset(options={"phase": "test", "case": 7})
    assert info == {"raise_phase": "test", "raise_case": 7}
    cases = eval_cases(1234, 30, 100)
    assert len(set(cases)) == 30 and cases == eval_cases(1234, 30, 100)
    assert all(0 <= c < 100 for c in cases)
    with pytest.raises(ValueError):
        _env(phase="holdout")


# ------------------------------------------------------------------ spaces


def test_discrete_action_space_is_the_upstream_cadrl_set():
    env = _env()
    assert env.action_space.n == 81
    table = env.action_table
    assert table[0] == (0.0, 0.0)
    speeds = [(math.exp((i + 1) / 5) - 1) / (math.e - 1) for i in range(5)]
    # build_action_space: product(rotations, speeds), rotations linspace(0, 2pi, 16, endpoint=False).
    vx, vy = table[1 + 3 * 5 + 4]
    rot = 3 * 2 * math.pi / 16
    assert (vx, vy) == pytest.approx((speeds[4] * math.cos(rot), speeds[4] * math.sin(rot)))
    assert max(math.hypot(*a) for a in table) == pytest.approx(env.v_pref)
    uni = discrete_action_table(CrowdNavSettings(kinematics="unicycle"), 1.0)
    assert uni[1] == pytest.approx((speeds[0], -math.pi / 4)) and uni[-1] == pytest.approx((1.0, math.pi / 4))


def test_continuous_action_mapping():
    s = CrowdNavSettings(action_mode="continuous")
    env = _env(settings=s)
    assert env.action_space.shape == (2,)
    assert continuous_to_velocity([1.0, 1.0], s, 1.0) == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert continuous_to_velocity([5.0, 0.0], s, 1.0) == pytest.approx((1.0, 0.0))
    uni = CrowdNavSettings(kinematics="unicycle", action_mode="continuous")
    assert continuous_to_velocity([-1.0, 1.0], uni, 1.0) == pytest.approx((0.0, math.pi / 4))
    up = _env(settings=uni).to_upstream_action(np.array([1.0, 0.0]))
    assert isinstance(up, ActionRot) and (up.v, up.r) == pytest.approx((1.0, 0.0))


@pytest.mark.parametrize("settings", [
    HOLONOMIC,
    CrowdNavSettings(action_mode="continuous"),
    CrowdNavSettings(kinematics="unicycle"),
])
def test_gymnasium_env_checker(settings):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unbounded Box observation warnings only
        check_env(_env(settings=settings), skip_render_check=True)


# ------------------------------------------------------------------ stepping / outcomes


def test_step_returns_state_and_info():
    seed_reward = make_crowdnav_validator().validate_code(D5_SEED_FUNCTION)
    env = _env(seed_reward, phase="val")
    env.reset(seed=0)
    d0 = env.sim.robot.get_position()
    obs, reward, terminated, truncated, info = env.step(env.action_space.n - 1)
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and not terminated and not truncated
    assert set(info) == INFO_KEYS and info["raise_outcome"] is None
    st = env.last_state
    assert len(st.humans) == 5 and st.time_step == 0.25 and st.global_time == pytest.approx(0.25)
    moved = math.hypot(st.robot.px - d0[0], st.robot.py - d0[1])
    assert moved == pytest.approx(0.25 * env.v_pref)
    assert st.progress == pytest.approx(8.0 - st.goal_distance)
    assert info["raise_speed"] == pytest.approx(env.v_pref)


def test_collision_terminates_episode():
    env = _env(phase="val")
    case, steps = _first_episode_with(env, "collision", lambda e: (lambda: _toward_goal(e)))
    reward, terminated, truncated, info, state = steps[-1]
    assert terminated and not truncated
    assert info["raise_collision"] and state.collision and not state.reached_goal
    assert reward == info["raise_native_reward"] == pytest.approx(-0.25)
    assert state.min_separation < 0.0
    assert all(not s[1] and not s[2] for s in steps[:-1])


def test_goal_terminates_episode():
    env = _env(phase="val")
    case, steps = _first_episode_with(env, "success", _orca)
    reward, terminated, truncated, info, state = steps[-1]
    assert terminated and not truncated
    assert state.reached_goal and not state.collision and info["raise_reached_goal"]
    assert state.goal_distance < env.sim.robot.radius
    assert reward == pytest.approx(1.0)


def test_time_limit_truncates_episode():
    env = _env(phase="val")
    case, steps = _first_episode_with(env, "timeout", lambda e: (lambda: ActionXY(0.0, 0.0)))
    reward, terminated, truncated, info, state = steps[-1]
    assert truncated and not terminated and state.timeout
    # Upstream: Timeout once global_time >= time_limit - 1 (checked before the clock advances).
    assert len(steps) == 97 and state.global_time == pytest.approx(24.25)
    assert reward == pytest.approx(0.0)


def test_seed_reward_reproduces_upstream_reward_and_danger_distance():
    seed_reward = make_crowdnav_validator().validate_code(D5_SEED_FUNCTION)
    env = _env(seed_reward, phase="val")
    rng = np.random.default_rng(0)
    n_steps = n_danger = 0
    outcomes = set()
    for case in range(12):
        env.reset(options={"case": case})
        done = False
        while not done:
            action = int(rng.integers(env.action_space.n)) if case % 2 else 80
            _obs, reward, terminated, truncated, info = env.step(action)
            assert reward == pytest.approx(info["raise_native_reward"], abs=1e-12)
            if info["raise_danger"]:
                n_danger += 1
                assert info["raise_min_separation"] == pytest.approx(info["raise_danger_min_dist"], abs=1e-12)
            n_steps += 1
            done = terminated or truncated
        outcomes.add(info["raise_outcome"])
    assert n_steps > 200 and n_danger > 0 and len(outcomes) >= 2


def test_native_reward_used_without_reward_fn():
    env = _env(None, phase="val")
    env.reset(seed=1)
    _, reward, *_rest, info = env.step(0)
    assert reward == info["raise_native_reward"]


def test_render_is_headless_and_off_by_default():
    assert _env().render() is None
    env = _env(render_mode="rgb_array", phase="val")
    env.reset(seed=0)
    env.step(80)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8


def test_unicycle_robot_turns_and_moves():
    env = _env(settings=CrowdNavSettings(kinematics="unicycle"), phase="val")
    obs, _ = env.reset(seed=0)
    theta0 = env.sim.robot.theta
    env.step(len(env.action_table) - 1)  # full speed, +pi/4 rotation
    assert env.sim.robot.theta == pytest.approx(theta0 + math.pi / 4)
    assert env.last_state.action == pytest.approx((1.0, math.pi / 4))


def test_mixed_scenarios_are_rejected(tmp_path):
    with open(DEFAULT_ENV_CONFIG_PATH, encoding="utf-8") as fh:
        text = fh.read().replace("test_sim = circle_crossing", "test_sim = mixed")
    path = os.path.join(str(tmp_path), "env.config")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    with pytest.raises(ValueError, match="mixed"):
        _env(settings=CrowdNavSettings(env_config_path=path))

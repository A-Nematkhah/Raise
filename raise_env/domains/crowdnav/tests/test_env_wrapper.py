"""Contract tests for the CrowdNav++ variable-human Gymnasium wrapper."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import raise_paths  # noqa: F401
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout
from domains.crowdnav.env_wrapper import (
    HUMAN_OBSERVATION_KEYS,
    RewardInjectedCrowdNavEnv,
    eval_cases,
    load_env_config,
    make_crowd_sim,
    scenario_case,
    step_min_separation,
)
from domains.crowdnav.settings import (
    DEFAULT_ENV_CONFIG_PATH,
    CrowdNavSettings,
    continuous_to_velocity,
)

INFO_KEYS = {
    "raise_outcome",
    "raise_collision",
    "raise_reached_goal",
    "raise_timeout",
    "raise_danger",
    "raise_danger_min_dist",
    "raise_min_separation",
    "raise_native_reward",
    "raise_progress",
    "raise_speed",
    "raise_goal_distance",
    "raise_global_time",
    "raise_phase",
    "raise_case",
    "raise_reward_components",
}


def _config(*, human_num: int = 2, val_size: int = 17, test_size: int = 19):
    config = load_env_config(DEFAULT_ENV_CONFIG_PATH)
    config.env.val_size = val_size
    config.env.test_size = test_size
    config.env.randomize_attributes = False
    config.env.use_wrapper = False
    config.sim.human_num = human_num
    config.sim.human_num_range = 0
    config.sim.predict_method = "none"
    config.humans.random_goal_changing = False
    config.humans.end_goal_changing = False
    return config


def _settings(*, human_num: int = 2, val_size: int = 17, test_size: int = 19):
    return CrowdNavSettings(
        kinematics="holonomic",
        action_mode="continuous",
        env_config_obj=_config(
            human_num=human_num, val_size=val_size, test_size=test_size
        ),
        human_num_override=human_num,
    )


def _env(
    *,
    phase: str = "train",
    human_num: int = 2,
    settings: CrowdNavSettings | None = None,
):
    return RewardInjectedCrowdNavEnv(
        phase=phase,
        settings=settings or _settings(human_num=human_num),
    )


def _rng_snapshot():
    state = np.random.get_state()
    return state[0], state[1].copy(), state[2], state[3], state[4]


def _assert_observations_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def _force_scene(env, outcome: str | None = None):
    env.reset(options={"case": 3})
    sim = env.sim
    robot = sim.robot
    for human in sim.humans:
        human.set(100.0, 100.0, 100.0, 100.0, 0.0, 0.0, 0.0)
    if outcome == "collision":
        sim.humans[0].set(
            robot.px + 0.2, robot.py, robot.px + 0.2, robot.py, 0.0, 0.0, 0.0
        )
    elif outcome == "success":
        robot.px, robot.py = robot.gx, robot.gy
    elif outcome == "timeout":
        sim.global_time = sim.time_limit - 1.0
        sim.humans[0].set(
            robot.px + 0.2, robot.py, robot.px + 0.2, robot.py, 0.0, 0.0, 0.0
        )


def _danger_scene(env):
    env.reset(options={"case": 2})
    sim = env.sim
    robot = sim.robot
    robot.set(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0)
    human = sim.humans[0]
    human.set(0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    human.gx, human.gy = 0.0, 0.0
    sim.potential = -10.0
    env._goal_distance = 10.0
    sim.generate_ob(reset=True, sort=False)


def test_upstream_python_config_and_human_override():
    upstream = load_env_config(DEFAULT_ENV_CONFIG_PATH)
    assert upstream.env.time_limit == 50
    assert upstream.env.time_step == 0.25
    assert upstream.sim.human_num == 20
    assert upstream.sim.human_num_range == 0
    assert upstream.sim.arena_size == 6
    assert upstream.robot.sensor_range == 5
    assert upstream.action_space.kinematics == "holonomic"
    assert CrowdNavSettings().kinematics == "unicycle"

    default_env = RewardInjectedCrowdNavEnv(settings=CrowdNavSettings())
    assert default_env.max_human_num == 20
    assert default_env.observation_space["spatial_edges"].shape == (20, 2)

    env = _env(human_num=2)
    obs, _ = env.reset(options={"case": 0})
    expected_human_num = env.settings.human_num_override
    assert env.human_num == expected_human_num
    assert len(env.sim.humans) == expected_human_num
    assert env.sim.max_human_num == expected_human_num
    assert env.observation_space["spatial_edges"].shape == (expected_human_num, 2)
    assert env.observation_space.contains(obs)


def test_hard_config_is_loadable_python_config():
    config_path = Path(__file__).resolve().parents[3] / "configs" / "crowdnav_hard.py"
    config = load_env_config(str(config_path))
    assert config.env.time_limit == 30
    assert config.reward.success_reward == 1.0
    assert config.reward.collision_penalty == -0.25
    assert config.sim.circle_radius == 6
    assert config.sim.human_num == 10
    assert config.sim.human_num_range == 0
    assert config.sim.predict_method == "const_vel"
    assert config.sim.predict_steps == 5
    assert config.data.pred_timestep == 0.25
    assert config.action_space.kinematics == "holonomic"
    assert config.robot.visible is True


def test_hard_config_keeps_exactly_ten_holonomic_humans():
    config_path = Path(__file__).resolve().parents[3] / "configs" / "crowdnav_hard.py"
    settings = CrowdNavSettings(
        kinematics="holonomic",
        action_mode="continuous",
        env_config_path=str(config_path),
    )
    env = RewardInjectedCrowdNavEnv(settings=settings)
    try:
        assert env.sim.robot.kinematics == "holonomic"
        assert env.min_human_num == env.max_human_num == 10
        assert env.observation_space["spatial_edges"].shape == (10, 12)
        for case in range(12):
            observation, _ = env.reset(options={"case": case})
            assert len(env.sim.humans) == 10
            assert env.sim.human_num_range == 0
            assert env.observation_space.contains(observation)
            assert observation["spatial_edges"].shape == (10, 12)
    finally:
        env.close()


def test_hard_config_observation_contains_constant_velocity_human_paths():
    config_path = Path(__file__).resolve().parents[3] / "configs" / "crowdnav_hard.py"
    env = RewardInjectedCrowdNavEnv(
        settings=CrowdNavSettings(
            kinematics="holonomic",
            action_mode="continuous",
            env_config_path=str(config_path),
        )
    )
    try:
        env.reset(options={"case": 0})
        robot = env.sim.robot
        human = env.sim.humans[0]
        human.px = robot.px + 1.0
        human.py = robot.py
        human.vx = 0.4
        human.vy = -0.2
        env.sim.human_visibility[0] = True
        env.sim.ob["visible_masks"][0] = True
        observation = env._format_observation(env.sim.ob)
        np.testing.assert_allclose(
            observation["spatial_edges"][0],
            [1.0, 0.0, 1.1, -0.05, 1.2, -0.1, 1.3, -0.15, 1.4, -0.2, 1.5, -0.25],
            atol=1e-6,
        )
    finally:
        env.close()


def test_generalization_config_uses_fixed_17_humans():
    config_path = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "crowdnav_generalization_17.py"
    )
    settings = CrowdNavSettings(
        kinematics="unicycle",
        action_mode="continuous",
        env_config_path=str(config_path),
    )
    env = RewardInjectedCrowdNavEnv(settings=settings)
    try:
        obs, _ = env.reset(options={"case": 12})
        assert env.time_limit == 30.0
        assert env.human_num == 17
        assert env.min_human_num == env.max_human_num == 17
        assert len(env.sim.humans) == 17
        assert env.sim.robot.visible
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_unicycle_respects_configured_variable_population_range_and_visibility():
    config = _config(human_num=10)
    config.sim.human_num_range = 3
    config.robot.visible = True
    settings = CrowdNavSettings(
        kinematics="unicycle",
        action_mode="continuous",
        env_config_obj=config,
    )
    env = RewardInjectedCrowdNavEnv(settings=settings)
    try:
        assert env.sim.robot.visible
        assert env.min_human_num == 7
        assert env.max_human_num == 13
        counts = set()
        for case in range(20):
            obs, _ = env.reset(options={"case": case})
            assert 7 <= len(env.sim.humans) <= 13
            assert env.sim.human_num_range == 3
            assert env.observation_space.contains(obs)
            counts.add(len(env.sim.humans))

        env.reset(options={"case": 31})
        first_count = len(env.sim.humans)
        env.reset(options={"case": 31})
        assert len(env.sim.humans) == first_count

        env.reset(options={"case": 9})
        env.sim.global_time = 4.75
        for _ in range(24):
            _obs, _reward, terminated, truncated, _info = env.step(
                np.zeros(2, dtype=np.float32)
            )
            assert 7 <= len(env.sim.humans) <= 13
            if terminated or truncated:
                break
        assert len(counts) > 1
    finally:
        env.close()


def test_dict_observation_space_matches_upstream_observations():
    env = _env()
    obs, info = env.reset(options={"case": 3})
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert set(obs) == set(HUMAN_OBSERVATION_KEYS)
    assert info == {"raise_phase": "train", "raise_case": 3}
    assert env.observation_space.contains(obs)
    assert obs["robot_node"].shape == (7,)
    assert obs["temporal_edges"].shape == (2,)
    assert obs["visible_masks"].shape == (env.max_human_num,)
    assert obs["spatial_edges"].shape == (env.max_human_num, 2)
    assert np.isscalar(obs["detected_human_num"])


def test_continuous_action_space_has_finite_kinematic_bounds():
    holonomic = _env()
    assert np.all(np.isfinite(holonomic.action_space.low))
    assert np.all(np.isfinite(holonomic.action_space.high))
    np.testing.assert_array_equal(holonomic.action_space.low, [-1.0, -1.0])
    np.testing.assert_array_equal(holonomic.action_space.high, [1.0, 1.0])

    unicycle_settings = CrowdNavSettings(
        kinematics="unicycle",
        action_mode="continuous",
        env_config_obj=_config(human_num=1),
        human_num_override=1,
    )
    unicycle = _env(settings=unicycle_settings)
    assert np.all(np.isfinite(unicycle.action_space.low))
    assert np.all(np.isfinite(unicycle.action_space.high))
    np.testing.assert_allclose(
        unicycle.action_space.low, [-0.1, -0.06], rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        unicycle.action_space.high, [0.087, 0.06], rtol=0.0, atol=1e-7
    )


def test_variable_human_range_is_read_from_python_config():
    config = _config(human_num=10)
    config.sim.human_num_range = 2
    settings = CrowdNavSettings(kinematics="holonomic", env_config_obj=config)
    env = _env(settings=settings)
    counts = set()
    for case in range(20):
        obs, _ = env.reset(options={"case": case})
        counts.add(len(env.sim.humans))
        assert env.observation_space.contains(obs)
    assert env.sim.min_human_num == 8
    assert env.sim.max_human_num == 12
    assert len(counts) > 1
    assert all(8 <= count <= 12 for count in counts)
    assert obs["spatial_edges"].shape == (12, 2)


def test_reset_cases_are_reproducible_and_restore_global_numpy_rng():
    a, b = _env(phase="val"), _env(phase="val")
    np.random.seed(123)
    before = _rng_snapshot()
    obs_a, info_a = a.reset(options={"case": 11})
    obs_b, info_b = b.reset(options={"case": 11})
    assert _rng_snapshot()[0] == before[0]
    np.testing.assert_array_equal(_rng_snapshot()[1], before[1])
    _assert_observations_equal(obs_a, obs_b)
    assert info_a == info_b == {"raise_phase": "val", "raise_case": 11}
    assert a.sim.rand_seed == b.sim.rand_seed == 11

    for _ in range(8):
        before = _rng_snapshot()
        next_a = a.step(np.array([0.2, -0.1], dtype=np.float32))
        next_b = b.step(np.array([0.2, -0.1], dtype=np.float32))
        _assert_observations_equal(next_a[0], next_b[0])
        assert _rng_snapshot()[0] == before[0]
        np.testing.assert_array_equal(_rng_snapshot()[1], before[1])

    obs_other, _ = a.reset(options={"case": 12})
    assert any(not np.array_equal(obs_a[k], obs_other[k]) for k in obs_a)


def test_phase_seed_offsets_case_sizes_and_eval_cases():
    settings = _settings(val_size=7, test_size=11)
    env = _env(phase="val", settings=settings)
    _, info = env.reset(options={"case": 6})
    assert info == {"raise_phase": "val", "raise_case": 6}
    assert env.sim.rand_seed == 6
    assert env.sim.case_size["val"] == 7
    assert env.sim.case_counter["val"] == 0
    assert env.sim.phase == "test"

    _, info = env.reset(options={"phase": "test", "case": 10})
    assert info == {"raise_phase": "test", "raise_case": 10}
    assert env.sim.rand_seed == 1010
    assert env.sim.case_size["test"] == 11
    assert env.sim.case_counter["test"] == 0

    train = _env(phase="train")
    _, train_info = train.reset(seed=5)
    assert train_info["raise_case"] == scenario_case(
        5, "train", train.sim.case_size["train"]
    )
    assert train.sim.rand_seed == 2000 + train_info["raise_case"]

    cases = eval_cases(1234, 7, 7)
    assert len(set(cases)) == 7
    assert cases == eval_cases(1234, 7, 7)
    assert all(0 <= case < 7 for case in cases)


def test_continuous_action_space_and_upstream_holonomic_clipping():
    env = _env()
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (2,)
    np.testing.assert_array_equal(env.action_space.low, [-env.v_pref, -env.v_pref])
    np.testing.assert_array_equal(env.action_space.high, [env.v_pref, env.v_pref])
    assert continuous_to_velocity([2.0, 0.0], env.settings, env.v_pref) == pytest.approx(
        (1.0, 0.0)
    )
    action = env.to_upstream_action(np.array([0.3, 0.4], dtype=np.float32))
    assert isinstance(action, ActionXY)
    assert math.hypot(action.vx, action.vy) == pytest.approx(0.5)


def test_visible_and_invisible_humans_follow_upstream_sensor_range():
    settings = _settings(human_num=2)
    config = settings.env_config_obj
    config.robot.sensor_range = 5.0
    env = _env(settings=settings)
    env.reset(options={"case": 0})
    sim = env.sim
    robot = sim.robot
    robot.px, robot.py = 0.0, 0.0
    robot.vx, robot.vy = 1.0, 0.0
    sim.humans[0].set(2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0)
    sim.humans[1].set(8.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0)
    obs = env._format_observation(sim.generate_ob(reset=True, sort=False))
    assert obs["visible_masks"].tolist() == [True, False]
    assert obs["detected_human_num"] == 1
    assert env.observation_space.contains(obs)
    _, _, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert tuple(h.visible for h in env.last_state.humans) == (True, False)


@pytest.mark.parametrize(
    ("forced", "outcome", "terminated", "truncated", "reward"),
    [
        ("collision", "collision", True, False, -20.0),
        ("success", "success", True, False, 10.0),
        ("timeout", "timeout", False, True, 0.0),
    ],
)
def test_outcome_termination_and_native_rewards(
    forced, outcome, terminated, truncated, reward
):
    env = _env(phase="test")
    _force_scene(env, forced)
    obs, actual_reward, actual_terminated, actual_truncated, info = env.step(
        np.zeros(2, dtype=np.float32)
    )
    assert env.observation_space.contains(obs)
    assert (actual_terminated, actual_truncated) == (terminated, truncated)
    assert info["raise_outcome"] == outcome
    assert info["raise_native_reward"] == actual_reward == pytest.approx(reward)
    assert info["raise_collision"] == (outcome == "collision")
    assert info["raise_reached_goal"] == (outcome == "success")
    assert info["raise_timeout"] == (outcome == "timeout")
    assert set(info) == INFO_KEYS


def test_visible_unicycle_robot_works_in_evaluation_phase():
    config = _config(human_num=2)
    config.robot.visible = True
    settings = CrowdNavSettings(
        kinematics="unicycle",
        action_mode="continuous",
        env_config_obj=config,
        human_num_override=2,
    )
    env = RewardInjectedCrowdNavEnv(phase="val", settings=settings)
    try:
        env.reset(options={"case": 0})
        assert env.sim.phase == "test"
        assert env.sim.robot.visible is True

        observation, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )

        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["raise_native_reward"] == pytest.approx(reward)
    finally:
        env.close()


def test_timeout_has_priority_over_collision():
    env = _env(phase="test")
    _force_scene(env, "timeout")
    _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated and truncated
    assert reward == 0.0
    assert info["raise_outcome"] == "timeout"


def test_step_min_separation_matches_upstream_current_state_collision_check():
    env = _env()
    env.reset(options={"case": 0})
    robot = env.sim.robot
    robot.px, robot.py = 0.0, 0.0
    env.sim.humans[0].px, env.sim.humans[0].py = 0.8, 0.0
    env.sim.humans[1].px, env.sim.humans[1].py = 2.0, 0.0
    action = ActionXY(1.0, 0.0)
    assert step_min_separation(env.sim, action) == pytest.approx(0.2)
    assert step_min_separation(env.sim, ActionXY(-1.0, 0.0)) == pytest.approx(0.2)


def test_native_reward_matches_direct_upstream_step():
    settings = _settings(human_num=2)
    wrapped = _env(settings=settings)
    wrapped.reset(options={"case": 9})
    raw = make_crowd_sim(settings.env_config_obj, settings)
    raw.reset(phase="train", test_case=9)
    raw.phase = "train"
    action = np.array([0.0, 0.0], dtype=np.float32)
    expected_ob, expected_reward, expected_done, expected_info = raw.step(action.copy())
    obs, reward, terminated, truncated, info = wrapped.step(action.copy())
    assert wrapped.observation_space.contains(obs)
    assert reward == info["raise_native_reward"] == pytest.approx(expected_reward)
    assert (terminated or truncated) == expected_done
    assert type(expected_info["info"]) in (Collision, Danger, Nothing, ReachGoal, Timeout)


def test_seed_reward_reproduces_upstream_reward_over_random_episodes():
    from domains.crowdnav.prompts import D5_SEED_FUNCTION
    from domains.crowdnav.reward_checks import make_crowdnav_validator

    config = _config(human_num=4)
    config.sim.human_num_range = 1
    config.env.time_limit = 8.0
    config.reward.success_reward = 3.5
    config.reward.collision_penalty = -7.0
    config.reward.discomfort_dist = 0.18
    config.reward.discomfort_penalty_factor = 2.3
    settings = CrowdNavSettings(
        kinematics="holonomic",
        action_mode="continuous",
        env_config_obj=config,
    )
    env = RewardInjectedCrowdNavEnv(phase="test", settings=settings)
    seed_reward = make_crowdnav_validator().validate_code(D5_SEED_FUNCTION)
    rng = np.random.default_rng(2305)
    steps_checked = 0
    try:
        for case in range(12):
            env.reset(options={"case": case})
            for _ in range(math.ceil(config.env.time_limit / config.env.time_step) + 2):
                _, _, terminated, truncated, info = env.step(
                    rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                )
                assert env.last_state is not None
                assert seed_reward.compute(env.last_state) == pytest.approx(
                    info["raise_native_reward"], abs=1e-6
                )
                steps_checked += 1
                if terminated or truncated:
                    break
        assert steps_checked >= 100
    finally:
        env.close()


def test_reward_state_serialization_keeps_new_fields_and_loads_legacy_states():
    from domains.crowdnav.stage1 import _state_from_dict, _state_to_dict

    env = _env(human_num=2)
    try:
        env.reset(options={"case": 4})
        env.step(np.array([0.1, -0.2], dtype=np.float32))
        state = env.last_state
        assert state is not None
        payload = _state_to_dict(state)
        assert _state_from_dict(payload) == state

        payload["robot"].pop("kinematics")
        for human in payload["humans"]:
            human.pop("visible")
        for key in (
            "danger",
            "success_reward",
            "collision_penalty",
            "discomfort_penalty_factor",
            "potential_delta",
            "potential_factor",
        ):
            payload.pop(key)
        legacy = _state_from_dict(payload)
        assert legacy.robot.kinematics == "holonomic"
        assert all(not human.visible for human in legacy.humans)
        assert legacy.success_reward == 1.0
        assert legacy.collision_penalty == -0.25
    finally:
        env.close()


def test_val_uses_test_future_danger_definition_and_val_scenarios():
    settings = _settings(human_num=1, val_size=7, test_size=11)
    val_env, test_env = _env(phase="val", settings=settings), _env(
        phase="test", settings=settings
    )
    _danger_scene(val_env)
    _danger_scene(test_env)
    assert val_env.sim.rand_seed == 2
    assert test_env.sim.rand_seed == 1002
    assert val_env.sim.phase == test_env.sim.phase == "test"

    _, val_reward, _, _, val_info = val_env.step(np.zeros(2, dtype=np.float32))
    _, test_reward, _, _, test_info = test_env.step(np.zeros(2, dtype=np.float32))
    assert val_info["raise_danger"] and test_info["raise_danger"]
    assert np.isfinite(val_info["raise_danger_min_dist"])
    assert np.isfinite(test_info["raise_danger_min_dist"])
    assert val_info["raise_danger"] == test_info["raise_danger"]
    assert val_info["raise_danger_min_dist"] == pytest.approx(
        test_info["raise_danger_min_dist"]
    )
    assert val_reward == pytest.approx(test_reward)
    assert val_env.current_phase == "val"
    assert val_env.sim.case_size["val"] == 7
    assert val_env.sim.case_counter["val"] == 3


@pytest.mark.parametrize("human_num", [1, 4])
def test_gymnasium_env_checker(human_num):
    env = _env(human_num=human_num)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)

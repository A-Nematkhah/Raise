"""
Tests for pluggable CrowdNav reward adapter + sandbox.

Does not require Python-RVO2 / full simulator: rvo2 is stubbed for calc_reward
integration checks, and LegacyReward / sandbox are tested on synthetic states.
"""

from __future__ import annotations

import math
import sys
import types
from typing import Any, List, Tuple
from unittest.mock import MagicMock

import pytest

# Stub rvo2 before any crowd_sim import (ORCA humans need the module present).
if "rvo2" not in sys.modules:
    sys.modules["rvo2"] = MagicMock()

from crowd_nav.reward_search.sandbox import (
    RewardSandboxError,
    RewardValidator,
    SandboxConfig,
)
from crowd_nav.reward_search.state import (
    HumanObservable,
    LegacyReward,
    RewardFunction,
    RewardState,
    RobotRewardState,
)
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.info import Collision, Danger, Nothing, ReachGoal, Timeout


def _reference_crowd_sim_reward(
    *,
    success_reward: float,
    collision_penalty: float,
    discomfort_dist: float,
    discomfort_penalty_factor: float,
    time_step: float,
    kinematics: str,
    potential: float,
    dmin: float,
    collision: bool,
    reaching_goal: bool,
    timeout: bool,
    action: Any,
    robot_px: float,
    robot_py: float,
    robot_gx: float,
    robot_gy: float,
) -> Tuple[float, float]:
    """Byte-for-byte copy of the pre-refactor CrowdSim.calc_reward scalar path."""
    if timeout:
        reward = 0
    elif collision:
        reward = collision_penalty
    elif reaching_goal:
        reward = success_reward
    elif dmin < discomfort_dist:
        reward = (dmin - discomfort_dist) * discomfort_penalty_factor * time_step
    else:
        potential_cur = math.sqrt((robot_px - robot_gx) ** 2 + (robot_py - robot_gy) ** 2)
        reward = 2 * (-abs(potential_cur) - potential)
        potential = -abs(potential_cur)

    if kinematics == "unicycle":
        r_spin = -5 * action.r ** 2
        if action.v < 0:
            r_back = -2 * abs(action.v)
        else:
            r_back = 0.0
        reward = reward + r_spin + r_back

    return float(reward), float(potential)


def _make_state(**kwargs) -> RewardState:
    defaults = dict(
        robot=RobotRewardState(
            px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=4.0, gy=0.0, v_pref=1.0
        ),
        humans=(HumanObservable(2.0, 0.0, 0.0, 0.0, 0.3),),
        dmin=1.5,
        discomfort_dist=0.25,
        collision=False,
        reaching_goal=False,
        timeout=False,
        action=ActionXY(0.5, 0.0),
        time_step=0.25,
        global_time=1.0,
        time_limit=50.0,
    )
    defaults.update(kwargs)
    return RewardState(**defaults)


RECORDED_CASES: List[dict] = [
    # progress / potential shaping
    dict(
        dmin=1.5,
        collision=False,
        reaching_goal=False,
        timeout=False,
        robot_px=0.0,
        robot_py=0.0,
        robot_gx=4.0,
        robot_gy=0.0,
        potential=-4.0,
        action=ActionXY(0.5, 0.0),
        kinematics="holonomic",
    ),
    # closer to goal
    dict(
        dmin=1.2,
        collision=False,
        reaching_goal=False,
        timeout=False,
        robot_px=1.0,
        robot_py=0.0,
        robot_gx=4.0,
        robot_gy=0.0,
        potential=-4.0,
        action=ActionXY(0.5, 0.0),
        kinematics="holonomic",
    ),
    # discomfort
    dict(
        dmin=0.1,
        collision=False,
        reaching_goal=False,
        timeout=False,
        robot_px=0.0,
        robot_py=0.0,
        robot_gx=4.0,
        robot_gy=0.0,
        potential=-4.0,
        action=ActionXY(0.3, 0.1),
        kinematics="holonomic",
    ),
    # collision
    dict(
        dmin=-0.05,
        collision=True,
        reaching_goal=False,
        timeout=False,
        robot_px=0.0,
        robot_py=0.0,
        robot_gx=4.0,
        robot_gy=0.0,
        potential=-4.0,
        action=ActionXY(0.5, 0.0),
        kinematics="holonomic",
    ),
    # success
    dict(
        dmin=2.0,
        collision=False,
        reaching_goal=True,
        timeout=False,
        robot_px=3.85,
        robot_py=0.0,
        robot_gx=4.0,
        robot_gy=0.0,
        potential=-0.2,
        action=ActionXY(0.1, 0.0),
        kinematics="holonomic",
    ),
    # timeout
    dict(
        dmin=1.0,
        collision=False,
        reaching_goal=False,
        timeout=True,
        robot_px=1.0,
        robot_py=1.0,
        robot_gx=4.0,
        robot_gy=0.0,
        potential=-3.0,
        action=ActionXY(0.0, 0.0),
        kinematics="holonomic",
    ),
    # unicycle spin + reverse
    dict(
        dmin=1.0,
        collision=False,
        reaching_goal=False,
        timeout=False,
        robot_px=0.0,
        robot_py=0.0,
        robot_gx=3.0,
        robot_gy=0.0,
        potential=-3.0,
        action=ActionRot(-0.2, 0.15),
        kinematics="unicycle",
    ),
]


@pytest.mark.parametrize("case", RECORDED_CASES)
def test_legacy_reward_bit_identical_to_original_formula(case):
    coeffs = dict(
        success_reward=10.0,
        collision_penalty=-20.0,
        discomfort_dist=0.25,
        discomfort_penalty_factor=10.0,
        time_step=0.25,
        kinematics=case["kinematics"],
        pot_factor=2.0,
        unicycle_spin_coef=-5.0,
    )
    legacy = LegacyReward(**coeffs)
    legacy.sync_potential(case["potential"])

    state = _make_state(
        robot=RobotRewardState(
            px=case["robot_px"],
            py=case["robot_py"],
            vx=0.5,
            vy=0.0,
            radius=0.3,
            gx=case["robot_gx"],
            gy=case["robot_gy"],
            v_pref=1.0,
        ),
        dmin=case["dmin"],
        collision=case["collision"],
        reaching_goal=case["reaching_goal"],
        timeout=case["timeout"],
        action=case["action"],
        time_step=0.25,
        discomfort_dist=0.25,
    )
    got = legacy.compute(state)
    expected, _ = _reference_crowd_sim_reward(
        success_reward=10.0,
        collision_penalty=-20.0,
        discomfort_dist=0.25,
        discomfort_penalty_factor=10.0,
        time_step=0.25,
        kinematics=case["kinematics"],
        potential=case["potential"],
        dmin=case["dmin"],
        collision=case["collision"],
        reaching_goal=case["reaching_goal"],
        timeout=case["timeout"],
        action=case["action"],
        robot_px=case["robot_px"],
        robot_py=case["robot_py"],
        robot_gx=case["robot_gx"],
        robot_gy=case["robot_gy"],
    )
    assert got == expected


def test_legacy_reward_potential_sequence_matches_original():
    """Multi-step potential shaping must stay bit-identical across an episode."""
    legacy = LegacyReward(
        success_reward=10.0,
        collision_penalty=-20.0,
        discomfort_dist=0.25,
        discomfort_penalty_factor=10.0,
        time_step=0.25,
        kinematics="holonomic",
    )
    potential = -5.0
    legacy.sync_potential(potential)
    positions = [0.0, 0.5, 1.0, 1.5, 2.0]
    for px in positions:
        state = _make_state(
            robot=RobotRewardState(
                px=px, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=5.0, gy=0.0, v_pref=1.0
            ),
            dmin=2.0,
        )
        got = legacy.compute(state)
        expected, potential = _reference_crowd_sim_reward(
            success_reward=10.0,
            collision_penalty=-20.0,
            discomfort_dist=0.25,
            discomfort_penalty_factor=10.0,
            time_step=0.25,
            kinematics="holonomic",
            potential=potential,
            dmin=2.0,
            collision=False,
            reaching_goal=False,
            timeout=False,
            action=ActionXY(0.5, 0.0),
            robot_px=px,
            robot_py=0.0,
            robot_gx=5.0,
            robot_gy=0.0,
        )
        assert got == expected


class _FixedReward(RewardFunction):
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def reset(self) -> None:
        return None

    def compute(self, state: RewardState) -> float:
        self.calls += 1
        return float(self.value)


class _Human:
    def __init__(self, px, py, radius=0.3):
        self.px, self.py = px, py
        self.vx = self.vy = 0.0
        self.radius = radius
        self.gx = px
        self.gy = py


class _Robot:
    def __init__(self):
        self.px = self.py = 0.0
        self.vx = 0.5
        self.vy = 0.0
        self.radius = 0.3
        self.gx = 4.0
        self.gy = 0.0
        self.v_pref = 1.0
        self.theta = 0.0
        self.kinematics = "holonomic"
        self.sensor_range = 5.0
        self.policy = types.SimpleNamespace(name="srnn")

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy


def _bind_calc_reward_env(reward_fn: RewardFunction, *, collision=False, reaching=False, timeout=False):
    """Minimal stand-in that reuses CrowdSim.calc_reward without full configure()."""
    from crowd_sim.envs.crowd_sim import CrowdSim

    env = CrowdSim(reward_fn=reward_fn)
    env.robot = _Robot()
    if collision:
        env.humans = [_Human(0.2, 0.0)]
    elif reaching:
        env.robot.px, env.robot.py = 3.9, 0.0
        env.humans = [_Human(10.0, 10.0)]
    else:
        env.humans = [_Human(2.0, 0.0)]
    env.discomfort_dist = 0.25
    env.success_reward = 10.0
    env.collision_penalty = -20.0
    env.discomfort_penalty_factor = 10.0
    env.time_step = 0.25
    env.time_limit = 50.0
    env.global_time = 49.0 if timeout else 1.0
    env.potential = -4.0
    env.reward_fn = reward_fn
    return env


def test_custom_reward_fn_invoked_without_changing_termination():
    fixed = _FixedReward(123.456)
    env = _bind_calc_reward_env(fixed, collision=True)
    reward, done, info = env.calc_reward(ActionXY(0.5, 0.0))
    assert fixed.calls == 1
    assert reward == pytest.approx(123.456)
    assert done is True
    assert isinstance(info, Collision)

    fixed2 = _FixedReward(-7.0)
    env2 = _bind_calc_reward_env(fixed2, reaching=True)
    reward2, done2, info2 = env2.calc_reward(ActionXY(0.1, 0.0))
    assert reward2 == pytest.approx(-7.0)
    assert done2 is True
    assert isinstance(info2, ReachGoal)

    fixed3 = _FixedReward(0.5)
    env3 = _bind_calc_reward_env(fixed3, timeout=True)
    reward3, done3, info3 = env3.calc_reward(ActionXY(0.0, 0.0))
    assert reward3 == pytest.approx(0.5)
    assert done3 is True
    assert isinstance(info3, Timeout)

    fixed4 = _FixedReward(1.25)
    env4 = _bind_calc_reward_env(fixed4)
    # Place human just inside discomfort band.
    env4.humans = [_Human(0.5, 0.0)]  # dist ≈ 0.5 - 0.6 = -0.1 → collision
    env4.humans = [_Human(0.8, 0.0)]  # closest ≈ 0.8-0.6=0.2 < 0.25 discomfort
    reward4, done4, info4 = env4.calc_reward(ActionXY(0.2, 0.0))
    assert reward4 == pytest.approx(1.25)
    assert done4 is False
    assert isinstance(info4, Danger)


VALID_CODE = """
def compute_reward(state, memory):
    reward = 0.0
    if state.reaching_goal:
        reward = reward + 10.0
    if state.collision:
        reward = reward - 20.0
    if state.dmin < state.discomfort_dist:
        reward = reward + (state.dmin - state.discomfort_dist)
    return float(reward)
"""


def test_sandbox_accepts_valid_code():
    fn = RewardValidator().validate_code(VALID_CODE)
    state = _make_state(reaching_goal=True)
    assert fn.compute(state) == pytest.approx(10.0)


def test_sandbox_rejects_import():
    code = "import os\ndef compute_reward(state, memory):\n    return 0.0\n"
    with pytest.raises(RewardSandboxError) as exc:
        RewardValidator().validate_code(code)
    assert "import" in str(exc.value).lower()


def test_sandbox_rejects_eval():
    code = "def compute_reward(state, memory):\n    return eval('1+1')\n"
    with pytest.raises(RewardSandboxError) as exc:
        RewardValidator().validate_code(code)
    assert "eval" in str(exc.value)


def test_sandbox_rejects_while():
    code = "def compute_reward(state, memory):\n    while True:\n        pass\n    return 0.0\n"
    with pytest.raises(RewardSandboxError) as exc:
        RewardValidator().validate_code(code)
    assert "while" in str(exc.value).lower()


def test_sandbox_rejects_infinite_loop_timeout():
    code = """
def compute_reward(state, memory):
    x = 0.0
    for i in range(10 ** 12):
        x = x + 1.0
    return x
"""
    validator = RewardValidator(config=SandboxConfig(timeout_seconds=0.2))
    with pytest.raises(RewardSandboxError) as exc:
        validator.validate_code(code)
    assert "timeout" in str(exc.value).lower()


def test_sandbox_rejects_non_finite():
    code = "def compute_reward(state, memory):\n    return math.inf\n"
    with pytest.raises(RewardSandboxError) as exc:
        RewardValidator().validate_code(code)
    assert "non-finite" in str(exc.value).lower() or "finite" in str(exc.value).lower()


def test_sandbox_rejects_numeric_overflow_before_rollout():
    code = """
def compute_reward(state, memory):
    return math.exp(abs(state.robot.px - state.robot.gx))
"""
    with pytest.raises(RewardSandboxError) as exc:
        RewardValidator().validate_code(code)
    assert "overflow" in str(exc.value).lower() or "finite" in str(exc.value).lower()


def test_sandbox_isinstance_is_available():
    code = """
def compute_reward(state, memory):
    if isinstance(state.collision, bool):
        return float(1.0)
    return float(0.0)
"""
    fn = RewardValidator().validate_code(code)
    assert fn.compute(_make_state()) == pytest.approx(1.0)


def test_prompt_memory_example_validates_in_sandbox():
    """Regression: prompts.py examples must always pass the sandbox."""
    from crowd_nav.reward_search.prompts import D5_SEED_FUNCTION, PROMPT_MEMORY_EXAMPLE

    validator = RewardValidator()
    for label, code in (
        ("PROMPT_MEMORY_EXAMPLE", PROMPT_MEMORY_EXAMPLE),
        ("D5_SEED_FUNCTION", D5_SEED_FUNCTION),
    ):
        reward_fn, err = validator.try_validate(code)
        assert reward_fn is not None, f"{label} failed sandbox: {err}"
        assert "class MyReward" not in code
        reward_fn.reset()
        s0 = _make_state(
            robot=RobotRewardState(
                px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=4.0, gy=0.0, v_pref=1.0
            )
        )
        s1 = _make_state(
            robot=RobotRewardState(
                px=1.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=4.0, gy=0.0, v_pref=1.0
            )
        )
        r0 = reward_fn.compute(s0)
        r1 = reward_fn.compute(s1)
        assert r0 == r0 and r1 == r1  # finite


def test_memory_cleared_on_reset():
    from crowd_nav.reward_search.prompts import PROMPT_MEMORY_EXAMPLE

    fn = RewardValidator().validate_code(PROMPT_MEMORY_EXAMPLE)
    s0 = _make_state(
        robot=RobotRewardState(
            px=0.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=4.0, gy=0.0, v_pref=1.0
        )
    )
    s1 = _make_state(
        robot=RobotRewardState(
            px=1.0, py=0.0, vx=0.5, vy=0.0, radius=0.3, gx=4.0, gy=0.0, v_pref=1.0
        )
    )
    fn.reset()
    first = fn.compute(s0)
    second = fn.compute(s1)
    assert second == pytest.approx(1.0)  # moved 1m closer → progress 1.0
    fn.reset()
    again = fn.compute(s0)
    assert again == pytest.approx(first)


def test_reward_adapter_governs_crowdsimpredrealgst():
    """
    CrowdSimPredRealGST.calc_reward must route through injected RewardFunction
    (via CrowdSimVarNum), not hard-coded LegacyReward / CrowdSimPred social term.
    """
    import numpy as np
    from crowd_sim.envs.crowd_sim_pred_real_gst import CrowdSimPredRealGST

    sentinel_value = 12345.0
    fixed = _FixedReward(sentinel_value)
    env = CrowdSimPredRealGST()
    assert type(env).__name__ == "CrowdSimPredRealGST"

    env.robot = _Robot()
    env.robot.visible = False
    env.robot.policy = types.SimpleNamespace(
        name="srnn",
        clip_action=lambda a, _v: a,
    )
    env.robot.step = lambda _action: None
    human = _Human(2.0, 0.0)
    human.step = lambda _action: None
    env.humans = [human]

    env.discomfort_dist = 0.25
    env.success_reward = 10.0
    env.collision_penalty = -20.0
    env.discomfort_penalty_factor = 10.0
    env.time_step = 0.25
    env.time_limit = 50.0
    env.global_time = 1.0
    env.potential = -4.0
    env.phase = "train"
    env.predict_steps = 5
    env.human_num = 1
    env.human_num_range = 0
    env.step_counter = 0
    env.random_goal_changing = False
    env.end_goal_changing = False
    env.record = False
    env.cur_human_states = np.zeros((1, 3), dtype=np.float64)
    env.config = types.SimpleNamespace(
        sim=types.SimpleNamespace(human_num=1, human_num_range=0),
        humans=types.SimpleNamespace(radius=0.3),
        args=types.SimpleNamespace(sort_humans=True),
    )
    env.set_reward_fn(fixed)

    # Avoid ORCA / FOV / GST obs machinery; keep the real step→calc_reward path.
    env.get_human_actions = lambda: [ActionXY(0.0, 0.0)]
    env.generate_ob = lambda reset=False, sort=False: {
        "robot_node": np.zeros((1, 7), dtype=np.float32),
        "temporal_edges": np.zeros((1, 2), dtype=np.float32),
        "spatial_edges": np.zeros((1, 12), dtype=np.float32),
        "visible_masks": np.ones((1,), dtype=bool),
        "detected_human_num": np.array([1.0], dtype=np.float32),
    }

    _ob, reward, _done, _info = env.step(ActionXY(0.1, 0.0))
    assert fixed.calls == 1
    assert reward == pytest.approx(sentinel_value)
    # Sentinel is far outside any LegacyReward scalar used in this repo.
    assert abs(reward) > 100.0

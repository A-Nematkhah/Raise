"""Hard CrowdNav++ configuration translated from the former INI settings."""

from copy import deepcopy

from crowd_nav.configs.config import Config as _UpstreamConfig

Config = _UpstreamConfig()
for _section in (
    "training",
    "env",
    "reward",
    "sim",
    "humans",
    "robot",
    "action_space",
    "orca",
    "sf",
    "data",
):
    setattr(Config, _section, deepcopy(getattr(_UpstreamConfig, _section)))

Config.env.time_limit = 30
Config.env.time_step = 0.25
Config.env.val_size = 100
Config.env.test_size = 500
Config.reward.success_reward = 1.0
Config.reward.collision_penalty = -0.25
Config.reward.discomfort_dist = 0.2
Config.reward.discomfort_penalty_factor = 0.5
Config.sim.circle_radius = 6
Config.sim.human_num = 10
Config.sim.human_num_range = 0
Config.sim.predict_method = "const_vel"
Config.sim.predict_steps = 5
Config.data.pred_timestep = 0.25
Config.action_space.kinematics = "holonomic"
Config.robot.visible = True

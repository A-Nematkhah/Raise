"""Fixed 17-human CrowdNav++ config for out-of-distribution evaluation."""

from copy import deepcopy

from configs.crowdnav_hard import Config as _HardConfig

Config = deepcopy(_HardConfig)
Config.sim.human_num = 17
Config.sim.human_num_range = 0

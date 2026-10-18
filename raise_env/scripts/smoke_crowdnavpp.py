#!/usr/bin/env python
"""Smoke-test the vendored CrowdNav++ variable-human simulator."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import raise_paths
import crowd_sim
from crowd_nav.configs.config import Config
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum


def _observation_shapes(observation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        key: {
            "type": type(value).__name__,
            "shape": tuple(np.shape(value)),
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
        }
        for key, value in observation.items()
    }


def main() -> int:
    vendor = Path(raise_paths.CROWDNAVPP_VENDOR_DIR).resolve()
    legacy = Path(raise_paths.LEGACY_CROWDNAV_VENDOR_DIR).resolve()
    module_path = Path(crowd_sim.__file__).resolve()
    if vendor not in module_path.parents:
        raise RuntimeError(f"Imported crowd_sim outside CrowdNav++ vendor: {module_path}")
    if any(Path(entry).resolve() == legacy for entry in sys.path if entry):
        raise RuntimeError(f"Legacy CrowdNav path remains importable: {legacy}")

    config = Config()
    config.sim.predict_method = "none"
    config.env.use_wrapper = False
    config.sim.human_num_range = 2
    phase = "train"

    print(f"crowd_sim module: {module_path}")
    print(
        "config: "
        f"time_limit={config.env.time_limit}s, time_step={config.env.time_step}s, "
        f"human_num={config.sim.human_num}, human_num_range={config.sim.human_num_range}, "
        f"kinematics={config.action_space.kinematics}, phase={phase}"
    )

    max_steps = int(config.env.time_limit / config.env.time_step) + 8
    for seed in (1, 7, 42):
        env = CrowdSimVarNum()
        env.configure(config)
        env.phase = phase
        env.nenv = 1
        env.thisSeed = seed
        observation = env.reset(phase=phase, test_case=0)
        initial_humans = int(env.human_num)
        observed_human_range = [initial_humans, initial_humans]
        info_types: Counter[str] = Counter()
        observation_shapes = _observation_shapes(observation)

        for step_index in range(max_steps):
            observation, _reward, done, info = env.step(env.action_space.sample())
            if not isinstance(info, dict) or "info" not in info:
                raise TypeError(f"Unexpected upstream info shape: {type(info).__name__}")
            info_types[type(info["info"]).__name__] += 1
            observed_human_range[0] = min(observed_human_range[0], int(env.human_num))
            observed_human_range[1] = max(observed_human_range[1], int(env.human_num))
            if done:
                break
        else:
            raise RuntimeError(f"Random-action episode did not terminate for seed {seed}")

        print(f"seed={seed} rand_seed={env.rand_seed} initial_humans={initial_humans}")
        print(f"  observations={observation_shapes}")
        print(
            f"  steps={step_index + 1} terminal_info={type(info['info']).__name__} "
            f"info_types={dict(info_types)} humans_seen={observed_human_range}"
        )
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

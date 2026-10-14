# Vendored: CrowdNav simulator (`crowd_sim`)

| | |
|---|---|
| Upstream | https://github.com/vita-epfl/CrowdNav |
| Commit | `20d678085c06831e658a65b9e20c8bb6f6ecdc10` (2021-10-02, upstream `master` HEAD) |
| License | MIT, Copyright (c) 2018 VITA lab at EPFL — see `LICENSE` in this directory |
| Paper | Chen, Liu, Kreiss, Alahi. *Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning.* ICRA 2019. arXiv:1809.08835 |

## What is vendored

* `crowd_sim/` — the complete upstream simulation package (environment
  `CrowdSim`, agents, states, actions, outcome info objects, and the `orca` /
  `linear` / `none` policies). **Byte-identical** to the upstream commit; no
  file was edited.
* `configs/env.config`, `configs/policy.config` — the upstream default
  configuration files from `crowd_nav/configs/` (unchanged). RAISE reads
  `env.config` to configure `CrowdSim` and the `[action_space]` section of
  `policy.config` for the discrete CADRL/SARL action set.

## What is not vendored

* `crowd_nav/` (CADRL / LSTM-RL / SARL value networks, imitation learning,
  trainer, explorer). RAISE trains its own Stable-Baselines3 PPO policy; the
  pieces of `crowd_nav` it mirrors are re-implemented in
  `domains/crowdnav/` and cite the upstream source:
  * the discrete action set (`CADRL.build_action_space`),
  * the robot-centric observation (`CADRL.rotate`, applied per human as in
    `MultiHumanRL.transform`),
  * the evaluation statistics (`Explorer.run_k_episodes`).

## Runtime requirements of the vendored code

* `gym` — used only for the `gym.Env` base class and the `CrowdSim-v0`
  registration in `crowd_sim/__init__.py`.
* Python-RVO2 (`import rvo2`) — ORCA for the simulated humans. Not on PyPI;
  build from https://github.com/sybrenstuvel/Python-RVO2 (needs Cython,
  CMake and a C++ compiler). See `raise_env/README.md`.
* `numpy`, `matplotlib`.

`raise_paths.ensure_raise_paths()` puts this directory on `sys.path` so the
upstream absolute imports (`from crowd_sim.envs ...`) resolve unchanged.

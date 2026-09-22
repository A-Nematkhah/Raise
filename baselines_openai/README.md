# Vendored OpenAI Baselines (trimmed)

This tree is a **trimmed** copy of [OpenAI Baselines](https://github.com/openai/baselines)
(MIT — see `LICENSE` / `NOTICE.md`), vendored so RAISE can
`pip install -e ../baselines_openai` without a git submodule.

## What remains

RAISE only needs:

- `baselines.logger`
- `baselines.bench`
- `baselines.common.vec_env` (+ transitive helpers: `atari_wrappers`,
  `running_mean_std`, `tf_util`, …)

Upstream algorithm packages (`a2c`, `ppo1`/`ppo2`, `deepq`, `ddpg`, `her`,
`gail`, `trpo_mpi`, `acer`, `acktr`), demo assets, Docker/CI configs, and
the `baselines.run` CLI were removed from this fork. RAISE's own PPO/A2C live
under `raise_env/rl/{ppo,a2c}`.

## Install

```bash
pip install tensorflow   # setup.py asserts TF >= 1.4
pip install -e .
```

Optional Atari support (not used by CrowdSim) needs `opencv-python` if you
import `baselines.common.atari_wrappers`.

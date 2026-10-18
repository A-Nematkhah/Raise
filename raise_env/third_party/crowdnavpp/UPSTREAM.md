# Vendored: CrowdNav++ variable-human simulator

| | |
|---|---|
| Upstream | https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph |
| Commit | `390773137be04ed14e27620dc5fd7c5e1a5b1f62` |
| License | MIT, Copyright (c) 2023 Shuijing Liu — see `LICENSE` |
| Simulator | `crowd_sim.envs.crowd_sim_var_num.CrowdSimVarNum` |

## Included files

- The complete upstream `crowd_sim/` package, including `CrowdSimVarNum`.
- `crowd_nav/configs/config.py`, its package initializers, and `arguments.py`,
  which the Python config imports.
- The upstream `crowd_nav/policy/` modules imported by the simulator's policy
  factory (`orca`, `social_force`, `srnn`, and their factory/base module).

Every included file was extracted from the Git object for the commit above and
verified against its Git blob hash. No source edits were applied.

## Excluded

The upstream reinforcement-learning trainers, GST implementation, and all
prediction weights/checkpoints are excluded. RAISE uses its own SB3 PPO
trainer and selects the variable-human simulator without prediction.

The CrowdNav++ source repository uses the same top-level `crowd_sim` package
name as the previously vendored vita-epfl CrowdNav package. That legacy vendor
was removed after the wrapper migration; `raise_paths.py` exposes only this
directory and guards against loading the legacy package from a stale path.

# CrowdNav domain pack

## CrowdNav++ environment

The active simulator is the unmodified variable-human `CrowdSimVarNum` from
CrowdNav++ (`third_party/crowdnavpp`). It is configured from the upstream
Python `Config` object or a Python config file. Environment dimensions,
population, robot/human properties, sensor range/FOV, time limits and native
reward parameters come from that config. For example, the upstream default
Python config uses 20 humans with no initial count range, a 12 m square arena,
holonomic simulator configuration, a 5 m sensor range, and full-circle FOV.
RAISE defaults to unicycle kinematics and continuous control. A fixed
`human_num_override` can be supplied for tests and smoke runs; it sets the
configured human range to zero before unicycle compatibility adjustment.
Unicycle reset in upstream samples from 1 through `human_num +
human_num_range`, but checks the lower bound `human_num - human_num_range`;
the wrapper expands a too-small unicycle range to `human_num - 1` so these
invariants agree. Otherwise `sim.human_num` and
`sim.human_num_range` control population bounds and changes.

The upstream simulator resets global NumPy state. The wrapper serializes
simulator RNG access, restores the caller's global RNG after reset, and keeps
per-environment simulator RNG state between steps.

**Deliberate val-phase adaptation:** upstream `CrowdSimVarNum.step` computes
`human_future_traj` only for `phase == "test"` but requests future-danger
scoring in every phase. The wrapper resets val scenarios as `phase="val"` so
they retain the val seed offset, case set, and case counter, then switches the
internal stepping phase to `"test"` so val and test use the same future-danger
definition. No upstream source is modified.

The RAISE domain: LLM-proposed reward functions for crowd-aware robot
navigation, trained and evaluated with Stable-Baselines3 PPO.

## Task

- Robot start/goal positions and human population come from the configured
  scenario. Human counts can vary within configured bounds. The robot sees all
  humans in its sensor range/FOV, with visibility recorded per human;
  `robot.visible` controls whether humans see and respond to the robot.
- Episode ends on upstream `ReachGoal`, `Collision` or `Timeout`.
  Time step and time limit come from the active Python config; timeout is
  checked before the simulator advances its clock.
- Reward functions are LLM-proposed Python:
  `compute_reward(state, memory) -> total_reward` or
  `(total_reward, reward_components)`. `total_reward` trains PPO;
  `reward_components` is a flat diagnostic dict.

## State contract (`CrowdNavRewardState`)

Robot (`state.robot`): `px, py, vx, vy, gx, gy, theta, radius, v_pref,
kinematics` (world frame; `theta` is the heading, meaningful for unicycle
robots). Humans (`state.humans`) is a tuple containing every human, each with
`px, py, vx, vy, radius, visible` (world frame); `visible` follows the
configured robot sensing range and field of view.

Step facts include `action`, `collision`, `reached_goal`, `timeout`, `danger`,
`min_separation`, `discomfort_dist`, `goal_distance`, `progress`, `time_step`,
`global_time`, and `time_limit`. `success_reward`, `collision_penalty`,
`discomfort_penalty_factor`, `potential_delta`, and `potential_factor` carry
the parameters needed to reproduce the configured native reward.

- `min_separation` is the smallest current robot-human boundary gap checked by
  upstream reward calculation (negative on collision), not a swept-path
  distance. In the native danger branch the penalty is
  `(dmin - discomfort_dist) * discomfort_penalty_factor * time_step`.
- `danger` is the upstream `Danger` classification. In val/test, it can be
  based on predicted future human positions; the reward penalty still uses
  current `dmin`.
- `potential_delta` is the upstream potential change computed before movement;
  `potential_factor` is 2 for holonomic robots and 3 for unicycle robots.
- `progress` is the decrease of `goal_distance` this step (m).
- `action` is the executed command: `(vx, vy)` for holonomic robots, `(v, r)`
  (speed, heading change) for unicycle robots.
- Episode memory: plain `dict` cleared on reset.

The native reward follows upstream priority: timeout, collision, goal,
discomfort/danger, then potential shaping. Reward parameters are config-driven;
the upstream default config sets success to +10, collision to -20, discomfort
distance to 0.25 m, and discomfort factor to 10. Unicycle steps also receive
the upstream rotation and backwards-motion penalties. The wrapper exposes the
native value in `info["raise_native_reward"]`; the seed function reproduces it.

## Environment and policy observation

`RewardInjectedCrowdNavEnv` (gymnasium) wraps one `CrowdSimVarNum`:

- Observation is a `gym.spaces.Dict` matching upstream keys:
  `robot_node`, `temporal_edges`, `visible_masks`, `spatial_edges`, and
  `detected_human_num`. Human arrays are padded to the configured maximum
  population.
- Kinematics and action mode follow the active settings; RAISE defaults to
  unicycle with continuous control. The action-space prompt describes the
  active controls.
- PPO uses `MultiInputPolicy` with a CrowdNav-specific feature extractor: robot
  state and velocity form the context features, while a shared human encoder
  mean-pools only entries marked visible by `visible_masks`. Padded/invisible
  humans therefore do not contribute to the pooled representation.
- Scenarios: upstream phases. Training uses `train` cases, selection
  evaluations use configured `val` cases and the final blind evaluation uses
  configured `test` cases. Upstream seeds the global numpy RNG per case; the
  wrapper restores that RNG afterwards and preserves per-environment random
  state between steps.
- Rendering is off during training (`render_mode=None`); frames are drawn
  headlessly with matplotlib only for post-run animations.

## Fitness

Official fitness = the mean per-episode outcome score on held-out validation
scenarios: +1 for success, -1 for collision, and -0.5 for timeout, reported
with its standard error. Native CrowdNav++ task return remains a diagnostic,
not the selection objective. PPO's training discount is separate. Also recorded:
SR / CR / TR (success / collision / timeout rates), NT (navigation time of
successes), PL (path length of successes), ITR (danger frequency), SD (mean
minimum distance in danger).

SB3 2.9.0 treats a time-limit truncation as an episode boundary for PPO and
bootstraps the training target with `gamma * V(terminal_observation)`. Fitness
evaluation instead uses the observed episode outcome; a timeout receives -0.5
for fitness regardless of its native reward. This is an intentional difference
between PPO's training target and the reported outcome-score fitness.

## Stage I (Score1)

Score1 is a cheap sanity filter over a dataset of real CrowdNav++ rollouts
(`scripts/collect_crowdnav_stage1_dataset.py`: FOV-aware ORCA at several
clearance / speed / reaction settings, Linear, slow Linear, stop and random
robots on the same `train` scenarios). The collector supports holonomic and
unicycle kinematics; unicycle runs must satisfy CrowdNav++'s human-count
initialization follows the simulator's configured variable-human rules.
Score1 returns are discounted with PPO's γ.

```
safety  = 2·AUC − 1 of success ≻ timeout ≻ collision on discounted returns
quality = 0.40 · efficiency (Spearman of return vs −nav_time, successes)
        + 0.25 · comfort    (Spearman of return vs −danger fraction, successes)
        + 0.35 · step Spearman(step reward, step rule)
Score1  = 0.5·safety + 0.5·max(0, safety)·quality
```

Hard rejects: `no_signal`, `prefers_failure` (safety < 0), `score1_error`,
`score1_timeout`. When the active kinematics differ from the recorded rollout
kinematics, Score1 re-expresses recorded actions before evaluating them.

## Design principles

- Goal progress and arrival, collision avoidance, comfort distance to humans
- Interpretability (locals only; no extra signature args)

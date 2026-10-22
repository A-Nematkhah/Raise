# CrowdNav domain pack

## CrowdNav++ environment

The active simulator is the unmodified variable-human `CrowdSimVarNum` from
CrowdNav++ (`third_party/crowdnavpp`). It is configured from the upstream
Python `Config` object or a Python config file. Environment dimensions,
population, robot/human properties, sensor range/FOV, time limits and native
reward parameters come from that config. For example, the upstream default
Python config uses 20 humans with no initial count range, a 12 m square arena,
holonomic simulator configuration, a 5 m sensor range, and full-circle FOV.
RAISE defaults to unicycle kinematics and continuous control unless an
environment config is provided, in which case its kinematics is used by
default. The hard config explicitly selects holonomic kinematics and uses
`human_num=10` and `human_num_range=0`, so the effective population is
fixed at exactly 10, including at reset and during episodes. Upstream's unicycle reset
samples from 1 through `human_num + human_num_range` even though its step
invariant enforces the configured lower bound. Without changing vendored code,
the wrapper adjusts the initial population into the configured interval before
upstream finishes reset. `human_num_override` remains available for tests and
sets a fixed count.
The hard config uses a 30-second time limit, enables five-step constant-
velocity human path prediction at 0.25-second intervals, and defaults to
`robot.visible=False` (EvoNav-style: humans ignore the robot; ORCA among
humans only). Set `env.robot_visible: true` if humans should react to the
robot. The PPO policy observation's `spatial_edges` contains each visible
human's current relative position followed by five constant-velocity
predicted positions, all relative to the robot's current position. The
evolved reward state remains based on observed current human states, not
predicted positions. Density generalization uses
`evaluation_variants.humans_{10,15,20}` (plus `*_random` for randomized
human traits / goal changes).

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
- Evaluation reports timeouts independently (`TR` and per-episode outcome).
  Navigation time, path length and minimum human separation are stored per
  episode for analysis.
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
`min_separation`, `min_separation_post`, `discomfort_dist`, `goal_distance`,
`progress`, `time_step`, `global_time`, and `time_limit`. `success_reward`,
`collision_penalty`, `discomfort_penalty_factor`, `potential_delta`, and
`potential_factor` carry the parameters needed to reproduce the configured
native reward.

- `min_separation` is the upstream pre-step `dmin`: the smallest robot-human
  boundary gap at the start of the step (negative on collision), not a
  swept-path distance. In the native danger branch the penalty is
  `(dmin - discomfort_dist) * discomfort_penalty_factor * time_step`.
- `min_separation_post` is the same boundary gap recomputed from the
  post-step `robot` / `humans` positions in the snapshot.
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
Evaluation fitness is independent of the learned reward: mean outcome score
(+1 success, -1 collision, -0.5 timeout). Success, collision and timeout are
reported separately as SR, CR and TR; navigation time, successful path length,
and per-episode minimum human separation are also recorded.

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

## Stage I (EvoNav Score_1)

Score_1 is a training-free fitness for LLM reward candidates over a dataset of
real CrowdNav++ rollouts (`scripts/collect_crowdnav_stage1_dataset.py`: FOV-aware
ORCA at several clearance / speed / reaction settings, Linear, slow Linear, stop
and random robots on the same `train` scenarios). The collector supports
holonomic and unicycle kinematics.

```
Score_1(r) = (1/M) * sum_j (1/F_j) * sum_f Spearman( rank_rules[j,f], rank_r[j,f] )
```

- `rank_rules`: at each scenario `j` and frame `f`, rank the `N_traj` trajectories
  by analytic class order success (shorter path first) ≻ other/timeout (shorter
  distance-to-goal first) ≻ fail/collision (shorter distance-to-goal first), via
  a scalar rule score (class offset + within-class term; see
  `domains.crowdnav.score1.rule_score_at_frame`).
- `rank_r`: rank by the candidate's cumulative per-frame reward up to `f`
  (`cumsum`). Early-terminated trajectories carry forward their final class and
  cumulative reward.
- Ties use average ranks (`raise_core.rules`). Constant ranks → NaN Spearman;
  `nan_policy=zero` (default) counts those frames as 0, or `skip` omits them.
- Hard rejects / errors: exception, non-finite reward, or Score1 worker timeout
  → `score=-inf` with `reject_reason` (`score1_error`, `score1_timeout`, …).

CLI: `python -m domains.crowdnav.score1 --dataset PATH --candidates DIR_OR_JSON`.
When active kinematics differ from recorded rollouts, Stage I re-expresses
recorded actions before evaluating them.

## Design principles

- Goal progress and arrival, collision avoidance, comfort distance to humans
- Interpretability (locals only; no extra signature args)

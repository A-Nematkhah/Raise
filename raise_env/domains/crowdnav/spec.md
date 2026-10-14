# CrowdNav domain pack

The RAISE domain: LLM-proposed reward functions for crowd-aware robot
navigation in the **vita-epfl/CrowdNav** simulator (`crowd_sim`, vendored
unmodified at `third_party/crowdnav`, commit `20d6780`), trained and
evaluated with Stable-Baselines3 PPO.

## Task

- A robot (radius 0.3 m, `v_pref` 1 m/s) crosses a circle of radius 4 m to
  the antipodal goal through 5 ORCA-controlled humans (`circle_crossing`,
  upstream `env.config`). Humans do not see the robot (`robot.visible=false`).
- Episode ends on `ReachGoal` (success), `Collision` or `Timeout`
  (`time_limit` 25 s, `time_step` 0.25 s). Termination is environment-owned
  and identical to upstream `CrowdSim.step` (the timeout is checked before the
  clock advances, so a robot that never arrives times out at step 97).
- Reward functions are LLM-proposed Python:
  `compute_reward(state, memory) -> total_reward` or
  `(total_reward, reward_components)`. `total_reward` trains PPO;
  `reward_components` is a flat diagnostic dict.

## State contract (`CrowdNavRewardState`)

Robot (`state.robot`): `px, py, vx, vy, gx, gy, theta, radius, v_pref`
(world frame; `theta` is the heading, meaningful for unicycle robots).  
Humans (`state.humans`): tuple of `px, py, vx, vy, radius` (world frame;
iterate, do not index).  
Step facts: `action`, `collision`, `reached_goal`, `timeout`,
`min_separation`, `discomfort_dist`, `goal_distance`, `progress`,
`time_step`, `global_time`, `time_limit`.

- `min_separation` is upstream's `dmin` for this step: closest robot-human
  boundary distance along the step's relative motion (negative on collision).
  Upstream's discomfort penalty uses it: `(dmin - discomfort_dist) * 0.5 * time_step`
  when `dmin < discomfort_dist` (0.2 m).
- `progress` is the decrease of `goal_distance` this step (m).
- `action` is the executed command: `(vx, vy)` for holonomic robots, `(v, r)`
  (speed, heading change) for unicycle robots.
- Episode memory: plain `dict` cleared on reset.

The upstream reward (collision −0.25, goal +1, discomfort penalty) is the
seed reward and stays in `info["raise_native_reward"]`.

## Environment and policy observation

`RewardInjectedCrowdNavEnv` (gymnasium) wraps one `CrowdSim`:

- Observation: 41 floats, robot-centric as in upstream `CADRL.rotate`
  (x axis towards the goal): `[dg, v_pref, theta, radius, vx, vy]` then per
  human `[px, py, vx, vy, radius, distance, radius_sum]`.
- Actions (`--action-mode`, `--robot-kinematics`):
  - `discrete` (default): the 81 CADRL/SARL actions (stop + 16 directions ×
    5 exponentially spaced speeds, upstream `policy.config`).
  - `continuous`: `Box([-1, 1]^2)` mapped to `(vx, vy)` capped at `v_pref`
    (holonomic) or `(v, r)` with `r ∈ [-π/4, π/4]` (unicycle).
- Scenarios: upstream phases. Training uses `train` cases, selection
  evaluations `val` cases (100) and the final blind evaluation `test` cases
  (500). Upstream seeds the global numpy RNG per case; the wrapper restores
  that RNG afterwards. Stepping is deterministic.
- Rendering is off during training (`render_mode=None`); frames are drawn
  headlessly with matplotlib only for post-run animations.

## Fitness

Official fitness = mean discounted upstream task return over the evaluation
episodes (`Explorer.run_k_episodes`: `Σ_t 0.9^(t·time_step·v_pref) · r_t`
with the upstream reward), reported with its standard error. Also recorded:
SR / CR / TR (success / collision / timeout rates), NT (navigation time of
successes), PL (path length of successes), ITR (danger frequency), SD (mean
minimum distance in danger).

## Stage I (Score1)

Score1 is a cheap sanity filter over a dataset of real upstream rollouts
(`scripts/collect_crowdnav_stage1_dataset.py`: ORCA, cautious ORCA, Linear,
slow Linear, stop and random robots on the same `train` scenarios). Returns
are discounted with PPO's γ.

```
safety  = 2·AUC − 1 of success ≻ timeout ≻ collision on discounted returns
quality = 0.40 · efficiency (Spearman of return vs −nav_time, successes)
        + 0.25 · comfort    (Spearman of return vs −danger fraction, successes)
        + 0.35 · step Spearman(step reward, step rule)
Score1  = 0.5·safety + 0.5·max(0, safety)·quality
```

Hard rejects: `no_signal`, `prefers_failure` (safety < 0), `score1_error`,
`score1_timeout`. Dataset actions are holonomic and re-expressed as `(v, r)`
for unicycle runs.

## Design principles

- Goal progress and arrival, collision avoidance, comfort distance to humans
- Interpretability (locals only; no extra signature args)

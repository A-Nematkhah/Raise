# Highway domain pack

The only RAISE domain: LLM-proposed reward functions for `highway-fast-v0`,
trained and evaluated with Stable-Baselines3 PPO.

## Task

- Drive an ego vehicle on a multi-lane highway (`highway-fast-v0`).
- Survive without collision / off-road while making forward progress at speed.
- Reward functions are LLM-proposed Python:
  `compute_reward(state, memory) -> (total_reward, reward_components)`.
  `total_reward` trains RL; `reward_components` is a flat diagnostic dict.
- Termination stays environment-owned.

## State contract (`HighwayRewardState`)

Ego: `x, y, vx, vy, heading, speed, lane_index, on_road` (absolute road frame;
`lane_index` is derived from `y`, lane *i* centred at `y = 4·i`)  
Others: tuple of `x, y, vx, vy, heading` (nearest vehicles, **ego-relative**;
iterate, do not index)  
Flags: `collision`, `off_road`, `timeout`, `action`, `time_step`, `global_time`, `time_limit`  
- `timeout` is True only on the last step of an episode truncated at the time
  limit **without** collision/off-road, i.e. a survived (successful) episode.
- `action` is the integer meta-action index (DiscreteMetaAction:
  0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER) or the raw numeric
  array under ContinuousAction.

Shaping helpers: `progress` (forward m since last frame), `speed`  
Episode memory: plain `dict` cleared on reset.

## Design principles

- Forward progress and reasonable high speed
- Collision / off-road penalties and clearance shaping
- Interpretability (locals only; no extra signature args)

## Stage I (Score1)

Hybrid Score1 (not Spearman-only):

```
0.25 · within-traj Spearman(highway_rule, cum_reward)
0.40 · preference AUC (success ≻ timeout ≻ collision)
0.20 · throughput alignment on success (speed/progress)
0.15 · overtake alignment on success (episode_return vs overtake_count)
−0.50 · crawl/lag penalty
−0.35 · collision-decoy penalty
```

Trajectory label `timeout` is the historical name for a non-collision
failure (off-road). It is unrelated to `state.timeout`: off-road frames carry
`off_road=True, timeout=False`.

Dataset should include **safe_fast** successes and **crawl** successes so
Score1 can punish “survive by going slow”. Collect with:

```bash
python scripts/collect_highway_stage1_dataset.py
```

## LLM sandbox

Highway `RewardValidator` enables an **AST Attribute allowlist**: only
`HighwayRewardState` / `EgoVehicle` / `NearbyVehicle` fields (+ `memory.get`).
Hallucinated names (`lane_position`, `distance`, `robot`, …) are rejected
before smoke. D.1 includes negative few-shots. SeedVariant (`--llm seed`)
rotates clearance / lane / speed-band / crawl structures (not only coef
tweaks).

Runaway code is contained at every stage:

- Validation compiles and smoke-tests in a killable child before the parent
  ever executes the source; module level may only bind constants
  (`restrict_top_level`), so nothing heavy runs at import time.
- Dataset Score1 runs in a persistent child worker (`score1_worker.py`,
  60 s per candidate); a timeout rejects the candidate (`score1_timeout`).
- PPO labels run under a stall watchdog (`watchdog.py`): no env step for
  `highway_stall_seconds` (default 300) fails the label, not the run.

## Evaluation metrics

Mapped onto RAISE `ProxyMetrics`:

- SR — episodes survived (no collision, no off-road)
- CR — collision rate
- TR — off-road rate
- NT — mean episode duration (s)
- PL — mean forward progress (m)
- ITR — mean ego speed (m/s)
- SD — mean nearest-vehicle gap (m)

Logged continuous fields: `mean_speed`, `mean_progress`, `lane_change_rate`
(lane changes per step), `overtakes_per_km` (net passes: a vehicle that drops
back ahead again does not count), `speed_p10`/`speed_p90`, `soft_success`
(survive **and** average ≥25 m/s **and** travel ≥400 m).

## Selection fitness

Candidates are ranked by `highway_fitness` (`metrics.py`, holdout metrics
when present):

```
gate = sigmoid speed gate on speed_p10 (hard −penalty below the speed floor)
F = gate · (SR − CR − 0.5·TR) + 0.35·tanh(progress/800) + 0.25·tanh(speed/v_target)
  + 0.15·soft_success·gate − lag − degeneracy − low_speed penalties
```

The trainer stamps `fitness` on the candidate; the LLM sees the raw holdout
metrics plus this fitness (`raise_loop/proxy_feedback.py`).

## Run flow

1. **Gen0**: the LLM proposes N reward functions (one batched call); each passes
   AST policy → spawned smoke test → compile.
2. **Each generation g < G**: Score1 on the Stage I dataset; every new
   executable candidate is PPO-trained for K2 env steps and evaluated on E2
   holdout episodes (eval seeds shared within the generation; PPO warm-starts
   from a parent's checkpoint when that parent is still in the population);
   rank by fitness; the LLM breeds the next generation
   (crossover of the top 2, mutation of the lower half with its evaluation
   evidence, random restarts; the best genome is carried over with elitism).
3. **Stage III**: every candidate of the final population is trained once for
   K3 env steps and evaluated on E3 episodes; the final candidate is the best
   Stage III fitness. No LLM call.

Defaults: N=6, G=4, K2=12k, E2=20, K3=70k, E3=30 (`raise_core/pipeline.py`).

## Env / policy

- Gymnasium id: `highway-fast-v0`
- Policy: Stable-Baselines3 PPO (MLP); action space `meta_default`
  (DiscreteMetaAction), `meta_fine` or `continuous`

## Pack layout

| File | Role |
|------|------|
| `spec.md` | This document |
| `prompts.py` | D1/D2/D4/D5 templates for HighwayRewardState |
| `state.py` | Dataclasses + smoke states |
| `stage1.py` | Score1 / smoke factory |
| `score1_worker.py` | Dataset Score1 in a killable child (pack entry) |
| `env_wrapper.py` | Obs → state + reward injection |
| `action_config.py` | Action-space modes |
| `adapter.py` | Stage II/III SB3 PPO trainers |
| `metrics.py` | Eval metrics + `highway_fitness` |
| `watchdog.py` | PPO stall watchdog |
| `post_run.py` / `viz.py` | Plots + rollout GIF after a non-fast run |
| `pack.py` | `get_pack()` |

## Dependencies

```bash
pip install -r requirements.txt
```

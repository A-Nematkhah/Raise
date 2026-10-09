# Highway domain pack

Diagnostic / fast validation domain for RAISE (additive; **not** the paper baseline).

Default scientific domain remains ``crowdnav``. Use ``--domain highway`` to
exercise Score1 → Stage II/III wiring on a cheap simulator.

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

Trajectory label `timeout` is the CrowdNav-shared name for a non-collision
failure (off-road). It is unrelated to `state.timeout`: off-road frames carry
`off_road=True, timeout=False`.

Dataset should include **safe_fast** successes and **crawl** successes so
Score1 can punish “survive by going slow”. Collect with:

```bash
python scripts/collect_highway_stage1_dataset.py
```

## Surrogate (Stage II labels)

Fit targets (highway only; CrowdNav stays SR/CR/TR):

```
SR, CR, TR, mean_speed, mean_progress, soft_success
```

Gate / AL quality uses `highway_fitness` on predicted `y_hat`
(not SR−CR−0.5·TR alone). Labels store `fitness` (alias `selection_scalar`).
Highway features use schema version 2 (lane-index fix); older surrogate
datasets are relabeled instead of mixed.

Warm bootstrap (optional) and closed-loop refit pick these targets via
`--domain highway` / `ClosedLoopConfig.domain`.

## LLM sandbox (Phase 3)

Highway `RewardValidator` enables an **AST Attribute allowlist**: only
`HighwayRewardState` / `EgoVehicle` / `NearbyVehicle` fields (+ `memory.get`).
Hallucinated names (`lane_position`, `distance`, `robot`, …) are rejected
before smoke. D.1 includes negative few-shots; D.3 repair maps them to
legal substitutes. SeedVariant rotates clearance / lane / speed-band /
crawl structures (not only coef tweaks).

Runaway code is contained at every stage:

- Validation compiles and smoke-tests in a killable child before the parent
  ever executes the source; module level may only bind constants
  (`restrict_top_level`), so nothing heavy runs at import time.
- Dataset Score1 runs in a persistent child worker (`score1_worker.py`,
  60 s per candidate); a timeout rejects the candidate (`score1_timeout`).
- PPO labels run under a stall watchdog (`watchdog.py`): no env step for
  `highway_stall_seconds` (default 300) fails the label, not the run.

## Closed-loop / Stage III (Phase 4)

- Hard Surrogate gate (highway profile): `n_labeled ≥ 16` **or** mean val
  MAE ≤ `0.35` (`--closed-loop-max-val-mae-gate`). Until then gate is soft.
- Parent order for next gen (`evolve_rank=pareto`, highway default):
  feasibility (SR > 0) + NSGA-II fronts with crowding over
  `−CR, −TR, progress, mean_speed, lane_change_rate, overtakes_per_km`
  (each extra toggled by `--no-highway-pareto-use-*`). Calibration default is
  `no_speed_floor`. Only CR/TR comparisons get a 2×Bernoulli-SE noise margin.
  `evolve_rank=scalar` (legacy `highway_fitness`) and `hybrid` remain options.
- Stage III input = gated kept ∪ best Stage II ∪ best fitness;
  Score1-best is forced only if `soft_success` / fitness look strong;
  unlabeled Score1-best is protected from surrogate drop.
- Highway profile Stage II budget: **K2=12k env steps, E2=20** holdout episodes
  (holdout seeds shared by every candidate in a generation).
- Final pick is a human choice from Pareto front 0
  (`scripts/eval_raise_checkpoint.py --pareto-front`), not an automatic scalar.
- Epoch logs include Score1 spread, soft_success μ, fitness μ, best_ever,
  cruise-plateau fraction.

## Evaluation metrics

Mapped onto RAISE `ProxyMetrics`:

- SR — episodes survived (no collision, no off-road)
- CR — collision rate
- TR — off-road rate
- NT — mean episode duration (s)
- PL — mean forward progress (m)
- ITR — mean ego speed (m/s) (CrowdNav uses intrusion %)
- SD — mean nearest-vehicle gap (m)

Primary **selection** for the next generation is **Pareto ranking**
(`evolve_rank=pareto`, see above). `highway_fitness` is still computed and
logged as a legacy scalar.

`soft_success` = fraction of episodes that survive **and** average ≥25 m/s
(= `V_TARGET`) **and** travel ≥400 m (eval diagnostic only; never a Pareto
or selection objective).

Logged continuous fields: `mean_speed`, `mean_progress`, `lane_change_rate`
(lane changes per step), `overtakes_per_km` (net passes: a vehicle that drops
back ahead again does not count), `speed_p10`/`speed_p90`. Surrogate targets
are the six listed above.

## Env / policy

- Gymnasium id: `highway-fast-v0`
- Policy: Stable-Baselines3 PPO (MLP)
- Stage II: short PPO proxy; Stage III: longer PPO; **no human H-sweep**
- Full RAISE path: warm surrogate bootstrap + closed-loop (Score1 ↔ Stage II
  short ↔ Surrogate gate / AL / proxy feedback) then Stage III
  (`scripts/run_raise_highway.py`)

## Pack layout

| File | Role |
|------|------|
| `spec.md` | This document |
| `prompts.py` | D1/D2/D3/D5 templates for HighwayRewardState |
| `state.py` | Dataclasses + smoke states |
| `stage1.py` | Score1 / smoke factory |
| `score1_worker.py` | Dataset Score1 in a killable child (pack entry) |
| `env_wrapper.py` | Obs → state + reward injection |
| `adapter.py` | Stage II/III SB3 trainers |
| `watchdog.py` | PPO stall watchdog |
| `pack.py` | `get_pack()` |

## Dependencies

Install separately (does not replace CrowdNav pins):

```bash
pip install -r requirements_highway.txt
```

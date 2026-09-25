# Highway domain pack

Diagnostic / fast validation domain for RAISE (additive; **not** the paper baseline).

Default scientific domain remains ``crowdnav``. Use ``--domain highway`` to
exercise Score1 → Stage II/III wiring on a cheap simulator.

## Task

- Drive an ego vehicle on a multi-lane highway (`highway-fast-v0`).
- Survive without collision / off-road while making forward progress at speed.
- Reward functions are LLM-proposed Python: `compute_reward(state, memory) -> float`.
- Termination stays environment-owned.

## State contract (`HighwayRewardState`)

Ego: `x, y, vx, vy, heading, speed, lane_index, on_road`  
Others: tuple of `x, y, vx, vy, heading` (nearest vehicles; iterate, do not index)  
Flags: `collision`, `off_road`, `timeout`, `action`, `time_step`, `global_time`, `time_limit`  
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
0.35 · throughput alignment on success (speed/progress)
− 0.50 · crawl_penalty
```

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

Warm bootstrap (optional) and closed-loop refit pick these targets via
`--domain highway` / `ClosedLoopConfig.domain`.

## LLM sandbox (Phase 3)

Highway `RewardValidator` enables an **AST Attribute allowlist**: only
`HighwayRewardState` / `EgoVehicle` / `NearbyVehicle` fields (+ `memory.get`).
Hallucinated names (`lane_position`, `distance`, `robot`, …) are rejected
before smoke. D.1 includes negative few-shots; D.3 repair maps them to
legal substitutes. SeedVariant rotates clearance / lane / speed-band /
crawl structures (not only coef tweaks).

## Closed-loop / Stage III (Phase 4)

- Hard Surrogate gate: `n_labeled ≥ 16` **or** mean val MAE ≤ `0.35`
  (`--closed-loop-max-val-mae-gate`). Until then gate is soft.
- Parent order for next gen (`evolve_rank=scalar`, highway default):
  labeled genomes by official `highway_fitness` (holdout); unlabeled by Score1.
  Optional diagnostic: `evolve_rank=pareto` (feasibility + NSGA-II).
- Stage III input = gated kept ∪ best Stage II ∪ best fitness;
  Score1-best is forced only if `soft_success` / fitness look strong;
  unlabeled Score1-best is protected from surrogate drop.
- Stage II eval default for 4h profile: **E2=20** (K2 stays 15k).
- Epoch logs include Score1 spread, soft_success μ, fitness μ, best_ever,
  cruise-plateau fraction.

## Evaluation metrics

Mapped onto RAISE `ProxyMetrics`:

- SR — episodes completed without collision
- CR — collision rate
- TR — off-road (or other early failure) rate when distinguished; else 0
- NT / PL / ITR / SD — filled with highway proxies (time, distance, 0, min gap)

Primary **selection** for the next generation is **`highway_fitness`**
(`evolve_rank=scalar`). Pareto ranking remains available as a diagnostic mode.

`soft_success` = fraction of episodes that survive **and** average ≥25 m/s
(= `V_TARGET`) **and** travel ≥400 m (eval diagnostic).

Logged continuous fields: `mean_speed`, `mean_progress`, `lane_change_rate`,
`high_speed_frac`, `speed_p10`/`speed_p90`. Surrogate labels remain SR/CR/TR.

## Env / policy

- Gymnasium id: `highway-fast-v0`
- Policy: Stable-Baselines3 PPO (MLP)
- Stage II: short PPO proxy; Stage III: longer PPO; **no human H-sweep**
- Full RAISE path: warm surrogate bootstrap + closed-loop (Score1 ↔ Stage II
  short ↔ Surrogate gate / AL / proxy feedback) then Stage III
  (`scripts/run_raise_highway_4h.py`)

## Pack layout

| File | Role |
|------|------|
| `spec.md` | This document |
| `prompts.py` | D1/D2/D3/D5 templates for HighwayRewardState |
| `state.py` | Dataclasses + smoke states |
| `stage1.py` | Score1 / smoke factory |
| `env_wrapper.py` | Obs → state + reward injection |
| `adapter.py` | Stage II/III SB3 trainers |
| `pack.py` | `get_pack()` |

## Dependencies

Install separately (does not replace CrowdNav pins):

```bash
pip install -r requirements_highway.txt
```

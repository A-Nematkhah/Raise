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

## Evaluation metrics

Mapped onto RAISE `ProxyMetrics`:

- SR — episodes completed without collision
- CR — collision rate
- TR — off-road (or other early failure) rate when distinguished; else 0
- NT / PL / ITR / SD — filled with highway proxies (time, distance, 0, min gap)

Primary selection scalar (highway):

```
SR - CR - 0.5·TR
+ 0.35·tanh(progress_m / 800)
+ 0.25·tanh(mean_speed / 25)
+ 0.15·soft_success
- crawl_penalty   # if SR high but mean_speed < 12 m/s
```

`soft_success` = fraction of episodes that survive **and** average ≥15 m/s
**and** travel ≥400 m. This stops “crawl forever to inflate SR”.

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

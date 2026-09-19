# CrowdNav domain pack

Robot crowd navigation for EvoNav Algorithm 1 (baseline). This pack is the
**default** domain; reward-search must behave identically to the pre-domains
baseline when `domain=crowdnav`.

## Task

- Guide a holonomic robot to a 2D goal while avoiding dynamic humans.
- Reward functions are LLM-proposed Python: `compute_reward(state, memory) -> float`.
- Termination (collision / goal / timeout) stays environment-owned.

## State contract (`RewardState`)

Robot: `px, py, vx, vy, radius, gx, gy, v_pref`  
Humans: tuple of `px, py, vx, vy, radius` (iterate; do not index as a sequence)  
Flags / scalars: `dmin`, `discomfort_dist`, `collision`, `reaching_goal`, `timeout`,
`action`, `time_step`, `global_time`, `time_limit`  
Episode memory: plain `dict` cleared on reset (no classes on state).

## Design principles for rewards

- Goal progress (dense shaping)
- Collision / proximity safety
- Interpretability (clear locals; no extra signature args)

## Evaluation metrics

SR, CR, TR, NT, PL, ITR, SD — primary selection scalar: `SR - CR - 0.5·TR`.

## Env / policy (this pack)

- Inferred obs: `CrowdSimPredRealGST-v0` + GST wrapper
- No prediction: `CrowdSimVarNum-v0`
- Policy: `selfAttn_merge_srnn`
- Stage II: A2C proxy; Stage III: PPO

## Pack layout

| File | Role |
|------|------|
| `spec.md` | This document (task + contracts for humans / future prompt fill) |
| `prompts.py` | Appendix D templates (canonical) |
| `pack.py` | `get_pack()` → `DomainPack` |
| `stage1.py` | Score1 / smoke factory for Stage I |
| `adapter.py` | Stage II/III façade over existing trainers |

How to add another real domain (no stubs): see [`../README.md`](../README.md).

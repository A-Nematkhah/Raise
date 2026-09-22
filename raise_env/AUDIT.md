# AUDIT.md — CrowdNav++ / Evonav environment

Source: local copy of [Shuijing725/CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
(Liu et al., ICRA 2023). See `NOTICE.md` and `LICENSE`.

This audit describes the **current** tree after the Phase-1 reward adapter
refactor (`crowd_nav/reward_search/`). Default scalar reward behavior is
unchanged (`LegacyReward`).

---

## 1. Episode lifecycle — `crowd_sim/envs/crowd_sim.py`

Base class `CrowdSim` (`gym.Env`). Training envs inherit
`CrowdSimVarNum` → `CrowdSimPred` / `CrowdSimPredRealGST`, which override
`reset` / `step` / `calc_reward` but share the same detection + reward split.

### `configure(config)`
Loads time limits, reward coeffs, FOV, human-goal randomization flags,
prediction buffers, then `set_robot(Robot(...))`. Installs
`LegacyReward.from_crowd_sim(self)` unless a `reward_fn` was
constructor-injected.

### `reset(phase='train', test_case=None)`
1. Clears humans; seeds from phase capacity + case counter + `thisSeed`.
2. `generate_robot_humans(phase)` places robot + circle-crossing humans.
3. Sets `agent.time_step` / policy timesteps.
4. Advances `case_counter`.
5. `generate_ob(reset=True)`.
6. Initializes potential:
   `potential = -‖robot_pos − goal‖`.
7. Calls `reward_fn.reset()` and, for `LegacyReward`,
   `sync_potential(self.potential)`.

### `step(action, update=True)`
1. Clip action via `robot.policy.clip_action`.
2. `get_human_actions()` — each human gets ObservableStates of others
   (FOV-gated; robot included only if `robot.visible`).
3. `calc_reward(action)` → `(reward, done, episode_info)`.
4. Apply `robot.step` + human steps; advance `global_time`, `step_counter`.
5. `generate_ob(reset=False)`.
6. Optionally random mid-episode / end-goal human goal changes.

### `calc_reward(action)` (post-refactor)
**Env-owned detection (never delegated to `reward_fn`):**
- `dmin`, `collision` from pairwise robot–human clearance
  (`‖Δp‖ − r_robot − r_human`).
- `reaching_goal` if ‖pos − goal‖ < `robot.radius`.
- `timeout` if `global_time >= time_limit - 1`.

**Termination / `episode_info` (env-owned):**
| Condition | `done` | `episode_info` |
|-----------|--------|----------------|
| timeout | True | `Timeout()` |
| collision | True | `Collision()` |
| reaching_goal | True | `ReachGoal()` |
| `dmin < discomfort_dist` | False | `Danger(dmin)` |
| else | False | `Nothing()` |

**Scalar reward:** `reward_fn.compute(RewardState(...))` only.
Default `LegacyReward` reproduces the original formula
(success / collision / discomfort / potential shaping + unicycle extras).

### State definitions (`crowd_sim/envs/utils/state.py`)
- **`FullState`**: `(px, py, vx, vy, radius, gx, gy, v_pref, theta)` — robot.
- **`ObservableState`**: `(px, py, vx, vy, radius)` — humans (and robot when
  visible to humans).
- Pluggable reward snapshot uses `crowd_nav.reward_search.state.RewardState`
  with `HumanObservable` mirroring `ObservableState`, humans filtered to
  those within `robot.sensor_range` (`Rsense`).

---

## 2. Reward-related config — `crowd_nav/configs/config.py`

Fields **read by** (Legacy) `calc_reward` / `LegacyReward`:

| Field | Default (root `config.py`) | Role |
|-------|----------------------------|------|
| `reward.success_reward` | `10` | Goal terminal reward |
| `reward.collision_penalty` | `-20` | Collision terminal reward |
| `reward.discomfort_dist` | `0.25` | Personal-zone radius (m) |
| `reward.discomfort_penalty_factor` | `10` | Discomfort shaping scale |
| `env.time_step` | `0.25` | Multiplies discomfort term |
| `action_space.kinematics` | `"holonomic"` | Unicycle adds spin/back penalties |

Also used around reward / episode length (not all inside `LegacyReward`):
- `env.time_limit = 50`
- `reward.gamma = 0.99` (RL discount; not inside `calc_reward`)
- `self.potential` (env state) for progress shaping

`CrowdSimVarNum` Legacy coeffs differ slightly: `pot_factor` 2 (holonomic) /
3 (unicycle); unicycle spin coef `-4.5` vs base `-5`.

---

## 3. Policy registry — `crowd_nav/policy/policy_factory.py`

```python
policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE
policy_factory['srnn'] = SRNN                    # DS-RNN-style baseline
policy_factory['selfAttn_merge_srnn'] = selfAttn_merge_SRNN  # CrowdNav++
```

Selection:
- Humans: `config.humans.policy` (usually `"orca"`).
- Robot: `config.robot.policy` (`"orca"` / `"social_force"` / `"srnn"` /
  `"selfAttn_merge_srnn"`).

`Agent.__init__` does `policy_factory[subconfig.policy](config)`.

**Phase 1+ note:** LLM / candidate rewards plug in via `env.reward_fn` /
`set_reward_fn(...)` and must **not** be registered here. Do not put reward
logic in `policy_factory` — keep termination env-owned and rewards on
`RewardFunction`.

---

## 4. CLI flags — `arguments.py`

| Flag | Default | Notes |
|------|---------|-------|
| `--algo` | `ppo` | Must be `a2c` \| `ppo` \| `acktr` |
| `--num-env-steps` | `20e6` | Total env steps; comment says 10e6 holonomic / 20e6 unicycle |
| `--num-processes` | `16` | Parallel envs |
| `--num-steps` | `30` | Rollout length per update |
| `--env-name` | `CrowdSimPredRealGST-v0` | Must match `sim.predict_method` |
| `--seed` | `425` | |
| `--no-cuda` | off | |
| `--output_dir` | `trained_models/my_model` | |

**Randomization is not a CLI flag.** Toggle in `config.py` (and saved
`trained_models/*/configs/config.py`):

| Paper setting (Sec. 5.1) | Config |
|--------------------------|--------|
| **With random** | `env.randomize_attributes = True` **and** `humans.random_goal_changing = True` |
| **Without random** | both `False` (see `ORCA_no_rand` / `SF_no_rand` / `GST_predictor_non_rand`) |

`Agent.sample_random_attributes()`: `v_pref ~ U(0.5, 1.5)`,
`radius ~ U(0.3, 0.5)` when `randomize_attributes` is True.
`humans.end_goal_changing` stays True in both paper settings (continuous flow).

There is **no** `--randomize-attributes` argparse flag.

---

## 5. Paper vs repo config divergences (Phase 1 checklist)

Paper (Sec. 5 / setup): **12 m × 12 m** workspace, **H ≤ 20**, **vmax = 1**,
robot radius **ρ₀**, **Rsense = 5 m**, holonomic, invisible robot, ORCA humans.

| Claim | Repo | Match? |
|-------|------|--------|
| Workspace 12×12 | `sim.arena_size = 6` → approx. **[-6, 6]²** (12 m side, origin-centered; not `[0,12]²`) | Equivalent extent, different origin |
| H up to 20 | `sim.human_num = 20`, `human_num_range = 0` | Yes (fixed 20; range allows variable count if set) |
| vmax = 1 | `robot.v_pref = 1`, `humans.v_pref = 1` (non-rand) | Yes |
| ρ₀ | `robot.radius = 0.3`, `humans.radius = 0.3` | Yes (paper “ρ”; 0.3 m) |
| Rsense = 5 | `robot.sensor_range = 5` | Yes |
| Holonomic | `action_space.kinematics = "holonomic"` | Yes |
| Invisible robot | `robot.visible = False` | Yes |
| Circle spawn | `sim.circle_radius = 6 * √2 ≈ 8.49` | Repo-specific spawn geometry |

**Default root `config.py` is the “with random” setting**
(`randomize_attributes=True`, `random_goal_changing=True`) plus GST
(`sim.predict_method='inferred'`, `CrowdSimPredRealGST-v0`). Paper Table I
metrics use that regime; Table II / `*_no_rand` checkpoints turn randomization
off.

Other Phase-1 awareness items:
- Prediction-aware social reward (`r_pred`) lives in `CrowdSimPred.calc_reward`
  (adds future-collision penalties on top of `super()`). GST path
  (`CrowdSimPredRealGST`) deliberately skips that until wrapper predictions
  exist.
- Pluggable rewards: inject via `CrowdSim(reward_fn=...)` or
  `env.set_reward_fn(...)`; sandbox LLM code via
  `crowd_nav.reward_search.sandbox.RewardValidator`.

---

## 6. Local smoke confirmation (this machine)

| Check | Result |
|-------|--------|
| Unit tests `crowd_nav/reward_search/tests/test_reward_adapter.py` | **16 passed** |
| `train.py --algo ppo --num-env-steps 3000 --num-processes 1 --seed 425 --no-cuda` | OK; checkpoints `trained_models/smoke_legacy/{00000,00099}.pt` |
| `test.py` → `ORCA_no_rand` / `00000.pt` (500 eps) | **SR 0.78, NT 15.87, PL 18.53, ITR 26.04%, SD 0.36** — matches paper Table II ORCA (no rand) exactly |
| GPU torch verification | System Python 3.10.11, `torch 2.11.0+cu128`, `torch.cuda.is_available() == True`; NVIDIA driver 610.62, CUDA UMD 13.3, RTX 3050 4GB |
| Project environment | `.venv` was removed on 2026-09-03 at the user's request; recreate it from `requirements_pinned.txt` before a full isolated run |

Do not mix this venv with the main Evonav project dependencies.

---

## 7. Stage II A2C vs `train.py --algo` (intentionally out of scope)

**Verdict (cosmetic CLI gap, not a Stage II correctness bug).**

| Path | Optimizer | Default |
|------|-----------|---------|
| Stage II `RealPolicyTrainer.train_and_eval` | `rl.a2c.A2C` when `Stage2Config.algo == "a2c"` (default); shares `Policy` / `RolloutStorage` / `make_vec_envs` with `train.py` | A2C (RMSprop, no clip) |
| Stage III `RealPolicyTrainer.train_and_eval` | always `rl.ppo.PPO` | PPO (Adam + clip) |
| Standalone `train.py` | **always** `ppo.PPO` — ignores `--algo` after `arguments.py` validates it | PPO |

Evidence:
- `stage2.py` branches on `config.algo.lower() == "a2c"` and constructs `A2C(...)`; else `ppo.PPO(...)`.
- `pipeline.py` builds `Stage2Config(...)` without overriding `algo`, so the Table-5 default `"a2c"` is used.
- `arguments.py` already allows `--algo a2c|ppo|acktr`; `train.py` never switches on that flag.

**Out of scope:** wiring `--algo a2c` into `train.py`'s agent construction. Algorithm 1 Stage II/III do not call `train.py`; they use their own loops. Standalone A2C via `python train.py --algo a2c` is unused for Table 1/2 reproduction. Revisit only if a baseline is trained through that CLI outside the Stage II loop.

---

## 8. Decisions (pre–GPU validation pass)

### 8.1 `EVOLUTION_RANDOMIZATION_REGIME` (first pass)

**Value:** `without_random` (enum: `without_random` | `with_random` | `both`).

**Rationale (verbatim):** (a) it's the only regime already validated end-to-end in the Phase 0 smoke test (ORCA_no_rand matched the paper's table exactly), (b) it has lower variance, making it easier to tell a real pipeline bug from noise before scaling up, (c) "with_random" is deferred to a second pass once "without_random" produces sane Table 1/2 numbers.

**Propagation:** single flag in pipeline / paper-scale config → `collect_stage1_dataset.py` (`randomize_attributes` / `random_goal_changing`), Stage II/III env construction, and final evaluation. Do **not** run `both` unless the budget is deliberately doubled (two full passes).

**GST checkpoint for this regime** (when `predict_method=inferred`):
`gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj`
(loads `…/checkpoint/epoch_100.pt` — **not** the `_rand` sibling).

### 8.2 Observation-space parity (Stage II/III vs CrowdNav++)

**Choice: (a)** — `predict_method="inferred"` for Stage II/III training and final eval, loading the GST checkpoint selected by §8.1 / §8.3, so evolved-reward policies see the same observation space (including GST-filled `m_t` / spatial edges) as CrowdNav++. Required for an honest Table 1 “reward-function-only” comparison.

**Why not (b):** keeping `predict_method="none"` would force relabeling every report as a non-predictive variant and would **not** reproduce paper Table 1 architecture match. GST cost at Stage II (G2×N short K2 runs) is acceptable for the first validation pass; Stage I Score1 remains predict-method-independent (dataset of `RewardState` only; collector may still use `predict_method=none` for ORCA/SF rollouts because it never trains a GST policy).

**Stage I:** collection and Score1 stay free of a GST GPU dependency. Collector applies randomization flags from the regime but does not wrap with `VecPretextNormalize`.

### 8.3 `pred.model_dir` ↔ regime

At startup of dataset collection, Stage II, Stage III, and final evaluation: if `predict_method == "inferred"`, assert the GST path’s `_rand` / no-suffix matches `EVOLUTION_RANDOMIZATION_REGIME` and fail loudly on mismatch. If `predict_method == "none"` (collector / any explicit non-predictive path), skip the assert and log that GST-regime consistency is N/A.

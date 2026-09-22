# RAISE Project Technical Report

**Audience:** Thesis supervisor  
**Repository:** RAISE (RAISE Algorithm 1 faithful replication on CrowdNav++)  
**Report basis:** Source code, configs, scripts, saved `results/`, evaluation JSON/MD, `AUDIT.md`, and git history  
**Code HEAD at report time:** `fa3325d9cb618498a204903f2040a7c51aa0f55e`  
**Report date:** 2026-09-11  

**Status legend used below**

| Label | Meaning |
|-------|---------|
| **Implemented & Validated** | Present in code and backed by saved outputs / documented local checks |
| **Implemented but Not Fully Validated** | Present in code; end-to-end or paper-scale evidence incomplete |
| **Partially Implemented** | Present with reduced budgets, stubs, or incomplete artifacts |
| **Planned / Conceptual** | Documented as future / explicitly excluded from this release |

---

## 1. Executive Summary

This repository implements a **reward-function search pipeline** for robot crowd navigation: an LLM proposes Python reward functions; a sandbox validates them; Stage I ranks them analytically on a fixed trajectory dataset (Score1); Stages II–III train a neural navigation policy under each candidate reward (short A2C proxy, then longer PPO) and refine rewards with LLM feedback. The simulation stack is a CrowdNav++ derivative (`raise_env/`), not a new simulator. **AMFRS** (multi-objective evolutionary extensions described in thesis-oriented docs) is **explicitly not included** in this release (`README.md`, `pipeline.py` notes).

The pipeline is **end-to-end runnable** and has been executed locally with real LLM (Groq), CUDA training, and GST-inferred observations. The most complete and independently re-evaluated run is `results/run_scaled_h5_gst` plus the P0 evaluation suite under `results/run_scaled_h5_gst/evals/`. Under that protocol (H=5 humans, GST on, without_random, E=150, three **evaluation** seeds), the best RAISE checkpoint reaches mean **SR ≈ 0.65**, but is **outperformed** by pretrained CrowdNav++ GST, ORCA, and Social Force baselines evaluated under the same H=5 protocol. Paper-scale budgets (K3=10⁷, multi-seed Algorithm-1 training) are **not** evidenced by completed non-stub runs in this repository.

**Current project status:** working **experimental research framework** for Algorithm 1 replication and debugging—not a validated claim that RAISE rewards beat CrowdNav++ under paper settings.

---

## 2. Problem Definition

| Element | As implemented in this repo |
|---------|------------------------------|
| **Problem** | Learn / search a scalar reward so that RL yields a robot policy that reaches a goal in a crowd while avoiding collisions (crowd navigation). |
| **Environment** | CrowdSim Gym envs (`crowd_sim/envs/`), primarily `CrowdSimPredRealGST-v0` when `predict_method=inferred`. Arena ~12 m extent (`sim.arena_size=6`), default paper-like setup in `crowd_nav/configs/config.py`. |
| **Agent** | Holonomic robot; policy network `selfAttn_merge_srnn` (CrowdNav++ attention-SRNN). Humans typically ORCA (`humans.policy="orca"`). |
| **Observation** | Graph / dict observation for SRNN; with GST wrapper (`VecPretextNormalize`), includes inferred future human trajectories when `use_wrapper=True`. |
| **Action space** | Continuous holonomic `ActionXY` (vx, vy), clipped by preferred speed (`srnn.clip_action`). |
| **Reward** | Pluggable `RewardFunction.compute(RewardState) → float`. Default `LegacyReward`; evolved candidates are sandboxed LLM functions. **Termination** (collision / goal / timeout) remains **environment-owned**. |
| **RL** | Stage II default **A2C**; Stage III **PPO**; standalone `train.py` also PPO. |
| **Objective (search)** | Stage I: maximize Score1 ∈ [−1,1]. Stage II/III selection scalar: `SR − CR − 0.5·TR` (`selection.py`). |
| **Evaluation criteria** | SR, CR, TR, NT, PL, ITR, SD (as logged in proxy/full metrics and P0 JSON). |

---

## 3. Overall Framework

### 3.1 Pipeline that actually exists

```text
CLI (run_raise.py / paper_scale)
  → RaisePipeline (pipeline.py)
       → LLM + seed reward (prompts D5 / llm.py)
       → RewardValidator sandbox
       → Stage I: StageIEvolver + Score1(dataset)
       → Stage II: Stage2Runner + A2C + D.3 refine
       → Stage III: Stage3Runner + PPO + D.3 refine (+ H-sweep if enabled)
       → JSON artifacts under results/<run>/
```

Mapped to the requested template (**only where real**):

| Template step | Status | Module |
|---------------|--------|--------|
| Input | Implemented | CLI + `RaiseRunConfig` + Stage I dataset |
| Candidate / reward generation | Implemented | `llm.py`, `prompts.py`, `evolver.py` |
| Validation | Implemented | `sandbox/` |
| Cheap evaluation | Implemented | Stage I Score1 (`scoring.py`) — offline, not a learned surrogate |
| Surrogate / proxy | **Proxy = Stage II short RL** (procedural). **Learned surrogate: Not Implemented** | `stage2.py` |
| RL training | Implemented | Stage II A2C / Stage III PPO |
| Evaluation | Implemented | `evaluate_proxy_policy` + scripts under `scripts/eval_*`, `run_p0_*` |
| Selection / ranking | Implemented | Score1 ranks; `selection.py` nav scalar; `best_trained` / elitism |
| Final reward | Implemented but selection can disagree with best eval | `final_candidate.json` |

### 3.2 ASCII diagram (actual)

```text
Groq/Ollama/seed LLM
        │
        ▼
 sandbox validate ─────────────────────────────┐
        │                                        │
        ▼                                        │
 Stage I Score1 on fixed traj dataset            │
        │                                        │
        ▼                                        │
 Stage II: fresh A2C (K2) + short eval           │
        │         ▲                              │
        │    D.3 refine (LLM) ───────────────────┘
        ▼
 Stage III: fresh PPO (K3) + full-episode eval
        │         ▲
        │    D.3 refine (optional)
        ▼
 checkpoints + metrics + optional H-sweep
        │
        ▼ (separate scripts)
 P0 suite: re-eval vs ORCA / SF / CrowdNav++ GST
```

**No active-learning loop and no ExtraTrees/RF surrogate appear in the execution path.**

---

## 4. Detailed Algorithm

### Stage 1 — Analytical evolution (**Implemented & Validated** on real runs)

| Item | Evidence |
|------|----------|
| **Input** | LLM; `D5_SEED_FUNCTION`; Stage I dataset path (`data/stage1_dataset`, manifest `n_scenarios: 100`) |
| **Processing** | Gen0 batch generation → sandbox → Score1 → G generations of crossover / mutation / random + reflection text |
| **Algorithm** | Score1 = mean over scenarios of mean over frames of Spearman(rule ranks, cumulative recomputed rewards) (`scoring.py`, `rules.py`) |
| **Output** | `stage1_population.json`, `best_stage1.json`, `stage1_rejections.jsonl` |
| **Files** | `evolver.py`, `scoring.py`, `rules.py`, `dataset.py`, `sandbox/*`, `prompts.py` |
| **Cost** | Dominated by LLM calls + CPU Score1 (no policy training) |
| **Why** | Cheap filter before RL (RAISE Algorithm 1 design) |

Example from `run_scaled_h5_gst`: N=8, G1=8, best Score1 id `cro_0009`, score ≈ **0.831**.

### Stage 2 — Proxy A2C + D.3 refine (**Implemented & Validated** with reduced budgets)

| Item | Evidence |
|------|----------|
| **Input** | Stage I population (`RewardCandidate` + `reward_fn`) |
| **Processing** | Each round: train **fresh** A2C for K2 steps with injected reward → eval E2 episodes at horizon T_short → LLM refine → sandbox (+ one repair) |
| **Algorithm** | A2C via `rl.a2c`; same `Policy` / `RolloutStorage` / `make_vec_envs` pattern as `train.py` (`stage2.py`) |
| **Output** | `stage2_train/rXX_*/proxy_metrics.txt`, checkpoints, `stage2_population.json`, `best_stage2.json` |
| **Files** | `stage2.py`, `selection.py`, `rl/*` |
| **Cost** | High: N × G2 × (K2 env steps); e.g. scaled run: 8 rounds × 8 candidates, K2=8000 |
| **Why** | Rank rewards by short on-policy training before expensive PPO |

### Stage 3 — PPO + refine + H-sweep (**Implemented but Not Fully Validated** vs paper)

| Item | Evidence |
|------|----------|
| **Input** | Stage II population |
| **Processing** | Fresh PPO for K3 steps → full-episode eval → D.3 refine → optional H-sweep |
| **Algorithm** | `rl.ppo.PPO` (Adam, clipped surrogate); fixed update schedule (no early stop) |
| **Output** | `stage3_train/.../full_metrics.txt`, checkpoints, `best_stage3.json`, `final_candidate.json`, `h_sweep.txt` (when enabled) |
| **Files** | `stage3.py` (reuses `evaluate_proxy_policy` from Stage II) |
| **Cost** | Dominant GPU cost; scaled run K3=**400_000** (paper constant in code: **10_000_000**) |
| **Why** | Near-final policy training / Table-6-style reporting |

For `run_scaled_h5_gst`, H-sweep files only report **H=5** because training used `human_num=5` (pipeline clips H ≤ train crowd size).

---

## 5. Reward / Objective Design

### 5.1 `LegacyReward` (default CrowdNav scalar) — **Implemented & Validated** (unit tests + smoke train)

Priority in `state.py` `LegacyReward.compute`:

1. timeout → `0`  
2. collision → `collision_penalty` (config default **−20**)  
3. reaching_goal → `success_reward` (default **+10**)  
4. `dmin < discomfort_dist` → `(dmin − discomfort_dist) * discomfort_penalty_factor * time_step`  
5. else potential shaping: `pot_factor * (−dist − previous_potential)`  
6. unicycle extras if kinematics unicycle  

Config defaults: `crowd_nav/configs/config.py` (`discomfort_dist=0.25`, `discomfort_penalty_factor=10`, `time_step=0.25`, `gamma=0.99` for RL discount—not inside `calc_reward`).

### 5.2 LLM / evolved rewards — **Implemented & Validated** (sandbox + runs)

Path: LLM text → `normalize_to_compute_reward` → AST policy → restricted exec → smoke states → `SandboxedReward` implementing `RewardFunction` with per-episode `memory` dict.

Contract: `compute_reward(state, memory) → finite float`.

Example final artifact reward (`mut_0060_v3` in `best_stage3.json`) uses dense terms (progress, speed, direction, discomfort, accel, heading, time) with clamp to ±1, plus terminal goal/collision/timeout—**code-derived, not a fixed formula in the repo core**.

### 5.3 Score1 rules — **Implemented**

Categories SUCCESS / OTHER / FAIL with tie-breaks on nav length or distance-to-goal (`rules.py`).

### 5.4 Selection scalar — **Implemented**

`SR - CR - 0.5 * TR` (`selection.py`). ITR/SD are logged but not primary elite fitness.

---

## 6. RL Algorithm

### 6.1 Policy

- Class: `rl.networks.model.Policy` with base from `config.robot.policy` (`selfAttn_merge_srnn`).  
- Recurrent SRNN + self / human–robot attention flags from `arguments.py` (`use_self_attn=True`, `use_hr_attn=True` defaults).

### 6.2 Stage II — A2C (**Implemented**)

- Optimizer path: `rl.a2c.A2C` when `Stage2Config.algo == "a2c"` (default).  
- Rollout length default in Stage2Config: `num_steps=5`.  
- Shared hyperparameter source when parsing argv: `arguments.py` (e.g. `lr=4e-5`, `gamma=0.99`, `gae_lambda=0.95`, `entropy_coef=0.0`, `value_loss_coef=0.5`—as defined there).

### 6.3 Stage III / `train.py` — PPO (**Implemented**)

From `arguments.py` defaults (used by Stage III argv builder / `train.py`):

| Hyperparameter | Default in `arguments.py` |
|----------------|---------------------------|
| `algo` | `ppo` |
| `num-env-steps` | `20e6` (standalone train; **overridden** by Stage3Config / run config) |
| `num-processes` | `16` (runs often override, e.g. 4) |
| `num-steps` / seq | `30` |
| `ppo-epoch` | `5` |
| `clip-param` | `0.2` |
| `lr` | `4e-5` |
| `gamma` | `0.99` |
| `gae-lambda` | `0.95` |
| `use-gae` | `True` |
| `entropy-coef` | `0.0` |
| `value-loss-coef` | `0.5` |
| `seed` | `425` |

PPO update (`rl/ppo/ppo.py`): advantage = returns − values, normalized; clipped policy loss; Adam.

**Note (`AUDIT.md`):** `train.py` always constructs PPO even if `--algo a2c` is passed—Algorithm 1 Stage II does **not** call `train.py`.

### 6.4 Evaluation protocol (code)

- Stage II: `eval_episodes`, `horizon_steps` (short).  
- Stage III: `eval_episodes`, horizon = env `time_limit/time_step` if unset.  
- P0 suite: E=150, eval seeds {425,426,427}, H=5 pinned (`P0_SUMMARY.md`).

---

## 7. Computational-Efficiency Mechanism

| Mechanism | Status | Where / how |
|-----------|--------|-------------|
| Offline / cheap trajectory Score1 | **Implemented & Validated** | `scoring.py` + `data/stage1_dataset` |
| Procedural proxy (short RL) | **Implemented & Validated** | Stage II A2C, reduced K2 |
| Learned surrogate (ExtraTrees/RF/etc.) | **Not Implemented** | No module in execution path |
| Active learning | **Not Implemented** | — |
| Sandbox candidate filtering | **Implemented & Validated** | `sandbox/`, rejection logs |
| Early stopping inside K2/K3 | **Not Implemented** | Fixed update counts in trainers |
| Fidelity scheduling | **Partially** | Discrete Stage I → II → III fidelities only |
| Ranking | **Implemented** | Score1; nav scalar |
| Reward prediction model | **Not Implemented** | — |
| Full RL only for selected candidates | **Partially** | All Stage I survivors enter II/III in current pipeline (no top-k truncation in `pipeline.py`) |
| Stub trainers for wiring | **Implemented** | `--fast` / dry-run stubs |
| Paper-scale multi-seed | **Implemented in code; Not Validated at full budget** | `paper_scale.py`; local `results/archive/paper_scale/cost_log.json` shows **stub-scale** wall times (~0.03–0.05 s) |

---

## 8. Surrogate / Proxy Model

**Learned surrogate: Not Implemented.**

**Proxy that exists:** Stage II short-horizon A2C train/eval producing `ProxyMetrics`. It is part of the live Algorithm 1 path (not a disconnected prototype).

There is **no** separate feature/label dataset, ExtraTrees/RF training, or surrogate ranking step in `RaisePipeline`.

---

## 9. Experimental Setup

Values below are taken from saved `config.json` / eval JSON only.

| Experiment | Environment | Reward | RL | Steps (train) | Episodes (eval) | Seed | Metrics available | Status |
|------------|-------------|--------|-----|---------------|-----------------|------|-------------------|--------|
| `run_scaled_h5_gst` | GST inferred, H=5, without_random | Evolved LLM rewards | A2C K2=8e3; PPO K3=4e5 | Stage configs in `config.json` | S2:50 / S3:150 | train 425 | Full stage JSONs + plots + P0 suite | **Complete Algorithm 1 + independent re-eval** |
| P0 suite (same run dir) | H=5, GST, without_random | Fixed RAISE ckpt + baselines | Eval only | — | 150 × seeds 425–427 | eval 425–427 | `p0_comparison.json` | **Validated comparison** |
| `run_5h_easy` | `predict_method=none`, H=5 | Evolved | A2C 8e3; PPO 1.5e5 | see config | S2:40 / S3:80 | 425 | stage artifacts | Archived → `results/archive/run_5h_easy/` |
| `run_1to2h` | GST inferred | Evolved | A2C 4e3; PPO 8e4 | see config | S2:20 / S3:50 | 425 | stage artifacts | Archived → `results/archive/run_1to2h/` |
| `run_1to1p5h_easy` | (incomplete) | — | — | — | — | — | only Stage1 + 1 Stage2 dir | Archived → `results/archive/run_1to1p5h_easy/` |
| `paper_scale` / `_paper_seed_dry` | stub | stub | stub | — | — | 425 | `cost_log.json` tiny wall times | Archived under `results/archive/` |
| `_seed_fast_ok` / `_prompt_fix_smoke` | wiring | seed/smoke | stubs | — | — | — | Smoke | Archived under `results/archive/` |
| AUDIT ORCA check | ORCA_no_rand | n/a (classical) | n/a | — | 500 | — | SR 0.78, NT 15.87, … | **Documented local validation** of baseline env |
| `train.py` smoke_legacy | CrowdSim | LegacyReward | PPO 3000 steps | 3000 | — | 425 | checkpoints noted in AUDIT | Smoke train |

**Not available in repository:** completed multi-seed **training** of Algorithm 1 at paper K3=1e7; DS-RNN baseline local reproduction (explicitly skipped in P0 JSON limitations).

---

## 10. Baselines

| Baseline | Reward / policy | Algorithm | Budget | Eval setup | Result (evidence) | Executed? |
|----------|-----------------|-----------|--------|------------|-------------------|-----------|
| CrowdNav++ GST | Learned CrowdNav++ reward/policy (pretrained) | Pretrained PPO policy `41200.pt` | Upstream training (not re-run here) | P0: H=5, E=150, seeds 425–427 | SR **0.982 ± 0.008**, scalar **0.973 ± 0.012** | **Yes (eval)** |
| ORCA | Classical ORCA | `ORCA_no_rand` / `00000.pt` | n/a | P0 same | SR **0.978 ± 0.008**, scalar **0.956 ± 0.015** | **Yes (eval)** |
| Social Force | Classical SF | `SF_no_rand` / `00000.pt` | n/a | P0 same | SR **0.773 ± 0.035**, scalar **0.547 ± 0.071** | **Yes (eval)** |
| ORCA paper-table check | ORCA | `test.py` 500 eps | n/a | AUDIT.md | SR 0.78, NT 15.87, PL 18.53, ITR 26.04%, SD 0.36 | **Yes (documented)** |
| LegacyReward PPO (matched K3) | LegacyReward | PPO | Same as RAISE K3 | — | **Not available** | **No** in P0 suite |
| DS-RNN | — | — | — | — | Skipped locally | **Not reproduced** |

**Caveat (from `p0_comparison.json`):** baselines were **pretrained at paper H=20** but evaluated here with **H pinned to 5**—comparison is useful for engineering triage, **not** a paper-table claim.

---

## 11. Results

### 11.1 P0 comparison (most citable numbers)

Source: `raise_env/results/run_scaled_h5_gst/evals/p0_comparison.json` and `P0_SUMMARY.md`.

| Method | SR mean±std | CR mean | TR mean | Scalar mean±std |
|--------|-------------|---------|---------|-----------------|
| CrowdNav++ GST | 0.982 ± 0.008 | 0.000 | 0.018 | 0.973 ± 0.012 |
| ORCA | 0.978 ± 0.008 | 0.022 | 0.000 | 0.956 ± 0.015 |
| SF | 0.773 ± 0.035 | 0.227 | 0.000 | 0.547 ± 0.071 |
| RAISE best-ever R0 `mut_0060_v2` | **0.653 ± 0.046** | 0.302 | 0.044 | **0.329 ± 0.100** |
| RAISE last-round R1 same genome | 0.507 ± 0.035 | 0.396 | 0.098 | 0.062 ± 0.078 |

Training-time logged metrics for R0 (seed 425): SR=0.680 — **reproduced** on re-eval seed 425 (P0_SUMMARY).

### 11.2 Algorithm 1 artifact snapshot (`run_scaled_h5_gst`)

| Stage | Best id (artifact) | Notable metric |
|-------|--------------------|----------------|
| I | `cro_0009` | Score1 ≈ 0.831 |
| II | `ran_0064_v2` | proxy SR=0.08, TR=0.84 (short-horizon; **not** comparable to Stage III SR) |
| III file `final_candidate` | `mut_0060_v3` | last_metrics SR≈0.467 (R1) — **worse than R0 independent eval** |

H-sweep on R1 `mut_0060` lineage: only H=5, SR=0.4667 (`h_sweep.txt`).

### 11.3 Other runs (brief)

| Run | Stage3 best logged SR (from best_stage3 metadata) | Notes |
|-----|---------------------------------------------------|-------|
| `run_5h_easy` | 0.275 | No GST; lower K3; now under `results/archive/` |
| `run_1to2h` | 0.0 | Severely reduced budget; now under `results/archive/` |

### 11.4 What is **not** evidenced

- Surrogate accuracy / ranking quality vs full RL  
- Multi-seed **training** mean±std for RAISE Algorithm 1  
- Paper K3=1e7 completion  
- Matched-budget LegacyReward PPO baseline  

---

## 12. Most Valid / Best Current Run

**Most Valid Current Result:**  
**`results/run_scaled_h5_gst` training run + `evals/P0_SUMMARY.md` / `p0_comparison.json`**

| Field | Value |
|-------|--------|
| **Experiment** | Scaled Algorithm 1 (`run_scaled_h5_gst`) + P0 multi-eval-seed suite |
| **Commit/version** | Report generated against git `fa3325d` (artifacts dated 2026-09-07/08; exact train commit not embedded in JSON—treat as local results under current tree) |
| **Environment** | CrowdSim + GST inferred; `without_random`; **H=5**; CUDA |
| **Reward** | Evolved `compute_reward` (best-performing eval: genome `mut_0060_v2` at Stage III round 0) |
| **Algorithm** | Stage II A2C (K2=8000); Stage III PPO (K3=400000); policy `selfAttn_merge_srnn` |
| **Training budget** | Reduced vs paper (not 1e7; G1/G2/G3 reduced; human_num=5) |
| **Evaluation** | Independent E=150 × seeds {425,426,427}; baselines included |
| **Main metrics** | RAISE R0: SR **0.653±0.046**, scalar **0.329±0.100**; still below GST/ORCA/SF under this protocol |
| **Result** | End-to-end Algorithm 1 **works**; current RAISE reward/policy **does not** outperform standard baselines at this budget/H |
| **Why most reliable** | (1) Complete stage artifacts; (2) real LLM+GPU train; (3) GST path on; (4) independent re-eval reproduces train metric; (5) explicit limitation list in JSON; (6) baselines run under same eval harness. |

**Important honesty note:** `final_candidate.json` / `best_stage3_id=mut_0060_v3` is **not** the strongest independently measured policy; P0 shows **R0 best-ever** beats **R1**. Prefer citing **P0 `RAISE_best_ever_r00`**, not the raw `final_candidate` alone.

If the supervisor asks for a paper-faithful claim: **no fully validated paper-scale Algorithm 1 result is present in this repository.**

---

## 13. Comparison With Previous Versions

From git log (titles only) and result folders:

| Change | Evidence | Effect on claims |
|--------|----------|------------------|
| Best-ever tracking / elitism | commit `fa3325d`; P0_SUMMARY “code fixes” | Addresses refine regression; **full Algorithm 1 not re-run** after fix in saved P0 note |
| Groq key manager / Ollama / CUDA defaults | recent commits | Engineering reliability |
| Scaled GST run vs easy/no-GST / tiny budgets | `run_scaled_h5_gst` vs `run_5h_easy` / `run_1to2h` | Scaled GST run is the only one with serious P0 baseline comparison |
| Paper-scale entry | stubs in `results/archive/paper_scale` | Wiring only; not a science result |

Older reduced runs show much weaker Stage III SR (including 0.0)—do **not** cite them as primary evidence.

---

## 14. Reproducibility

Commands below are taken from `README.md` / `README_RAISE.md` / scripts.

1. **Install** (`raise_env/`): Python 3.10 venv; `pip install -r requirements_pinned.txt`; install PyTorch CUDA wheel as pinned; `pip install -e ../baselines_openai --no-build-isolation`; install Python-RVO2.  
2. **API keys:** `groq_keys.json` from example, or `GROQ_API_KEY` / Ollama.  
3. **Stage I data:** `data/stage1_dataset/` (`stage1_dataset.npz`, manifest `n_scenarios: 100`). Collect via `python scripts/collect_stage1_dataset.py` if regenerating.  
4. **GST weights:** required for `predict_method=inferred` (paths under `gst_updated/results/...` per AUDIT/regime).  
5. **Fast wiring:** `python scripts/run_raise.py --fast --output-dir results/raise_fast`  
6. **Tests:** `pytest crowd_nav/reward_search/tests -m "not slow"`  
7. **Full-ish local run:** `python scripts/run_raise.py --llm groq --device cuda --regime without_random --stage1-dataset data/stage1_dataset ...` (see README_RAISE matrix).  
8. **P0-style eval:** `scripts/run_p0_eval_suite.py`, `eval_raise_checkpoint.py`, `run_p0_gst_baseline.py`  
9. **Expected outputs:** `results/<run>/{config,manifest,best_stage*,stage*_population,stage2_train,stage3_train,plots}/`

**Reproducibility gaps:** single training seed in main run; `.venv` noted removed in AUDIT (must recreate); results may be machine-local; paper multi-seed training not completed.

---

## 15. Current Limitations

Evidence-based only:

1. **Budgets below paper** for the best documented run (K3=4e5 ≪ 1e7; reduced G; H=5).  
2. **RAISE underperforms** CrowdNav++/ORCA/SF under the P0 H=5 protocol.  
3. **Single training seed** (425); multi-seed is eval-only.  
4. **Selection/refine regression:** last-round checkpoint worse than earlier best-ever (measured).  
5. **No matched LegacyReward PPO** baseline at same K3.  
6. **Baseline H mismatch** (pretrained @20, eval @5)—stated in P0 JSON.  
7. **No learned surrogate / active learning / AMFRS.**  
8. **Stage II short-horizon metrics** not comparable to Stage III SR (e.g. Stage II “best” SR=0.08).  
9. **`final_candidate.json` can mislead** relative to best evaluated policy.  
10. **Paper-scale full run not present** (only stub cost log).  
11. **Incomplete / debug runs** were archived under `results/archive/` (e.g. `run_1to1p5h_easy`).  
12. **DS-RNN** not reproduced locally (P0).  

---

## 16. Recommended Next Steps

| Priority | Problem | Why it matters | Proposed solution | Expected benefit | Required implementation | Difficulty |
|----------|---------|----------------|-------------------|------------------|-------------------------|------------|
| **P1 Critical** | Finalist selection ≠ best eval | Wrong “winner” for claims | Persist & select `best_trained` by nav scalar; re-run P0 after | Credible reporting | Already partly coded post-P0; **re-run Algorithm 1** to validate | Medium |
| **P1 Critical** | No matched LegacyReward PPO | Cannot isolate “reward search helps” | Train LegacyReward PPO at same H/K3/seed protocol; eval with P0 suite | Fair ablation | `train.py` or Stage3 with `LegacyReward` | Medium–High (GPU) |
| **P1 Critical** | H=5 / reduced budget vs paper | Cannot support paper tables | Either label all claims as H=5 scaled study **or** run H=20 + larger K3 | Scientific validity | Config + compute | High |
| **P2 Important** | Stage II noise / unverified refine | Wastes Stage III | Multi-seed proxy; accept/reject refine (see master plan doc) | Better candidates | `stage2.py` | Medium |
| **P3 Optional** | Cost of full population Stage III | GPU wall | Top-k finalists only | More seeds possible | `pipeline.py` | Low–Medium |

---

## 17. Technical Assessment

### Current Maturity

**Experimental Framework** (bordering **Research Prototype**).

**Why:** End-to-end Algorithm 1 is implemented, tested at unit level, and executed with real LLM+GPU+GST; however, paper-scale validation is missing, primary RAISE policies do not beat standard baselines under the best documented protocol, and several scientific controls (matched Legacy PPO, multi-seed training) are absent.

### What is already demonstrated?

- Pluggable reward API + sandbox validation.  
- Stage I Score1 evolution on a 100-scenario dataset.  
- Stage II/III train/eval loops with artifact logging.  
- Independent multi-eval-seed comparison harness.  
- Env/baseline sanity (ORCA table match in AUDIT; P0 baseline evals).  

### What is not yet demonstrated?

- RAISE reward search **outperforming** CrowdNav++ under a fair, paper-aligned protocol.  
- Completed paper K3 / multi-seed Algorithm 1.  
- AMFRS / surrogate / active learning.  
- Statistical significance across training seeds.  

### Evidence still missing for a strong academic claim

1. Multi-seed **training** results (mean±std).  
2. H=20 (or declared scaled protocol) with matched baselines.  
3. Matched-budget LegacyReward (and ideally fixed CrowdNav++ retrains).  
4. Post-selection-fix full re-run artifacts.  
5. Clear separation of proxy metrics vs final metrics in all reports.  

---

## 18. Suggested Presentation to Supervisor

**Research Problem**  
Search a scalar reward for RL-based robot navigation among ORCA humans in CrowdNav++, aiming for high success and low collision.

**Proposed Method (as implemented)**  
RAISE Algorithm 1: LLM proposes rewards → sandbox → Score1 offline ranking → short A2C proxy + LLM refine → longer PPO + LLM refine. AMFRS not in this codebase.

**Current Implementation**  
Full pipeline in `raise_env/crowd_nav/reward_search/` on CrowdNav++ envs; Groq LLM; GST optional; results under `results/`.

**Best Validated Result**  
`run_scaled_h5_gst` + P0 eval: best RAISE policy **SR ≈ 0.65 ± 0.05** (H=5, E=150, 3 eval seeds), while CrowdNav++ GST / ORCA remain ≈ **0.98** under the same eval protocol. Budgets are below paper scale.

**Main Contribution (honest)**  
A reproducible engineering stack for Algorithm 1 reward search and evaluation on CrowdNav++, with measured gaps vs strong baselines—not yet a superior navigation method under paper conditions.

**Current Limitation**  
Reduced compute settings, single training seed, selection/refine pathologies, missing matched Legacy PPO, no AMFRS.

**Next Experiment**  
(1) Re-run Algorithm 1 with best-ever selection locked; (2) matched LegacyReward PPO at same H/K3; (3) P0-style eval—report whether reward search helps at all before scaling to paper K3.

---

## FACT-CHECK SUMMARY

| # | Claim | Evidence path |
|---|--------|----------------|
| 1 | AMFRS not in this release | `README.md`; `pipeline.py` manifest notes |
| 2 | Entry: `scripts/run_raise.py` → `RaisePipeline` | `scripts/run_raise.py`, `pipeline.py` |
| 3 | Stage I = Score1 Spearman on dataset | `scoring.py`, `rules.py` |
| 4 | Stage II default A2C; Stage III PPO | `stage2.py`, `stage3.py`, `AUDIT.md` §7 |
| 5 | Sandbox validates LLM rewards | `sandbox/validator.py`, rejection jsonl in results |
| 6 | No learned surrogate / active learning | Code search / pipeline imports |
| 7 | Best documented run dir | `results/run_scaled_h5_gst/` |
| 8 | That run config: H=5, GST, K2=8000, K3=400000, Groq | `results/run_scaled_h5_gst/config.json` |
| 9 | Stage I dataset 100 scenarios | `data/stage1_dataset/manifest.json` |
| 10 | P0 RAISE R0 SR≈0.653 vs GST≈0.982 | `results/run_scaled_h5_gst/evals/p0_comparison.json` |
| 11 | R0 beats R1 (selection regression) | `P0_SUMMARY.md` |
| 12 | `final_candidate` is `mut_0060_v3` with weaker R1 metrics | `best_stage3.json` / `final_candidate.json` |
| 13 | Paper-scale folder is stub timings | `results/archive/paper_scale/cost_log.json` |
| 14 | ORCA AUDIT match Table II | `AUDIT.md` §6 |
| 15 | Git HEAD at report | `fa3325d9cb618498a204903f2040a7c51aa0f55e` |

---

*End of report. This document describes what is present and measured in the repository as of the audit date; it does not claim paper-table superiority for RAISE.*

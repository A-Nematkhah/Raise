# AMFRS / RAISE Stage 1–2–3 Master Plan

**Status:** Reference document only — no implementation implied by this file.  
**Repo reality:** This tree ships **RAISE Algorithm 1** (faithful replication). **AMFRS is not implemented** (`pipeline.py` / `evolver.py` explicitly exclude novelty archive, Pareto, adaptive controller).  
**Sources:** Code under `raise_env/crowd_nav/reward_search/` + prior Stage I/II/III analysis in project discussions.  
**Last consolidated:** 2026-09-10

Labels used throughout:

- **Not Implemented** — proposed or mentioned, absent from current code  
- **Needs Validation** — idea exists; necessity or effect not yet measured  
- **New Observation** — important fact in code not emphasized in earlier chat, or only lightly touched  

---

## 1. Executive Summary

The project today is a CrowdNav++ fork plus an RAISE Algorithm 1 pipeline: LLM-proposed reward functions are sandboxed, ranked analytically on a fixed trajectory dataset (**Stage I**), refined under short proxy RL (**Stage II**, typically A2C), then trained/evaluated at near-full budget with PPO (**Stage III**), optionally with a human-count sweep. Entry is primarily `scripts/run_raise.py` → `RaisePipeline.run()`, with paper budgets via `configs/paper_scale.yaml` / `run_raise_paper_scale.py`.

The main architectural tension is a **three-stage objective mismatch**: Stage I optimizes Spearman alignment with coarse analytical rules on offline ORCA-like trajectories; Stage II optimizes a noisy short-horizon navigation scalar under A2C; Stage III optimizes long PPO training and reports H-generalization. Selection uses a simple scalar `SR - CR - 0.5·TR` (`selection.py`). There is **no surrogate model, no active learning loop, and no AMFRS multi-objective evolution** in code.

Intended roles:

| Stage | Intended role (paper / design) | Current role in code |
|-------|--------------------------------|----------------------|
| I | Cheap analytical filter of reward programs | Score1 + LLM evolve/mutate/crossover |
| II | Proxy train/eval + D.3 refine | Fresh A2C/K2 + LLM refine (often unverified same round) |
| III | Full PPO + final metrics / Table-6 H-sweep | Same refine pattern at high cost + H-sweep on best-ever |

Highest-leverage changes toward a credible path to **AMFRS-capable architecture** (without boiling the ocean): (1) make Stage I scoring trustworthy (pad/holdout/anti-exploit/richer rules); (2) stabilize Stage II ranking and **accept/reject** refinements; (3) turn Stage III into a **finalist tournament** with generalization/safety in the objective; (4) only then add AMFRS primitives (archive, Pareto, diversity) on stable interfaces (`RewardCandidate`, metrics, sandbox).

Path: **stabilize RAISE baseline metrics → close Stage I↔II↔III feedback loops → modularize shared train/refine → introduce AMFRS mechanisms as additive modules.**

---

## 2. Current Architecture

### 2.1 Entry points

| Entry | Role |
|-------|------|
| `scripts/run_raise.py` | Algorithm 1 end-to-end |
| `scripts/run_raise_paper_scale.py` | Multi-seed paper budgets; loads `configs/paper_scale.yaml` |
| `scripts/collect_stage1_dataset.py` | Build Stage I dataset |
| `scripts/run_stage2_smoke.py` / `run_stage3_smoke.py` | Stage wiring smokes |
| `train.py` / `test.py` | Standalone CrowdNav++ PPO (not called by Algorithm 1 stages) |
| `scripts/eval_raise_checkpoint.py`, `visualize_raise.py`, `report.py`, `plot_raise_run.py` | Post-hoc analysis |

### 2.2 Pipeline / orchestrator

- `crowd_nav/reward_search/pipeline.py` — `RaisePipeline` / `RaiseRunConfig`  
- Order: seed reward → Stage I (`StageIEvolver`) → Stage II (`Stage2Runner`) → Stage III (`Stage3Runner`) → JSON artifacts  
- Shared `RewardValidator`; optional `checkpoint_store` for paper-scale resume  

### 2.3 Environment

- `crowd_sim/envs/*` — Gym envs; termination env-owned  
- `build_reward_state` + pluggable `reward_fn` (`state.py`)  
- Stage II/III default: GST-inferred obs (`CrowdSimPredRealGST-v0`) via `VecPretextNormalize` when `predict_method=inferred`  

### 2.4 Reward system

- Contract: `RewardFunction.reset/compute(RewardState)`  
- Default: `LegacyReward`  
- LLM path: sandbox → `SandboxedReward`  
- Candidates: `RewardCandidate` in `evolver.py`  

### 2.5 Trajectory generation / evaluation

- Stage I: pre-collected `Stage1Dataset` (`dataset.py`); Score1 (`scoring.py` + `rules.py`)  
- Stage II/III: on-policy rollouts via `make_vec_envs` + `evaluate_proxy_policy` (defined in `stage2.py`, reused by Stage III)  

### 2.6 Surrogate

- **Not Implemented.** No learned proxy of Stage III metrics from cheap features.  
- Closest existing piece: Stage II itself as a *procedural* proxy (short RL), not a fitted surrogate.

### 2.7 Active learning

- **Not Implemented.** No query of new trajectories / scenarios based on model disagreement.

### 2.8 Safety / constraints

- Sandbox AST/runtime restrictions (research-grade, not OS isolation)  
- Env-owned collision/timeout/goal  
- Metrics ITR/SD collected but **not primary elite scalar**  
- Constraint-aware reward optimization (**Not Implemented**)

### 2.9 RL evaluation

- Stage II: short horizon `T_short`, E2 episodes, A2C (default)  
- Stage III: full episode horizon, E3 episodes, PPO; H-sweep `{5,10,15,20}` (pipeline may clip to `H ≤ human_num`)  
- Scalar: `SR - CR - 0.5·TR` (**New Observation:** centralized in `selection.py`)

### 2.10 Experiment configuration

- `RaiseRunConfig`, CLI flags, `regime.py`, `presets.py` / `paper_scale.yaml`  
- Env hyperparameters: `crowd_nav/configs/config.py` + runtime overrides in stage trainers  

### 2.11 Data flow (current)

```text
LLM code
  → sandbox validate
  → Stage I Score1(dataset of RewardState trajectories)
  → RewardCandidate population
  → Stage II: inject reward_fn → A2C K2 → ProxyMetrics → D.3 refine
  → Stage III: inject reward_fn → PPO K3 → ProxyMetrics → D.3 refine
  → best_trained / final_candidate JSON + optional H-sweep
```

```text
Current:
User/CLI → run_raise / paper_scale
        → RaisePipeline
             → StageIEvolver (Score1)
             → Stage2Runner (A2C proxy + refine)
             → Stage3Runner (PPO + refine + H-sweep)
             → results/*/manifest.json

Target (high level):
User/CLI → Pipeline
        → Stage I (trustworthy Score1 ± optional cheap on-policy signal)
        → Stage II (stable proxy + accept/reject refine + structured feedback)
        → Stage III (finalist tournament + generalization/safety objective)
        ↘ AMFRS modules (Pareto/archive/diversity)   [Not Implemented]
        ↘ Optional surrogate / active data          [Not Implemented]
```

---

## 3. Stage 1

### 3.1 Current State

- **Code:** `evolver.py`, `scoring.py`, `rules.py`, `dataset.py`; wired by `pipeline._score_fn` / `StageIEvolver`  
- **Loop:** Gen0 LLM batch → sandbox fill → Score1 → G1 generations of elite keep + crossover + mutate lower half + random restarts; template `reflection`; track `global_best`  
- **Score1:** mean over scenarios of mean over frames of Spearman(rule preference ranks, cumulative recomputed reward ranks)  
- **Rules:** SUCCESS ≫ OTHER ≫ FAIL; tie-break nav length (success) or dist-to-goal (other/fail)  
- **Smoke Score1:** `make_smoke_score_fn` for `--fast` only (different metric)  
- **AMFRS:** explicitly no novelty/archive  

### 3.2 Problems

**Problem S1-1 — Score1 ↔ real navigation gap**

- **Problem:** High Score1 ≠ high Stage II/III SR/CR.  
- **Location:** `scoring.py` (`score1_for_dataset`); consumers `evolver.score_population`, `pipeline`.  
- **Why:** Optimizes rank correlation with analytical rules on fixed offline trajs, not policy-induced returns.  
- **Impact:** Wrong candidates promoted to expensive RL stages.  
- **Severity:** Critical  
- **Proposed solution:** Keep Score1 as cheap filter; add holdout Score1; optionally blend mini-rollout / Stage-II calibration (`α·Score1 + β·proxy_nav`).  
- **Why this solution:** Preserves paper idea while aligning search with downstream success.  
- **Implementation approach:** Split dataset train/holdout in `scoring`/`dataset`; optional hook in `evolver` every N gens for top-k short eval (reuse Stage II eval helper or thinner stub).  
- **Dependencies:** Dataset quality; Stage II eval API stability.  
- **Validation:** Correlate Stage I rank vs Stage II scalar on a fixed candidate set; holdout vs train Score1 gap. **Needs Validation** for α/β.

**Problem S1-2 — Coarse analytical rules**

- **Problem:** Only three terminal buckets + simple tie-breaks; no discomfort/social/mid-episode structure.  
- **Location:** `rules.py` (`TrajectoryCategory`, `rule_preference_score`).  
- **Why:** Evolution overfits a blunt preference model.  
- **Impact:** Rewards that game categories look good; social quality ignored.  
- **Severity:** High  
- **Proposed solution:** Enrich rule scores with intermediate signals (dmin/discomfort, potential progress, optional intrusion proxies from stored states).  
- **Why:** Denser supervision without full RL.  
- **Implementation approach:** Extend `rule_preference_score` or add frame-local terms; update tests in `test_*score*`.  
- **Dependencies:** Dataset states must contain fields used (already in `RewardState`).  
- **Validation:** LegacyReward and known good/bad hand rewards ordering. **Needs Validation** for exact rule weights.

**Problem S1-3 — Trajectory padding inflates cumulative reward**

- **Problem:** `TrajectoryRecord.state_at` holds last state; cumulative `compute` may keep adding reward on pads.  
- **Location:** `dataset.py` `state_at`; `scoring.py` `_cumulative_reward` / `_scenario_frame_correlations`.  
- **Why:** Artificial late-frame returns distort Spearman.  
- **Impact:** Systematic bias favoring certain reward shapes / lengths.  
- **Severity:** High  
- **Proposed solution:** Mask padded frames from correlation and/or force zero reward contribution on pads.  
- **Why:** Removes scoring artifact.  
- **Implementation approach:** Track true length; in frame loop skip `f >= traj.length` for that traj or freeze cumulative.  
- **Dependencies:** None  
- **Validation:** Unit tests with unequal-length trajs; Score1 invariance to pad length.

**Problem S1-4 — Success rule score nearly constant across frames**

- **Problem:** For SUCCESS, rule score uses episode `nav_length`, constant in `f`.  
- **Location:** `rules.py` + usage in `_scenario_frame_correlations`.  
- **Why:** Many frame-wise correlations repeat a static ranking for success-heavy scenarios.  
- **Impact:** Weak temporal discrimination among successful behaviors.  
- **Severity:** Medium  
- **Proposed solution:** Frame-aware success preference (e.g., dist-to-goal progress) while keeping terminal category dominance.  
- **Why:** Uses time structure already in states.  
- **Implementation approach:** Change `rule_preference_score` call site to pass frame features for SUCCESS.  
- **Dependencies:** S1-2 design choices.  
- **Validation:** **Needs Validation** — ensure still matches Figure-3 spirit / paper claims if fidelity required.

**Problem S1-5 — Constant / explosive rewards**

- **Problem:** Near-constant cumulatives → Spearman NaN (frames dropped); explosive scales → brittle ranks. No hard anti-exploit in Score1.  
- **Location:** `scoring.py`; soft text only in `evolver._build_reflection`.  
- **Why:** LLM can emit degenerate shaping.  
- **Impact:** Empty/noisy scores; unstable evolution.  
- **Severity:** High  
- **Proposed solution:** Reject or penalize non-finite, low-variance, or extreme-magnitude reward streams; optional “must not lose to LegacyReward by large margin on holdout.”  
- **Why:** Filters useless programs before RL.  
- **Implementation approach:** Guards inside `_cumulative_reward` / `score1_for_dataset`; evolver treats penalty as low score.  
- **Dependencies:** Define thresholds (**Needs Validation**).  
- **Validation:** Synthetic constant/NaN/Inf rewards in tests.

**Problem S1-6 — Offline distribution mismatch**

- **Problem:** Dataset from collector policies ≠ Stage II/III GST neural policy state distribution.  
- **Location:** `collect_stage1_dataset.py` + `dataset.py`; consumers Stage I only.  
- **Why:** Classic offline eval mismatch.  
- **Impact:** Stage I winners fail under learned policies.  
- **Severity:** High  
- **Proposed solution:** Stratified/hard scenarios; optional periodic on-policy traj refresh (**active data** — Not Implemented); keep Score1 + proxy blend (S1-1).  
- **Why:** Reduces covariate shift.  
- **Implementation approach:** Collector flags aligned with `regime.py`; document GST N/A for Stage I collection (AUDIT).  
- **Dependencies:** Regime consistency; storage budget.  
- **Validation:** Compare Score1 ranks on ORCA trajs vs short neural rollouts. **Needs Validation**.

**Problem S1-7 — Weak population diversity**

- **Problem:** Invalid LLM outputs → parent clone fallback; no novelty filter (by design).  
- **Location:** `evolver.py` `_fill_valid`, `_clone_valid`, `_next_generation`.  
- **Why:** Collapse to similar genomes.  
- **Impact:** Stage II wastes budget on near-duplicates.  
- **Severity:** Medium  
- **Proposed solution:** Light diversity: code hash / behavior fingerprint on fixed smoke states; reject near-duplicates; optional AMFRS archive later.  
- **Why:** Cheap diversity without full AMFRS.  
- **Implementation approach:** Fingerprint via reward vector on `default_smoke_states`; distance threshold.  
- **Dependencies:** Sandbox smoke states.  
- **Validation:** Duplicate rate across gens.

**Problem S1-8 — Reflection is template-only**

- **Problem:** `_build_reflection` is fixed prose, not failure analysis.  
- **Location:** `evolver.py` `_build_reflection`.  
- **Why:** Mutations lack actionable weaknesses.  
- **Impact:** Weaker D.2-style improvement.  
- **Severity:** Medium  
- **Proposed solution:** Per-scenario Spearman breakdown + textual failure hints into mutation prompts.  
- **Why:** Grounds LLM edits in Score1 structure.  
- **Implementation approach:** Extend `score1_for_dataset` to return diagnostics; pass into `format_d2_mutation`.  
- **Dependencies:** `prompts.py`.  
- **Validation:** Qualitative + Score1 lift vs template baseline. **Needs Validation**.

**Problem S1-9 — Smoke vs paper Score1 confusion**

- **Problem:** `--fast` / smoke metric is not Score1.  
- **Location:** `scoring.make_smoke_score_fn`; `pipeline` / CLI.  
- **Why:** Easy to misread smoke runs as algorithmic evidence.  
- **Impact:** False conclusions in debugging/thesis drafts.  
- **Severity:** Medium (process) / Low (code if gated)  
- **Proposed solution:** Hard banner in artifacts `score1_mode`; refuse paper claims when smoke.  
- **Why:** Already partially documented; strengthen reporting.  
- **Implementation approach:** `manifest.json` + console warnings (mostly present).  
- **Dependencies:** Reporting  
- **Validation:** Test manifest field.

**Problem S1-10 — Gen0 batch parse fragility / LLM cost**

- **Problem:** One large completion + multi-function parse; failures cascade regenerations.  
- **Location:** `evolver.initialize_population`, `llm.split_reward_function_sources`.  
- **Why:** Provider/format brittle.  
- **Impact:** Cost and empty slots.  
- **Severity:** Medium  
- **Proposed solution:** Prefer per-candidate generation when batch parse fails early; tighter extract tests; rejection categories already logged.  
- **Why:** Reliability.  
- **Implementation approach:** Adjust Gen0 policy; keep rejection jsonl.  
- **Dependencies:** `llm.py`  
- **Validation:** Scripted LLM tests (`test_evolver`, `test_llm_extract`).

**Problem S1-11 — Global best vs last generation regression** *(New Observation / discussed with pipeline)*

- **Problem:** Last gen best can be worse than `global_best`; pipeline patches via `_include_global_best`.  
- **Location:** `evolver.run`, `pipeline._include_global_best`.  
- **Why:** Evolutionary noise.  
- **Impact:** Without pipeline guard, Stage II misses best Score1 genome.  
- **Severity:** Medium (mitigated in pipeline)  
- **Proposed solution:** Keep elitism; optionally re-insert elite every generation inside evolver too.  
- **Why:** Defense in depth.  
- **Implementation approach:** Mirror Stage II elite inject in evolver.  
- **Dependencies:** None  
- **Validation:** Unit test regression gen.

### 3.3 Recommended Architecture After Stage 1

```text
Dataset (train + holdout scenarios)
    → Score1 engine
         • masked pads
         • richer rules (optional weights)
         • anti-exploit gates
         • diagnostics per scenario
    → StageIEvolver
         • diversity fingerprint
         • mutation prompts with diagnostics
         • optional top-k cheap rollout blend   [Needs Validation]
    → Population + global_best + score report
```

Stage I remains **offline-first** and sandbox-gated; it does not own full PPO.

### 3.4 Stage 1 Implementation Plan

1. **Pad masking + finite/variance guards** in `scoring.py` / `dataset` usage (prerequisite for trustworthy numbers).  
2. **Holdout split** + report train vs holdout Score1.  
3. **LegacyReward / fixture regression tests** for Score1 monotonicity sanity.  
4. **Richer rules** (incremental; version rule_id in manifest).  
5. **Diagnostics → mutation prompts**.  
6. **Light diversity fingerprints**.  
7. **Optional** mini-rollout / α–β blend (**Needs Validation**; depends on Stage II eval harness).  
8. **Optional** active traj refresh (**Not Implemented**; after AMFRS/data layer).  

---

## 4. Stage 2

### 4.1 Current State

- **Code:** `stage2.py` (`Stage2Runner`, `RealPolicyTrainer`, `StubPolicyTrainer`, `evaluate_proxy_policy`, `ProxyMetrics`)  
- **Pattern:** For each of G2 rounds × population: fresh train K2 (default A2C, practical K2=5e4; paper 8000) → eval E2 @ T_short → record trained snapshot / `best_trained` → D.3 refine (+ one repair) unless elite genome protected → elitism inject  
- **Reward injection:** `make_vec_envs(..., reward_fn=)`  
- **Does not call** `train.py`  
- **Resume:** optional checkpoint store (paper-scale)  

### 4.2 Problems

**Problem S2-1 — Noisy short proxy rankings**

- **Problem:** Single-seed short K2 + short horizon → high variance ranks.  
- **Location:** `RealPolicyTrainer.train_and_eval`, `evaluate_proxy_policy`, `Stage2Config`.  
- **Why:** Incomplete learning + short episodes.  
- **Impact:** Mis-order candidates; bad refine targets.  
- **Severity:** Critical  
- **Proposed solution:** Multi-seed proxy eval (2–3); optional median aggregate; consider slightly longer horizon curriculum.  
- **Why:** Stabilizes tournament without full K3.  
- **Implementation approach:** Loop seeds in trainer or eval; store mean±std in metadata.  
- **Dependencies:** Compute budget.  
- **Validation:** Rank stability across repeats. **Needs Validation** for seed count vs cost.

**Problem S2-2 — Fresh policy every round**

- **Problem:** Always cold-start A2C for each candidate/round.  
- **Location:** `RealPolicyTrainer.train_and_eval`.  
- **Why:** No transfer when reward unchanged or slightly edited.  
- **Impact:** Cost↑, noise↑.  
- **Severity:** High  
- **Proposed solution:** Warm-start / short fine-tune when genome unchanged or edit distance small; cold-start on large edits.  
- **Why:** Saves budget for comparisons that matter.  
- **Implementation approach:** Cache checkpoint keyed by code hash; load into `Policy` when allowed.  
- **Dependencies:** Checkpoint compatibility.  
- **Validation:** **Needs Validation** — warm-start may bias comparisons.

**Problem S2-3 — Refine without same-round verification**

- **Problem:** LLM changes code after metrics; improvement tested only next round.  
- **Location:** `Stage2Runner.refine_candidate`, `_train_refine_one`.  
- **Why:** Open-loop mutation.  
- **Impact:** Regressions burn a full round (elite partially protected).  
- **Severity:** Critical  
- **Proposed solution:** Accept/reject: mini-train or cheap re-eval; keep new code only if scalar improves (with tolerance).  
- **Why:** Turns refine into hill-climbing.  
- **Implementation approach:** After validate, run reduced K2′ or eval-only with frozen short policy (**Needs Validation** which verifier).  
- **Dependencies:** S2-1 stability helps.  
- **Validation:** Fraction of accepted refines that help next full round.

**Problem S2-4 — Unstructured LLM feedback**

- **Problem:** D.3 gets raw metric strings, not failure modes.  
- **Location:** `ProxyMetrics.feedback_text`; `refine_candidate`; `prompts.format_d3_*`.  
- **Why:** Model guesses generic edits.  
- **Impact:** Low-quality refinements.  
- **Severity:** High  
- **Proposed solution:** Behavior/failure-mode summaries (collision timing, timeout distance, ITR, crowding).  
- **Why:** Closed-loop reward debugging.  
- **Implementation approach:** Aggregate from episode infos in `evaluate_proxy_policy`; extend D.3 prompt schema + regression tests.  
- **Dependencies:** Info dict richness from env.  
- **Validation:** Prompt schema tests; qualitative refine quality. **Needs Validation** for taxonomy.

**Problem S2-5 — Algorithm / protocol mismatch with Stage III**

- **Problem:** Stage II A2C + T_short vs Stage III PPO + full horizon.  
- **Location:** `Stage2Config.algo`, horizon; vs `stage3.py`.  
- **Why:** Different optimal rewards possible.  
- **Impact:** Stage II winners fail Stage III.  
- **Severity:** High  
- **Proposed solution:** Options — (A) short PPO in II; (B) curriculum horizon toward time_limit; (C) keep A2C but calibrate ranking empirically.  
- **Why:** Reduce objective mismatch.  
- **Implementation approach:** Config flag `stage2_algo`; share trainer module.  
- **Dependencies:** Conflict with cost (see §10). **Decision Pending** (D-2).  
- **Validation:** Rank correlation II vs III before/after.

**Problem S2-6 — Oversimplified selection scalar**

- **Problem:** Elite uses `SR - CR - 0.5·TR`; ITR/SD/NT/PL underused.  
- **Location:** `selection.py`, `ProxyMetrics.scalar_score`, `best_trained` updates.  
- **Why:** Social/efficiency not in primary fitness.  
- **Impact:** Unsafe or slow policies can win.  
- **Severity:** High  
- **Proposed solution:** Explicit multi-metric scalar or Pareto selection for elite; document weights.  
- **Why:** Aligns with crowd-nav goals / AMFRS multi-objective direction.  
- **Implementation approach:** Extend `selection.py`; pipeline uses same helper.  
- **Dependencies:** Stage III should share scalar (D-3).  
- **Validation:** Sensitivity analysis on weights. **Needs Validation**.

**Problem S2-7 — Population × rounds compute wall**

- **Problem:** N×G2×(train+eval) dominates wall-clock.  
- **Location:** `Stage2Runner.run` / `run_round`.  
- **Why:** Full population every round.  
- **Impact:** Forces tiny K2 or few runs → worse science.  
- **Severity:** High  
- **Proposed solution:** Adaptive early-stop for clearly bad candidates; shrink N after first rounds; promote top-k only.  
- **Why:** Budget → informative comparisons.  
- **Implementation approach:** After 25–50% K2 peek SR; stop if hopeless.  
- **Dependencies:** **Needs Validation** thresholds.  
- **Validation:** Same finalist quality at lower GPU-hours.

**Problem S2-8 — Stub trainer ≠ real behavior**

- **Problem:** Hash-based metrics for tests.  
- **Location:** `StubPolicyTrainer`.  
- **Why:** CI cannot catch ranking bugs.  
- **Impact:** Silent logic regressions.  
- **Severity:** Medium  
- **Proposed solution:** Keep stub for wiring; add optional “tiny real env” slow marker tests.  
- **Why:** Separates unit vs integration.  
- **Implementation approach:** pytest marks (already have slow patterns elsewhere).  
- **Dependencies:** CI time  
- **Validation:** Marked integration test.

**Problem S2-9 — Code duplication with Stage III** *(New Observation)*

- **Problem:** Near-copy of train/refine/elite/resume loops.  
- **Location:** `stage2.py` vs `stage3.py`.  
- **Why:** Divergent fixes.  
- **Impact:** Innov applied to one stage only → protocol skew.  
- **Severity:** Medium (maintainability)  
- **Proposed solution:** Shared `PolicyTrainHarness` + stage configs.  
- **Why:** Modularity for AMFRS later.  
- **Implementation approach:** Extract carefully with characterization tests.  
- **Dependencies:** Before large II/III innov.  
- **Validation:** Smoke parity.

### 4.3 Proposed Improvements

(All retained/merged from prior discussion.)

1. Multi-seed proxy metrics  
2. Accept/reject refine  
3. Warm-start / fine-tune when safe  
4. Structured failure-mode D.3 feedback  
5. Shared multi-metric / Pareto elite selection  
6. Short PPO and/or horizon curriculum toward Stage III  
7. Adaptive compute / early-stop  
8. Uncertainty-aware selection (UCB / worst-quartile) — **Not Implemented**, **Needs Validation**  
9. Relative ranking / shared init tournaments — **Not Implemented**, **Needs Validation**  
10. Learned surrogate of Stage III from cheap features — **Not Implemented**, **Needs Validation**, high cost  
11. Light diversity / behavior archive (AMFRS-lite) — **Not Implemented**  
12. Dual-loop (freeze policy, adapt reward) — **Not Implemented**, **Needs Validation**  
13. Constraint-aware Stage II (CR/ITR thresholds) — **Not Implemented**  

### 4.4 Recommended Architecture After Stage 2

```text
Population from Stage I
  → ProxyTrainEval harness (shared with III)
       • multi-seed metrics + uncertainty
       • optional warm-start
       • early-stop
  → Elite selection (multi-metric / Pareto)
  → Refine proposal (D.3 + failure modes)
       → sandbox
       → accept/reject verifier
  → Elitism + diversity filter
  → Population out + trained_snapshots
```

### 4.5 Implementation Order

1. Extract/share metrics + selection helpers (already partly `selection.py`).  
2. Multi-seed aggregation + metadata.  
3. Accept/reject refine gate.  
4. Failure-mode feedback → prompts.  
5. Early-stop / top-k promotion.  
6. Trainer dedup with Stage III.  
7. Algo/horizon alignment experiment (Decision D-2).  
8. Only then surrogates / dual-loop / full uncertainty machinery.  

### 4.6 Validation Criteria

- Rank correlation of Stage II ordering across replications (↑).  
- Correlation of Stage II elite vs Stage III final metrics (↑).  
- Accept rate of refines that improve next full train (↑).  
- GPU-hours per useful finalist (↓).  
- No sandbox regressions (`test_stage2`, prompt schema).  

---

## 5. Stage 3

### 5.1 Current State

- **Code:** `stage3.py` (`Stage3Runner`, PPO `RealPolicyTrainer`, `TrainEvalBundle`, `HumanSweepReport`)  
- **Reuses:** `ProxyMetrics`, `evaluate_proxy_policy` from Stage II  
- **Budgets:** `STAGE3_STEPS=5e5` default; `STAGE3_PAPER_STEPS=1e7`; G3≈3; E3=500; full episode horizon  
- **End:** H-sweep preferably on `best_trained`  
- **Refine:** D.3 + repair; `*_v3` ids; elite protection  

### 5.2 Problems

**Problem S3-1 — Extreme cost vs evolutionary signal**

- **Problem:** Most compute confirms rather than searches.  
- **Location:** `STAGE3_*`, `Stage3Runner.run`.  
- **Why:** Full pop × G3 × K3.  
- **Impact:** Few seeds; weak statistics; thesis risk.  
- **Severity:** Critical  
- **Proposed solution:** Admit only top-k finalists from Stage II; treat Stage III as tournament.  
- **Why:** Spend K3 where decisions matter.  
- **Implementation approach:** Pipeline truncates population; config `stage3_max_finalists`.  
- **Dependencies:** Reliable Stage II ranking (S2-1, S2-3).  
- **Validation:** Same or better final SR at lower total steps.

**Problem S3-2 — Refine-without-verify at highest cost**

- **Problem:** Same open-loop refine as II, but each mistake costs K3.  
- **Location:** `refine_candidate` / `_train_refine_one` analogs in `stage3.py`.  
- **Why:** Identical pattern.  
- **Impact:** Catastrophic budget waste.  
- **Severity:** Critical  
- **Proposed solution:** Stricter accept/reject than II (mandatory mini-verify or disallow refine on last round).  
- **Why:** Final stage should be conservative.  
- **Implementation approach:** Shared refine gate; `allow_refine_rounds={0,1}` only.  
- **Dependencies:** S2-3 design reuse.  
- **Validation:** Count harmful accepted refines → 0 target.

**Problem S3-3 — Too few refinement rounds (G3≈3)**

- **Problem:** Little room to edit rewards under true PPO behavior.  
- **Location:** `Stage3Config.rounds`.  
- **Why:** Table-6 style budget.  
- **Impact:** Relies on Stage II quality almost entirely.  
- **Severity:** Medium  
- **Proposed solution:** Do not blindly raise G3; instead improve II + finalist tournament + optional single co-adaptation loop on winner.  
- **Why:** Raising G3 alone is cost explosion (conflict with S3-1).  
- **Implementation approach:** See D-4.  
- **Dependencies:** S3-1  
- **Validation:** Ablate G3=1 vs 3 under top-k-only.

**Problem S3-4 — Rank mismatch from Stage II**

- **Problem:** II winners ≠ III winners.  
- **Location:** Cross-stage; no feedback module.  
- **Why:** Algo/horizon/eval mismatch (S2-5).  
- **Impact:** Wrong finalists consume K3.  
- **Severity:** High  
- **Proposed solution:** Cross-stage calibration; align protocols; log rank inversions.  
- **Why:** Closes loop.  
- **Implementation approach:** Reporting table II vs III; optional reweight II scalar (**Needs Validation**).  
- **Dependencies:** D-2  
- **Validation:** Spearman of ranks II↔III.

**Problem S3-5 — H-sweep is report-only / zero-shot**

- **Problem:** Train at `train_human_num`, eval other H; often only best-ever; pipeline may collapse set when `human_num` low.  
- **Location:** `evaluate_at_human_counts`, `run_generalization_sweep`, `pipeline` H clipping.  
- **Why:** Generalization not in selection objective.  
- **Impact:** Table-6 style numbers without optimizing for them.  
- **Severity:** High  
- **Proposed solution:** Generalization-aware objective, e.g. involve `min_H SR` or mean across H in final pick; optionally sweep multiple finalists.  
- **Why:** Makes H-sweep decision-relevant.  
- **Implementation approach:** Extend selection after sweep; store in `best_stage3` metadata.  
- **Dependencies:** Cost of multi-candidate sweep. **Needs Validation** for formula.  
- **Validation:** Improve worst-H SR without collapsing H_train SR.

**Problem S3-6 — Elite scalar ignores social metrics / H profile**

- **Problem:** Same `SR-CR-0.5TR` primacy.  
- **Location:** `selection.py` + Stage3 `best_trained`.  
- **Why:** Underweights ITR/SD and H robustness.  
- **Impact:** “Successful but invasive” winners.  
- **Severity:** High  
- **Proposed solution:** Multi-objective / constrained selection (CR/ITR caps); AMFRS Pareto later.  
- **Why:** Thesis differentiation + safety.  
- **Implementation approach:** Shared selector; Stage III final choose uses H-aware metrics.  
- **Dependencies:** D-3  
- **Validation:** Pareto front plots; constraint violation rates.

**Problem S3-7 — Expensive eval, thin LLM critique**

- **Problem:** E3=500 yields stable means but D.3 still sees flat summaries.  
- **Location:** Eval + refine prompts.  
- **Why:** Wasted signal.  
- **Impact:** Weak last-mile reward edits.  
- **Severity:** Medium  
- **Proposed solution:** Same failure-mode package as II, fed from full-episode PPO rollouts.  
- **Why:** Best behavioral data is here.  
- **Implementation approach:** Share diagnostics builder.  
- **Dependencies:** S2-4  
- **Validation:** Refine accept quality.

**Problem S3-8 — GST / hardware / regime fragility**

- **Problem:** Inferred GST + 20 humans + PPO sensitive to VRAM/`num_processes`/regime mismatch.  
- **Location:** `regime.py`, trainers, AUDIT §8.  
- **Why:** Heavy stack.  
- **Impact:** Failed runs misread as reward failure.  
- **Severity:** Medium  
- **Proposed solution:** Loud asserts (partly exist); OOM-safe process defaults; separate “reward ablation” profile with `predict_method=none` labeled non-paper.  
- **Why:** Scientific validity.  
- **Implementation approach:** Config profiles in presets.  
- **Dependencies:** Docs  
- **Validation:** Smoke on target GPU.

**Problem S3-9 — `best_trained` vs final population confusion**

- **Problem:** Final selection is best-ever trained snapshot semantics; last pop may differ post-refine.  
- **Location:** `Stage3Runner`, `pipeline` final_candidate.  
- **Why:** Pre-refine snapshot vs refined code lineage.  
- **Impact:** Misinterpretation of artifacts.  
- **Severity:** Medium  
- **Proposed solution:** Artifact schema clarifying `evaluated_genome` vs `proposed_refined_genome`; always serialize both.  
- **Why:** Reproducibility.  
- **Implementation approach:** `reporting.candidate_to_dict` fields.  
- **Dependencies:** None  
- **Validation:** Tests on manifest keys.

**Problem S3-10 — Duplication with Stage II**

- **Problem:** Same as S2-9.  
- **Severity:** Medium  
- **Proposed solution:** Shared harness.  
- **Dependencies:** Refactor before large innov.

### 5.3 Proposed Improvements

1. Top-k finalists only  
2. Mandatory accept/reject (stricter than II)  
3. Multi-seed reporting (paper-scale already multi-seed at pipeline level)  
4. H-aware / constrained final selection  
5. Failure-mode D.3 from PPO rollouts  
6. Shared trainer/refine module with II  
7. Finalist tournament with fair seeds — **Not Implemented**  
8. Controlled reward–policy co-adaptation — **Not Implemented**, **Needs Validation**  
9. Budgeted early-stop / promotion across finalists — **Not Implemented**  
10. Distill/warm-start from Stage II policy — **Not Implemented**, **Needs Validation**  
11. Post-hoc reward term attribution — **Not Implemented**  
12. Cross-stage feedback to proxy weights — **Not Implemented**, **Needs Validation**  
13. Safety/social overlay (Lagrangian / Pareto) — **Not Implemented** (AMFRS-aligned)

### 5.4 Recommended Architecture After Stage 3

```text
Top-k finalists (from Stage II)
  → Shared PPO harness (K3 schedule, early-stop optional)
  → Full-episode multi-seed metrics
  → Optional conservative refine (accept/reject only)
  → H-sweep for each remaining finalist (or winner + runner-up)
  → Final selector: task + social + generalization
  → Artifacts: policy ckpt, genome, diagnostics, H table
  → [Later] AMFRS Pareto archive update
```

### 5.5 Implementation Order

1. Artifact clarity for best_trained vs refined code.  
2. Pipeline top-k admission.  
3. Shared accept/reject refine with Stage II.  
4. H-aware final selection.  
5. Failure-mode diagnostics shared.  
6. Trainer deduplication.  
7. Tournament / early-stop / distill experiments.  
8. Co-adaptation / Pareto AMFRS (after interfaces stable).  

### 5.6 Validation Criteria

- Wall-clock ↓ at iso final quality.  
- Worst-H SR ↑ or constrained CR/ITR ↓ without tanking H_train SR.  
- Documented multi-seed mean±std.  
- Zero silent refine regressions on elite.  
- Paper-scale path still runnable with explicit K3=1e7 profile.  

---

## 6. Cross-Stage Dependencies

```text
Stage 1 trustworthy Score1
        ↓
RewardCandidate + sandbox contract (stable)
        ↓
Stage 2 stable ranking + verified refine
        ↓
Finalist set + shared metrics schema
        ↓
Stage 3 tournament + H/social objective
        ↓
AMFRS modules (Pareto/archive/diversity/active data)
```

| Dependency | What must precede what | Why | If violated |
|------------|------------------------|-----|-------------|
| S1 pad/guards before S1–S2 correlation studies | Else measure garbage Score1 | False research conclusions |
| S2 multi-seed / accept-reject before S3 top-k | Else top-k is noise | Waste K3 on wrong genomes |
| Shared selection scalar (D-3) before claiming AMFRS multi-obj | Else II/III optimize different fitness | Incomparable fronts |
| Trainer dedup before co-adaptation innov | Else double implementation | Divergent bugs |
| Failure-mode schema before fancy LLM refine claims | Else prompts untestable | Non-reproducible “innov” |
| Do **not** require full AMFRS before fixing Score1/proxy | AMFRS on bad metrics amplifies error | Beautiful archive of wrong rewards |
| Surrogate/active learning after logged II/III datasets exist | Need training data | Premature ML layer |

**Surrogate / active learning:** **Not Implemented.** Depend on structured metric logs + optional traj dumps from Stages II/III (`reporting.py` already mentions AMFRS-style raw records — **New Observation**).

---

## 7. Complete Problem Inventory

| ID | Stage | Problem | Severity | Proposed Solution | Priority | Dependency |
| -- | ----- | ------- | -------- | ----------------- | -------- | ---------- |
| S1-1 | 1 | Score1≠nav success | Critical | Holdout + optional proxy blend | P0 | Dataset; II eval |
| S1-2 | 1 | Coarse rules | High | Richer rule features | P1 | RewardState fields |
| S1-3 | 1 | Pad inflation | High | Mask/zero pads | P0 | None |
| S1-4 | 1 | Static success rule across frames | Medium | Frame-aware success term | P2 | S1-2; paper-fidelity D-1 |
| S1-5 | 1 | Constant/explosive rewards | High | Anti-exploit gates | P0 | Thresholds |
| S1-6 | 1 | Offline mismatch | High | Better data + optional on-policy refresh | P1 | Collector; later AL |
| S1-7 | 1 | Low diversity / clones | Medium | Fingerprint diversity | P2 | Smoke states |
| S1-8 | 1 | Template reflection | Medium | Diagnostics in prompts | P1 | scoring diagnostics |
| S1-9 | 1 | Smoke vs Score1 confusion | Medium | Artifact/CLI guards | P2 | reporting |
| S1-10 | 1 | Gen0 parse fragility | Medium | Fallback gen policy | P2 | llm extract |
| S1-11 | 1 | Gen regression vs global_best | Medium | Stronger in-evolver elitism | P2 | pipeline already helps |
| S2-1 | 2 | Noisy proxy ranks | Critical | Multi-seed metrics | P0 | Compute |
| S2-2 | 2 | Always cold-start | High | Conditional warm-start | P2 | **Needs Validation** |
| S2-3 | 2 | Unverified refine | Critical | Accept/reject | P0 | Verifier design |
| S2-4 | 2 | Flat LLM feedback | High | Failure-mode prompts | P1 | eval infos |
| S2-5 | 2 | A2C/short vs PPO/full | High | Align algo/horizon or calibrate | P1 | D-2 |
| S2-6 | 2 | Scalar ignores social | High | Multi-metric/Pareto select | P1 | D-3 |
| S2-7 | 2 | N×G2 cost wall | High | Early-stop / shrink N | P1 | Thresholds |
| S2-8 | 2 | Stub ≠ real | Medium | Slow integration tests | P3 | CI |
| S2-9 | 2 | Dup with Stage III | Medium | Shared harness | P1 | Characterization tests |
| S3-1 | 3 | Cost vs search | Critical | Top-k tournament | P0 | Good Stage II |
| S3-2 | 3 | Costly unverified refine | Critical | Strict accept/reject / limit rounds | P0 | S2-3 |
| S3-3 | 3 | Few G3 rounds | Medium | Don’t inflate G3; improve II + co-adapt once | P2 | D-4 |
| S3-4 | 3 | II↔III rank mismatch | High | Protocol align + logging | P1 | D-2 |
| S3-5 | 3 | H-sweep not in objective | High | H-aware selection | P1 | Sweep cost |
| S3-6 | 3 | Social underweighted | High | Constrained/Pareto final pick | P1 | D-3 |
| S3-7 | 3 | Thin critique despite rich eval | Medium | PPO failure modes → D.3 | P1 | S2-4 |
| S3-8 | 3 | GST/hardware fragility | Medium | Profiles + asserts | P2 | Docs |
| S3-9 | 3 | Artifact genome ambiguity | Medium | Schema split evaluated vs refined | P1 | reporting |
| S3-10 | 3 | Dup with Stage II | Medium | Shared harness | P1 | S2-9 |
| X-1 | Cross | No AMFRS in repo | — | Additive modules later | P3 | Stable metrics |
| X-2 | Cross | No surrogate | — | Optional after logs | P3 | Data |
| X-3 | Cross | No active learning | — | Optional after S1-6 | P3 | Data layer |

---

## 8. Complete Improvement Inventory

| ID | Stage | Improvement | Motivation | Expected Benefit | Cost | Priority |
| -- | ----- | ----------- | ---------- | ---------------- | ---- | -------- |
| I-S1-A | 1 | Pad mask + anti-exploit | Scoring bugs | Trustworthy Score1 | Low impl / low compute / low maint | P0 |
| I-S1-B | 1 | Holdout Score1 | Overfit dataset | Generalization of filter | Low–mid impl / low compute | P0 |
| I-S1-C | 1 | Richer rules | Coarse Figure-3 ops | Better analytical signal | Mid impl / low compute / mid maint | P1 |
| I-S1-D | 1 | Diagnostics→mutation | Weak reflection | Better LLM edits | Mid impl / low compute | P1 |
| I-S1-E | 1 | Diversity fingerprints | Clonal collapse | Wider search | Low–mid / low compute | P2 |
| I-S1-F | 1 | Mini-rollout / αβ blend | Offline gap | Align to nav | Mid–high impl / mid compute | P2 **Needs Validation** |
| I-S1-G | 1 | Active traj refresh | Covariate shift | On-policy Score1 | High impl / high compute / high maint | P3 **Not Implemented** |
| I-S2-A | 2 | Multi-seed proxy | Rank noise | Stable elite | Low impl / **high** compute | P0 |
| I-S2-B | 2 | Accept/reject refine | Open-loop LLM | Fewer regressions | Mid impl / mid compute | P0 |
| I-S2-C | 2 | Failure-mode D.3 | Blind refine | Higher quality edits | Mid impl / low–mid compute | P1 |
| I-S2-D | 2 | Multi-metric elite | Social blindness | Safer winners | Low–mid impl / low compute | P1 |
| I-S2-E | 2 | Early-stop / top-k | Cost wall | More experiments | Mid impl / saves compute | P1 |
| I-S2-F | 2 | Warm-start | Cold-start waste | Speed | Mid impl / **bias risk** | P2 **Needs Validation** |
| I-S2-G | 2 | PPO-short / horizon curriculum | II↔III mismatch | Better transfer | Mid impl / mid compute | P1 **Decision Pending** |
| I-S2-H | 2 | Uncertainty-aware select | Single-shot luck | Robust choice | Mid–high / mid compute | P2 **Needs Validation** |
| I-S2-I | 2 | Relative ranking / shared init | Absolute SR noise | Fairer compare | High impl / mid compute | P2 **Needs Validation** |
| I-S2-J | 2 | Learned surrogate | Expensive II/III | Cheap screening | **High** all costs | P3 **Needs Validation** |
| I-S2-K | 2 | Dual-loop freeze policy | Credit assignment | Better reward fit | High / high compute | P3 **Needs Validation** |
| I-S2-L | 2 | Constraint-aware proxy | Safety | Feasible rewards | Mid / low–mid | P2 |
| I-S2-M | 2 | AMFRS-lite diversity archive | Homogeneity | Coverage of modes | Mid / low–mid | P2 (pre-AMFRS) |
| I-S3-A | 3 | Top-k finalists | Cost | Focused K3 | Low impl / saves compute | P0 |
| I-S3-B | 3 | Strict refine gate | Costly regressions | Conservative finals | Mid / mid | P0 |
| I-S3-C | 3 | H-aware selection | Report-only sweep | True generalization | Mid / mid–high compute | P1 |
| I-S3-D | 3 | Constrained/Pareto final | Social | Thesis + safety | Mid / low–mid | P1 |
| I-S3-E | 3 | PPO failure-mode refine | Thin critique | Better last edits | Mid / low | P1 |
| I-S3-F | 3 | Shared harness w/ II | Duplication | Maintainability | Mid–high impl / low runtime | P1 |
| I-S3-G | 3 | Fair tournament seeds | Confounded compares | Scientific fairness | Mid / mid | P1 |
| I-S3-H | 3 | Co-adaptation loop | Few G3 | On-policy reward fit | High / high | P2 **Needs Validation** |
| I-S3-I | 3 | Budgeted promotion | Waste on losers | Allocate K3 smartly | Mid / saves compute | P2 |
| I-S3-J | 3 | Distill from II | Cold PPO | Faster converge | Mid / **Needs Validation** | P2 |
| I-S3-K | 3 | Reward attribution audit | Opacity | Analysis chapter | Mid / mid compute | P2 |
| I-S3-L | 3 | Cross-stage feedback | Persistent mismatch | Better proxy | Mid–high / mid | P2 **Needs Validation** |
| I-X-A | Cross | AMFRS Pareto/archive | Thesis goal | Multi-obj evolution | High / mid–high | P3 after P0–P1 |
| I-X-B | Cross | Surrogate module | Cost | Screen candidates | High all | P3 |
| I-X-C | Cross | Active learning data | Offline gap | Better Score1 data | High all | P3 |

Cost legend: implementation complexity, computational cost, runtime impact, maintenance cost considered qualitatively above.

---

## 9. Architecture Decisions

**Decision D-1 — Paper fidelity vs Stage I rule enrichment**

- **Question:** May we change analytical rules away from coarse Figure-3 buckets?  
- **Option A:** Keep rules strict for faithful RAISE replication; add enrichment only under a flagged `rules_version`.  
- **Option B:** Replace rules globally with richer preferences.  
- **Recommended:** **A**  
- **Reason:** Repo claims faithful Algorithm 1; enrichment is AMFRS/thesis delta and must be ablatable.  
- **Trade-offs:** Slower innovation vs clean baselines.  
- **Consequences:** Dual scoring paths in `rules.py`/`scoring.py`.  
- **Status:** Recommended (not yet implemented)

**Decision D-2 — Stage II algorithm alignment**

- **Question:** A2C short (status quo) vs short PPO vs horizon curriculum?  
- **Option A:** Keep A2C; invest in multi-seed + accept/reject.  
- **Option B:** Switch Stage II to PPO with small K2.  
- **Option C:** Curriculum horizon with either algo.  
- **Recommended:** **A first**, experiment **B/C** as ablations.  
- **Reason:** Lowest risk; mismatch may shrink once noise fixed; PPO-in-II raises cost and overlaps III.  
- **Trade-offs:** Possible residual mismatch.  
- **Consequences:** Calibration logging mandatory.  
- **Status:** **Decision Pending** (empirically)

**Decision D-3 — Selection fitness**

- **Question:** Keep `SR-CR-0.5TR` or multi-objective?  
- **Option A:** Single scalar (status quo).  
- **Option B:** Weighted scalar including ITR/SD.  
- **Option C:** Pareto / constrained selection (AMFRS-ready).  
- **Recommended:** **B short-term**, design APIs for **C**.  
- **Reason:** Immediate safety signal without full AMFRS; avoid pretending Pareto exists.  
- **Trade-offs:** Weight tuning (**Needs Validation**).  
- **Consequences:** `selection.py` becomes shared policy.  
- **Status:** Recommended direction; weights **Decision Pending**

**Decision D-4 — Stage III role**

- **Question:** Continue population evolution at K3 or finalist tournament?  
- **Option A:** Status quo N×G3.  
- **Option B:** Top-k tournament + optional one co-adapt on winner.  
- **Recommended:** **B**  
- **Reason:** Matches compute reality and thesis narrative (“full eval”).  
- **Trade-offs:** Less in-III exploration (accept; move search to I/II).  
- **Consequences:** Pipeline population truncate.  
- **Status:** Recommended

**Decision D-5 — When to introduce AMFRS**

- **Question:** Add archive/Pareto now or after metric stabilization?  
- **Option A:** Now.  
- **Option B:** After P0 Score1+proxy gates.  
- **Recommended:** **B**  
- **Reason:** AMFRS amplifies whatever fitness you give it.  
- **Trade-offs:** Delayed headline feature.  
- **Consequences:** Keep interfaces (`RewardCandidate`, metrics dicts, raw episode logs) clean.  
- **Status:** Recommended

**Decision D-6 — Surrogate / active learning**

- **Question:** Build ML surrogate early?  
- **Option A:** Yes as core path.  
- **Option B:** Defer until multi-run logs exist; use Stage II as procedural proxy.  
- **Recommended:** **B**  
- **Reason:** Avoid over-engineering; insufficient labeled III outcomes initially.  
- **Status:** Recommended; surrogate remains **Needs Validation** if revisited

---

## 10. Conflicts / Redundant Ideas

### 10.1 Conflicts

| Conflict | Side A | Side B | Resolution |
|----------|--------|--------|------------|
| Raise Stage III G3 for more refine | More search in III | Explodes cost (S3-1); fights top-k tournament | Prefer **verified refine + better II**, not larger G3 |
| Warm-start Stage II | Saves compute | Biases fair comparison | Allow only when genome unchanged or for fine-tune **inside** accept/reject verifier, not across different rewards without care |
| Richer Stage I rules | Better filter | Breaks paper fidelity | Dual `rules_version` (D-1) |
| Switch II to PPO | Align with III | Makes II expensive; duplicates III | Try after multi-seed; not default (D-2) |
| Mini-rollout inside Stage I vs strengthen Stage II | Both close offline gap | Two on-policy paths | Prefer **Stage II as on-policy filter**; Stage I mini-rollout optional light touch |
| Full AMFRS diversity now vs light fingerprints | Thesis feature | Complexity before metrics work | Fingerprints first; AMFRS archive later |
| Optimize H-sweep for all candidates | Fair generalization compare | Huge cost | Sweep top-2 finalists max initially |

### 10.2 Redundant / overlapping ideas (groupings)

| Theme | Ideas merged | Prefer implementation |
|-------|--------------|----------------------|
| Stabilize proxy rank | Multi-seed, uncertainty UCB, relative ranking | Start **multi-seed**; add UCB/relative only if still unstable |
| Stop bad LLM edits | Accept/reject II, strict gate III, protect_elite (exists) | Keep elite protect; add **accept/reject** as main innov |
| Offline→online gap | Richer rules, holdout, mini-rollout I, active data, II blend | **Holdout + II quality** first |
| Social/safety | Scalar weights, constraints II, Pareto AMFRS, H-aware | Weighted/constrained scalar → Pareto |
| Cost control | Early-stop II, top-k III, budgeted promotion, don’t raise G3 | **top-k + early-stop** |
| Shared code | Dedup trainers, shared diagnostics, shared selection | One harness module |
| Surrogate vs Stage II | Learned surrogate vs procedural proxy | Keep Stage II; surrogate later |

### 10.3 Over-engineering / premature optimization risks

- Learned Stage-III surrogate before ≤ tens of diverse full runs  
- Full AMFRS controller before Score1 pad bugs fixed  
- Dual-loop co-adaptation as default path before accept/reject exists  
- Per-H training (not eval) for all H — usually unnecessary vs zero-shot sweep + H-aware select  

### 10.4 Unnecessary abstractions to avoid early

- Separate microservices / plugin systems beyond Python modules  
- New reward language beyond `compute_reward(state, memory)`  
- Replacing sandbox wholesale (improve policy, don’t rewrite)  

---

## 11. Target Architecture

### 11.1 Modules & responsibilities

| Module | Responsibility |
|--------|----------------|
| `sandbox/` | Validate/execute LLM reward code |
| `state.py` | RewardState contract |
| `dataset/rules/scoring` | Versioned analytical Score1 + diagnostics |
| `evolver` | Stage I search (+ diversity) |
| `train_harness` (**target**, Not Implemented as unified) | Shared vec-env train/eval for II/III |
| `stage2` / `stage3` configs | Budgets & protocols only |
| `refine` (**target** extract) | D.3 propose + repair + accept/reject |
| `selection` | Task/social/H-aware / future Pareto |
| `diagnostics` (**target**) | Failure-mode summaries from rollouts |
| `pipeline` | Orchestration + artifacts |
| `amfrs/` (**Not Implemented**) | Archive, Pareto, novelty, adaptive controller |
| `surrogate/` (**Not Implemented**) | Optional cheap predictor |
| `active_data/` (**Not Implemented**) | Optional traj acquisition |

### 11.2 Interfaces (stable)

- `RewardCandidate{code, reward_fn, score, metadata}`  
- `ProxyMetrics` / metrics dict with SR,CR,TR,NT,PL,ITR,SD (+ future fields)  
- `RewardValidator.validate_code`  
- `score_fn(reward_fn) -> float` (+ diagnostics object)  
- `TrainEvalResult{metrics, ckpt, diagnostics, seeds}`  

### 11.3 Flows

**Experiment flow**

```text
Config/regime → Pipeline
  → Stage I (Score1 train/holdout)
  → Stage II (proxy multi-seed + verified refine)
  → Stage III (finalists + PPO + H-aware select)
  → Optional AMFRS update of archive
  → Manifest / reports
```

**Reward evaluation flow**

```text
code → sandbox → RewardFunction
  → Score1 diagnostics OR
  → env.step via build_reward_state → compute
```

**Surrogate flow (target, optional)**

```text
logs(feature(reward), proxy_metrics, final_metrics)
  → fit surrogate
  → screen Stage I pop before Stage II
```

**RL train/eval flow**

```text
reward_fn → make_vec_envs → Policy → A2C/PPO updates → evaluate_proxy_policy → metrics/diagnostics
```

**Active-learning flow (target, optional)**

```text
disagreement / failure clusters → collect new trajs → extend Stage1Dataset → rescore
```

**Safety/constraint flow (target)**

```text
metrics → constraint check (CR, ITR, ...)
  → reject elite OR Lagrangian penalty OR Pareto dominate
```

### 11.4 Diagram

```text
                    ┌──────────── sandbox ────────────┐
LLM / seed code ──► │ AST + exec + smoke               │
                    └─────────────┬───────────────────┘
                                  ▼
                         RewardCandidate
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   Stage I Score1          Stage II proxy            Stage III PPO
   (rules+dataset)         (multi-seed)              (finalists)
          │                       │                       │
          │                       ├─ diagnostics ─────────┤
          │                       ├─ accept/reject refine ┤
          │                       ▼                       ▼
          │                  selection.py ◄──────── H-sweep/social
          │                       │
          └───────────────────────┴──────────► artifacts
                                  │
                                  ▼
                     [AMFRS Pareto/archive]   Not Implemented
                     [surrogate/active data] Not Implemented
```

---

## 12. Final Implementation Roadmap

### Phase 1 — Trustworthy Stage I (P0)

| Task | What | Where | Why | Dependency | Expected result | Validation |
|------|------|-------|-----|------------|-----------------|------------|
| T1 | Pad masking + finite/variance guards | `scoring.py`, tests | Fix Score1 bias | None | Stable Score1 | Unit tests unequal lengths / constant r |
| T2 | Holdout split + manifest fields | `dataset`/`scoring`/`pipeline` | Detect overfit | T1 | train/holdout scores logged | Fixture dataset split |
| T3 | Smoke/paper Score1 hard labeling | CLI/reporting | Avoid false claims | None | Clear mode in artifacts | Test manifest |

### Phase 2 — Stable Stage II gate (P0–P1)

| Task | What | Where | Why | Dependency | Expected result | Validation |
|------|------|-------|-----|------------|-----------------|------------|
| T4 | Multi-seed proxy aggregate | `stage2.py` | Rank stability | Phase1 optional | mean±std in metadata | Repeatability study |
| T5 | Accept/reject refine | `stage2.py`, prompts | Stop regressions | T4 helpful | Only improving codes kept | Synthetic refine tests |
| T6 | Failure-mode diagnostics v0 | `stage2.evaluate_proxy_policy`, `prompts.py` | Better D.3 | Env infos | Richer feedback text | Prompt schema tests |
| T7 | Selection weights v0 (incl. ITR) | `selection.py` | Social signal | Agree D-3 weights | Shared elite fitness | Sensitivity notes |

### Phase 3 — Stage III as finalist eval (P0–P1)

| Task | What | Where | Why | Dependency | Expected result | Validation |
|------|------|-------|-----|------------|-----------------|------------|
| T8 | Top-k admission | `pipeline.py` | Save K3 | T4–T5 | Smaller III pop | Iso-quality GPU-hours |
| T9 | Strict refine policy | `stage3.py` | Costly mistakes | T5 | Conservative III | No elite refine harm |
| T10 | Artifact genome clarity | `reporting.py`, pipeline | Reproducibility | None | evaluated vs refined fields | Unit tests |
| T11 | H-aware final select | `stage3.py`, `selection.py` | Generalization | T8 | Winner considers H | Worst-H metric |

### Phase 4 — Maintainability + alignment (P1)

| Task | What | Where | Why | Dependency | Expected result | Validation |
|------|------|-------|-----|------------|-----------------|------------|
| T12 | Shared train/refine harness | new module ← stage2/3 | One innov path | Characterization tests | Less duplication | Smokes II/III |
| T13 | II↔III rank logging | reporting | Measure mismatch | T4,T8 | Calibration table | Offline analysis |
| T14 | Stage I diagnostics→mutation | `evolver`, `scoring` | Better I search | T1–T2 | Higher holdout Score1 | Ablation |

### Phase 5 — Thesis / AMFRS-ready (P2–P3)

| Task | What | Where | Why | Dependency | Expected result | Validation |
|------|------|-------|-----|------------|-----------------|------------|
| T15 | Diversity fingerprints / lite archive | evolver (+ future amfrs) | Coverage | Phase1 | Lower clone rate | Stats |
| T16 | Rules_version rich rules | `rules.py` | Thesis delta | D-1 | Ablatable enrichment | Paper baseline intact |
| T17 | Constrained / Pareto selection | `selection` / `amfrs/` | Multi-obj | T7,T11 | Fronts not single scalar | Plots |
| T18 | Optional co-adapt on III winner | stage3 | On-policy reward | T9,T12 | Lift without full G3↑ | **Needs Validation** |
| T19 | Surrogate / active data | new packages | Scale | Many III logs | Cheap screening | **Needs Validation** |

Order is **dependency-first**, not chat chronology.

---

## 13. What NOT to Change Yet

- **Do not** implement full AMFRS (Pareto controller, novelty archive, adaptive multi-obj) before Phase 1–3 metrics work.  
- **Do not** replace `compute_reward(state, memory)` contract or rewrite sandbox from scratch.  
- **Do not** make learned surrogates a default dependency.  
- **Do not** inflate `G3` or default `K3` as a substitute for top-k + accept/reject.  
- **Do not** silently change Figure-3 rules without `rules_version` (breaks faithful baseline).  
- **Do not** remove elite protection / global_best injection until accept/reject proven.  
- **Do not** merge Stage II and III into one budget (loses cheap filter).  
- **Do not** treat `--fast` / stub metrics as scientific evidence.  
- **Do not** retarget `train.py` as Algorithm 1 entry (stages intentionally bypass it).  
- **Avoid** per-H full retraining for Table-6 style sweeps initially.  

---

## 14. Minimal Path to AMFRS

AMFRS needs trustworthy **candidates**, **vector-valued outcomes**, and **archive-friendly artifacts**. Minimal path:

### Essential

1. Fix Stage I scoring artifacts (pad, anti-exploit, holdout).  
2. Stabilize Stage II ranking (multi-seed) + **accept/reject** refine.  
3. Stage III **top-k tournament** + clear genome/metrics artifacts.  
4. Evolve `selection.py` from scalar → **vector metrics** (SR, CR, TR, ITR, …) suitable for Pareto.  
5. Keep sandbox + `RewardCandidate` as the unit of evolution.  
6. Persist raw per-episode records (lean on / extend `reporting.py` AMFRS-oriented notes).  

### Optional (after essentials)

- Richer rules_version  
- Failure-mode prompts  
- Shared harness  
- H-aware objective  
- Light diversity fingerprints  
- Single co-adaptation pass on final winner  

### Explicitly defer

- Full adaptive AMFRS controller  
- Learned surrogate as core  
- Active learning infrastructure  
- Dual-loop as default Stage II  
- Large G3 population evolution at 1e7 steps  

### Critical stance

AMFRS on top of noisy Score1 and unverified D.3 refines would produce a **multi-objective archive of noise**. The minimum necessary change is not “add Pareto,” it is **make the fitness channels honest**, then plug AMFRS in as an additive selector/archiver.

---

## Final Recommendation

1. **Phase 1:** Make Stage I Score1 scientifically trustworthy (mask pads, anti-exploit, holdout).  
2. **Phase 2:** Make Stage II a reliable gate (multi-seed metrics + accept/reject refine + shared clearer fitness).  
3. **Phase 3:** Convert Stage III into a **finalist PPO tournament** with H/social-aware selection and strict refine policy.  
4. **Only then** add AMFRS (Pareto/archive/diversity) on stable candidate/metrics interfaces — do not expand G3/K3 or build surrogates as a substitute for those gates.

This document is the single source of truth for subsequent Stage 1–3 / AMFRS refactor planning; update it when Decisions D-2/D-3 weights are resolved empirically.

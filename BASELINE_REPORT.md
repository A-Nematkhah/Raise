# BASELINE LOCK REPORT — `baseline-pre-amfrs`

**Date:** 2026-09-19 (paper-default fidelity lock)  
**Purpose:** Freeze the RAISE Algorithm 1 replication (CrowdNav++ fork) before AMFRS work.  
**Tag:** `baseline-pre-amfrs` — use for diffs; defaults now match paper K2 unit, R2/R3, and no Stage II/III elitism. Read **Documented deviations** below before claiming full paper parity.

**Scope of this lock:** Domain Pack + fidelity honesty + paper-faithful **defaults** (K2=`gradient_steps`/8000, `final_rank=llm`, `elitism=off`, best from R2/R3). **No AMFRS.**

**Honesty note:** Do **not** label this tree “byte-faithful.” Algorithm 1 skeleton and Tables 3–6 knobs are largely matched; remaining deviations are listed below. Cross-check: arXiv:2605.11859.

---

## Verdict (three categories — do not blur)

### Faithful to Algorithm 1 / Tables 3–6 (structural)

| Area | Evidence |
|------|----------|
| Stage I population / gens | `N=8`, `G1=10` — `StageIConfig`, `RaiseRunConfig`, `configs/paper_scale.yaml` |
| Stage I Score1 structure | Spearman(rules, cumulative recomputed reward) over dataset — Eq. 1 / `scoring.py` |
| Category order | Success ≫ Other ≫ Fail — `rules.py` (Figure 3 category dominance) |
| Stage II algo / rounds / eval | A2C, `G2=16`, `E2=50`, `T_short=100` — `Stage2Config` |
| Stage III algo / rounds / eval / H set | PPO, `G3=3`, `E3=500`, H∈{5,10,15,20} — `Stage3Config` |
| Pass all N candidates stage-to-stage | Matches Algorithm 1 (`P←P`), not the prose “small subset” |
| Fresh policy each round | Stage II/III trainers |
| D.3 refine uses raw metrics | Not rank ids — §4.2 |
| Paper-scale numeric budgets | YAML `K2=8000`, `K3=1e7` — **see K2 unit deviation** |
| Fail-closed `--llm seed` | `run_raise.py` / `run_raise_paper_scale.py` |
| Sandbox bans `__import__` | AST + restricted builtins |
| GST regime asserts | `regime.py` + stage entry points |
| Dynamic `num_processes` | `None` → `min(16, cpu_count-1)` |

### Documented deviations (must cite in any “vs paper” claim)

| Deviation | Paper (arXiv:2605.11859) | This baseline | Why / notes |
|-----------|--------------------------|---------------|-------------|
| **R2 / R3 ranking** | Alg. 1 lines 20, 30: LLM evaluation of `{M(r)}` | Default **`final_rank=llm`** (seed → multi-objective lex fallback); `best_stage*` follows R2/R3. Opt-in `--final-rank scalar` | Within-round `best_trained` tracking still uses SR−CR−0.5·TR |
| **M(r) tuple** | Alg. 1 lists `(SR, NT, PL, ITR, SD)` (no CR/TR); §5.1 has seven metrics | Logs seven metrics; LLM/lex rank uses all seven | Paper inconsistent; we keep seven logs. |
| **K2 unit** | §4.3.2: **K2 gradient steps**; Table 5 = 8000 | Default **`gradient_steps`** + K2=`8000`; opt-in `--k2-unit env_steps` | `env_steps = 8000 × num_steps × num_processes` |
| **K2 default (CLI)** | Table 5: 8000 | Default **8000** gradient steps (same as paper_scale) | Previously 50k env steps; that was a practical deviation |
| **K3 default** | Table 6: `1e7` **environment** steps (§4.3.3) | Default `5e5`; paper_scale `1e7` | Hardware; unit matches paper for K3. |
| **Stage I next-gen split** | Unspecified counts; mutation/crossover/random | Config claims 2/4/2; **runtime** ≈ 2 crossover + 4 mutation + **1** random + **1 elite carry** (`_next_generation`) | Elite slot steals one random (Stage I evolver; separate from Stage II/III elitism flag). |
| **Elitism (II/III)** | Algorithm 1 has no elitism | Default **off**; opt-in `--elitism` | Inject/protect when enabled |
| **Mutation parent vs D.2 prompt** | Body §4.2: mutation addresses weaknesses; appendix D.2 said elite | Mutates lower half; prompt says **underperforming parent** | Aligned to §4.2 |
| **Reflection** | “accumulated across generations” | Accumulates up to 3 gen notes with all Score1 ids | Bounded accumulation |
| **N_traj mix** | Only “10 diverse trajectories” | Our mix: ORCA/SF/noise/random (see collector) | Do **not** claim the mix is from the paper. Older datasets may have duplicate ORCA/SF rollouts (~5 unique / 10). |
| **Score1 Success short/long** | Figure 3: Success short nav ≻ Success long | Was `nav_length=f+1` (ties all Success at frame f); **fixed toward** `min(f+1, traj.length)` in fidelity pass | Avoids final-length leak into early frames while restoring short≻long after shorter Success ends. |
| **Reward interface** | `cal_reward(st)` / `compute_reward(inst, traj)` / D.5 | `compute_reward(state, memory)` + `RewardState` (no `m_t` predictions) | Paper itself inconsistent; sandbox-friendly contract. |
| **D.5 seed** | `2·(‖prev−start‖ − ‖curr−prev‖)` | Goal-distance potential × 2 (CrowdNav++-style) | Paper formula looks buggy; prose mentions distance-to-goal shaping. |
| **Obs / GST** | CrowdNav++ predictive | Default `predict_method=inferred` | AUDIT §8.2 |
| **Regime default** | with/without random tables | Default `without_random` | First honest pass |
| **`--easy` / `--fast`** | N/A | Stubs / smoke Score1 | Not for paper claims |

### Open gaps (empirical / future)

| Item | Status |
|------|--------|
| `run_scaled_h5_gst` ≪ ORCA/GST under reduced budget | Empirical — not a closed Score1 pad bug |
| Proxy Spearman Stage I↔II↔III (§4.3.4) | Written to `proxy_consistency.json` each run (`proxy_consistency.py`) |
| AMFRS / surrogate / active learning | Not in this repo |
| `train.py --algo a2c` | Unused by Algorithm 1 (`AUDIT.md`) |

Part A pad freeze + degeneracy reporting remain closed. Success-tier nav-length policy: see fidelity pass / `scoring.py`.

---

## Part A — Score1 fixes

### 1. Cumulative reward freeze on padding

`_cumulative_reward` only calls `reward_fn.compute()` while `f < traj.length`; afterwards repeats the last real cumulative.

### 2. Per-frame Success nav-length (Figure 3)

- **Lock intent:** no leak of *future* final length into early frames of an unfinished traj.  
- **Fidelity:** `nav_length = min(f + 1, traj.length)` so after a short Success ends, later frames keep that short length and prefer short Success over long Success still running / longer completed.  
- Tests: `test_score1_baseline_lock.py` (pad + Success short vs long discrimination).

### 3. `degenerate_fraction`

`Score1Result` exposes degeneracy; evolver warns at high rates.

---

## Part B — Cleanup summary

(Unchanged in spirit from 2026-09-19 lock — caches removed, results archived under `results/archive/`, credentials clean. See git history for file moves.)

**Kept live primary run:** `raise_env/results/run_scaled_h5_gst/` (pre-Score1-lock artifacts — not evidence of post-lock reproduction).

---

## Part C — Fidelity detail (Algorithm 1)

### Stage I

| Knob | Paper | Code | Notes |
|------|-------|------|-------|
| N, G1 | 8, 10 | 8, 10 | Faithful |
| M, N_traj | 100, 10 | collector defaults | Mix is ours (ORCA/SF/noise/random). Re-collect after 2026-09 seed fix so random trajs are not bit-duplicates. |
| Operators | 2/4/2 style | default **exact 2/4/2**; opt-in `--elitism` keeps top-1 (steals one random → 2/4/1) | Runtime elite is behind `--elitism` |
| Score1 | Eq. 1 | `scoring.py` | Structure faithful; Success tier see deviations |

### Stage II

| Knob | Paper | Code | Notes |
|------|-------|------|-------|
| K2 | 8000 **gradient** steps | default **8000 gradient_steps** (`k2_unit`) | Env-steps realized = K2 × num_steps × num_processes (see cost note) |
| Final R2 | LLM multi-objective | default `final_rank=llm` (fallback multiobjective_lex) | `--final-rank scalar` is engineering opt-in |
| Elitism | — | default **off**; `--elitism` enables inject/protect | Faithful default |

### Stage III

| Knob | Paper | Code | Notes |
|------|-------|------|-------|
| K3 | 1e7 **env** steps | 5e5 / 1e7 | Unit OK; default scaled |
| Final R3 | LLM multi-objective | default `final_rank=llm` | Same as R2 |
| H-sweep | {5,10,15,20} | yes (clipped) | Faithful structure |

### Elitism (explicitly non-paper)

Opt-in `--elitism`: Stage I runtime keep-top-1; Stage I→II `_include_global_best`; Stage II/III `_inject_elite` + `protect_elite_refine`. **Default off** (matches Algorithm 1).

### K2 wall-clock / env-step cost (planning)

With `k2_unit=gradient_steps`, each Stage II train does roughly
`K2 × num_steps × num_processes` env steps (default `num_steps=5`).
Example: K2=8000, 16 processes → ~640k env steps **per train**; 8×16=128 trains → ~82M env steps (~⅓ of one Stage III K3=1e7×24 if scaled that way).
`num_processes` is machine-dependent when left auto — prefer explicit `--num-processes` for reproducible budgets (bootstrap defaults to 1).

---

## Part D — Verification

### Tests (update when re-locking)

```text
pytest crowd_nav/reward_search/tests -m "not slow"
```

### Tag

```bash
git tag -a baseline-pre-amfrs -m "Paper-faithful RAISE baseline (defaults match Alg.1 / §4.3.2)"
git diff baseline-pre-amfrs
```

This tag is the AMFRS comparison point. Move it only when intentionally re-locking the baseline.

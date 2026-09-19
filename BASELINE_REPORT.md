# BASELINE LOCK REPORT — `baseline-pre-amfrs`

**Date:** 2026-09-19 (fidelity honesty pass: 2026-09-19 evening)  
**Purpose:** Freeze the EvoNav Algorithm 1 replication (CrowdNav++ fork) before AMFRS work.  
**Tag:** `baseline-pre-amfrs` — use for diffs, but read **Documented deviations** below before claiming paper parity.

**Scope of the original lock:** Stage I Score1 pad/nav/degeneracy fixes, cleanup, tests, smoke. **No AMFRS.**

**Honesty note:** Do **not** label this tree “byte-faithful.” Algorithm 1 skeleton and Tables 3–6 knobs are largely matched; several operators, ranking, K2 units, and Stage I data diversity are deliberate or accidental deviations (see below). Cross-check: arXiv:2605.11859 + fidelity review vs commit/tag.

---

## Verdict (three categories — do not blur)

### Faithful to Algorithm 1 / Tables 3–6 (structural)

| Area | Evidence |
|------|----------|
| Stage I population / gens | `N=8`, `G1=10` — `StageIConfig`, `EvoNavRunConfig`, `configs/paper_scale.yaml` |
| Stage I Score1 structure | Spearman(rules, cumulative recomputed reward) over dataset — Eq. 1 / `scoring.py` |
| Category order | Success ≫ Other ≫ Fail — `rules.py` (Figure 3 category dominance) |
| Stage II algo / rounds / eval | A2C, `G2=16`, `E2=50`, `T_short=100` — `Stage2Config` |
| Stage III algo / rounds / eval / H set | PPO, `G3=3`, `E3=500`, H∈{5,10,15,20} — `Stage3Config` |
| Pass all N candidates stage-to-stage | Matches Algorithm 1 (`P←P`), not the prose “small subset” |
| Fresh policy each round | Stage II/III trainers |
| D.3 refine uses raw metrics | Not rank ids — §4.2 |
| Paper-scale numeric budgets | YAML `K2=8000`, `K3=1e7` — **see K2 unit deviation** |
| Fail-closed `--llm seed` | `run_evonav.py` / `run_evonav_paper_scale.py` |
| Sandbox bans `__import__` | AST + restricted builtins |
| GST regime asserts | `regime.py` + stage entry points |
| Dynamic `num_processes` | `None` → `min(16, cpu_count-1)` |

### Documented deviations (must cite in any “vs paper” claim)

| Deviation | Paper (arXiv:2605.11859) | This baseline | Why / notes |
|-----------|--------------------------|---------------|-------------|
| **R2 / R3 ranking** | Alg. 1 lines 20, 30: LLM evaluation of `{M(r)}` | Default **scalar** `SR−CR−0.5·TR`; CLI `--final-rank llm` | Elite selection still uses scalar; R2/R3 artifacts recorded |
| **M(r) tuple** | Alg. 1 lists `(SR, NT, PL, ITR, SD)` (no CR/TR); §5.1 has seven metrics | Logs seven metrics; selects on SR/CR/TR scalar | Paper inconsistent; we keep seven logs. |
| **K2 unit** | §4.3.2: **K2 gradient steps**; Table 5 = 8000 | Default **env_steps**; CLI `--k2-unit gradient_steps` | With env_steps+8000, A2C updates ≪ 8000 |
| **K2 default (non-paper CLI)** | Table 5: 8000 | Default **50_000** env steps; paper_scale forces 8000 | Practical ranking; still not gradient-step unless flagged |
| **K3 default** | Table 6: `1e7` **environment** steps (§4.3.3) | Default `5e5`; paper_scale `1e7` | Hardware; unit matches paper for K3. |
| **Stage I next-gen split** | Unspecified counts; mutation/crossover/random | Config claims 2/4/2; **runtime** ≈ 2 crossover + 4 mutation + **1** random + **1 elite carry** (`_next_generation`) | Elite slot steals one random. |
| **Elitism** | Algorithm 1 has no elitism | Default **on**; CLI `--no-elitism` | Inject/protect disableable |
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

**Kept live primary run:** `evonav_env/results/run_scaled_h5_gst/` (pre-Score1-lock artifacts — not evidence of post-lock reproduction).

---

## Part C — Fidelity detail (Algorithm 1)

### Stage I

| Knob | Paper | Code | Notes |
|------|-------|------|-------|
| N, G1 | 8, 10 | 8, 10 | Faithful |
| M, N_traj | 100, 10 | collector defaults | Mix is ours |
| Operators | unspecified split | runtime elite + 2/4/1 | Deviation |
| Score1 | Eq. 1 | `scoring.py` | Structure faithful; Success tier see deviations |

### Stage II

| Knob | Paper | Code | Notes |
|------|-------|------|-------|
| K2 | 8000 **gradient** steps | env steps (8k or 50k) | Major unit deviation |
| Final R2 | LLM multi-objective | nav scalar | Deviation |
| Elitism | — | on by default | Deviation |

### Stage III

| Knob | Paper | Code | Notes |
|------|-------|------|-------|
| K3 | 1e7 **env** steps | 5e5 / 1e7 | Unit OK; default scaled |
| Final R3 | LLM multi-objective | nav scalar | Deviation |
| H-sweep | {5,10,15,20} | yes (clipped) | Faithful structure |

### Elitism (explicitly non-paper)

Stage I `global_best`; pipeline handoff; Stage II/III `_inject_elite` + `protect_elite_refine`. Useful engineering; **not** Algorithm 1.

---

## Part D — Verification

### Tests (update when re-locking)

```text
pytest crowd_nav/reward_search/tests -m "not slow"
```

### Tag

```bash
git tag -a baseline-pre-amfrs -m "EvoNav Algorithm 1 baseline before AMFRS"
git diff baseline-pre-amfrs
```

Re-tag or add `baseline-fidelity-honest` after this honesty + Score1/collector pass if you need a clean AMFRS comparison point.

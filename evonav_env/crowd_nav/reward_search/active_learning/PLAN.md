# Active Learning — Technical Plan

**Status:** skeleton only (stubs raise `NotImplementedError`).  
**Package:** `crowd_nav.reward_search.active_learning`  
**Queue / logs:** `evonav_env/data/active_learning/`  
**Hard dependency:** Surrogate v1 (`artifacts/surrogate` + uncertainty API).  
**Soft dependency:** Stage I collector (`scripts/collect_stage1_dataset.py`), surrogate `dataset_io.append_example`.

Implement **after** surrogate bootstrap produces a real model. Until then, only fill stubs + unit tests with fake predictions.

---

## 1. Goal

Whenever Score1, surrogate, and/or short Stage II **disagree** or the surrogate is **uncertain**, automatically acquire the most informative new data:

1. new Stage I scenarios / trajectories (cheap), and/or  
2. new surrogate labels via short Stage II (medium cost),  

then append to the right dataset and periodically **re-fit** the surrogate.

Active Learning is an **incremental continuation** of surrogate data collection — not a replacement for the one-shot `bootstrap_surrogate` pass.

---

## 2. Relationship to Surrogate

```text
Bootstrap (once)          Active Learning (recurring)
─────────────────         ───────────────────────────
build many labels   →     query only uncertain / disagreeing items
fit model           →     re-fit when n_new ≥ refit_every
save artifacts/     →     update same artifacts/surrogate
```

**Rule:** If no surrogate model is on disk, AL must no-op or exit 2 with a clear message (`Surrogate model required`).

---

## 3. Query kinds

| `kind` | What we acquire | Writes to | Cost |
|--------|-----------------|-----------|------|
| `stage1_scenario` | 1 scenario × N_traj (or 1 traj hard case) | side pool or merged Stage I dataset | low |
| `stage2_label` | Score1 features + Stage II short metrics for a candidate | `data/surrogate_dataset/*.jsonl` | medium |
| `stage3_label` | (v2+) short Stage III metrics | same / separate tag | high |

**v1 scope:** implement `stage1_scenario` + `stage2_label` only.

---

## 4. Scoring function (priority)

Implement `query.score_queries`.

Inputs per candidate \(c\):

- \(s_1(c)\): Score1 (train score / raw)  
- \(\hat{y}(c), u(c)\): surrogate prediction + uncertainty  
- optional \(y_{\mathrm{II}}(c)\): if a short Stage II already ran this generation  

### Priority heuristics (v1, combine with weights)

1. **Uncertainty:** \(u(c)\) (normalized to ~[0,1] via dataset quantiles).  
2. **Disagreement Score1 vs surrogate:**  
   \[
   d_{1\hat{y}} = \lvert \mathrm{rank\_z}(s_1) - \mathrm{rank\_z}(\hat{y}) \rvert
   \]
   (use z-scores or rank percentiles within the current batch).  
3. **Borderline promotion:** \(\hat{y}\) near the promote/drop threshold.  
4. **Diversity bonus:** prefer candidates whose `behavior_fingerprint` is far from those already queued / recently labeled (L2 distance).

```text
priority = w_u * u_norm + w_d * d_1y + w_b * borderline - w_div * redundancy
```

Default weights: `w_u=0.45`, `w_d=0.35`, `w_b=0.15`, `w_div=0.15` (tune later).

Emit top-`k` `QueryItem`s. For each item set `reason` string for logs (e.g. `uncertainty=0.82; disagree_s1_surr=0.61`).

### Mapping kind

- If missing Stage II label and candidate still in play → prefer `stage2_label`.  
- If Score1 scenario diagnostics show a cluster of weak `sid`s shared by many uncertain candidates → emit `stage1_scenario` with those `scenario_id` seeds / layout keys.  
- Cap per step: `max_stage2_labels`, `max_stage1_scenarios` (config).

---

## 5. Loop (`loop.run_active_learning_step`)

One iteration:

1. Load surrogate from `surrogate_model_dir`.  
2. Load current candidate batch (from last Stage I population JSON path **or** args).  
3. `predict` each → `score_queries` → `enqueue`.  
4. `dequeue_batch(limit=max_queries)`.  
5. For each item: `acquire.execute_query` → `mark_done`.  
6. If `n_done_since_refit >= refit_every` or `force_refit`: call surrogate `fit` on full `surrogate_dataset` and overwrite `artifacts/surrogate`.  
7. Write step summary into `data/active_learning/steps.jsonl`.

### CLI (later)

```bash
python scripts/run_active_learning_step.py \
  --surrogate artifacts/surrogate \
  --candidates results/<run>/stage1_population.json \
  --max-queries 5 \
  --refit-every 20
```

Optional: call from `EvoNavPipeline` every N Stage I generations behind `--active-learning`.

---

## 6. Acquire details

### 6.1 `stage2_label`

Reuse surrogate bootstrap labeling path:

- validate candidate code;
- Score1 + `extract_candidate_features`;
- Stage II short (same budget constants as bootstrap config.json);
- `surrogate.dataset_io.append_example`.

### 6.2 `stage1_scenario`

Options (pick one for v1):

**A (safer):** write new scenarios under `data/active_learning/stage1_extra/` without mutating the locked paper dataset; Score1 fn can optionally **union** train set with extra (config `score1_include_al_extra=True`).  

**B (aggressive):** append into `data/stage1_dataset` (requires rewrite npz — harder; avoid in v1).

Recommend **A** for tomorrow+.

Collector: factor a function from `collect_stage1_dataset.py`:

```python
collect_scenarios(scenario_ids, out_dir, n_traj=10, ...)
```

so AL can request specific layouts / seeds.

---

## 7. On-disk layout

```text
data/active_learning/
  queue.jsonl           # pending QueryItems
  done.jsonl            # finished + result blob
  steps.jsonl           # per-loop summaries
  manifest.json
  stage1_extra/         # optional traj pool (fmt=npz)
```

---

## 8. Pipeline integration (later)

| Hook | Behavior |
|------|----------|
| End of Stage I generation | optional AL step on underperformers |
| Before Stage III | AL `stage2_label` on uncertain top-half |
| Nightly cron / manual | `run_active_learning_step` on latest run dir |

Always record: queries issued, wall time, whether refit ran, surrogate val metric delta.

---

## 9. Tests (minimum)

| Test | Assert |
|------|--------|
| `test_score_queries_orders_by_uncertainty` | higher u → higher priority |
| `test_queue_roundtrip` | enqueue / dequeue_batch |
| `test_loop_noop_without_model` | clear error / empty summary |
| `test_execute_stage2_label_stub` | with stub trainer appends jsonl |

---

## 10. Acceptance criteria for “done v1”

1. Given a fitted surrogate + fake population, `score_queries` returns deterministic top-k.  
2. Queue persists across process restarts.  
3. One `stage2_label` query appends a valid surrogate example.  
4. `refit_every` triggers `SurrogateModel.fit` and updates artifacts.  
5. Documented in `docs/WHATS_NEW.md` when enabled by default (off by default initially).

---

## 11. Non-goals (v1)

- Bayesian GP surrogates / deep acquisition networks.  
- Human-in-the-loop labeling UI.  
- Replacing full Stage I recollect.  
- Running AL without surrogate uncertainty.  
- Automatic merge into paper-scale Stage I npz.

---

## 12. Suggested implementation order

1. Finish **surrogate** bootstrap v1 first.  
2. `query.QueryItem` + `queue` jsonl I/O.  
3. `score_queries` with uncertainty-only ranking (weights later).  
4. `acquire` for `stage2_label` reusing bootstrap helpers.  
5. `loop` + CLI.  
6. `stage1_scenario` extra pool.  
7. Pipeline flag `--active-learning`.  

---

## 13. Config knobs (yaml or CLI)

```yaml
active_learning:
  enabled: false
  max_queries_per_step: 5
  refit_every: 20
  weights: { u: 0.45, disagree: 0.35, borderline: 0.15, diversity: 0.15 }
  promote_threshold: 0.0    # on predicted scalar; tune from val set
  stage1_extra_dir: data/active_learning/stage1_extra
  label_budget: stage2_short
```

Store a copy next to the surrogate `config.json` when AL runs so experiments are reproducible.

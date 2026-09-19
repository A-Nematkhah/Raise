# Surrogate — Technical Plan

**Status:** skeleton only (stubs raise `NotImplementedError`).  
**Package:** `crowd_nav.reward_search.surrogate`  
**Data root:** `evonav_env/data/surrogate_dataset/`  
**Artifact root:** `evonav_env/artifacts/surrogate/`  
**Depends on:** Stage I dataset + Score1 (`scoring.py`), optional Stage II trainer via Domain Pack.

This document is the implementation contract for the first working version. Active Learning consumes this API but lives in a sibling package.

---

## 1. Goal

Train a **cheap regressor/classifier** that maps inexpensive features of a reward candidate to **expensive proxy outcomes** (primarily short Stage II metrics), so Algorithm 1 / AMFRS can:

- drop hopeless candidates before long Stage II/III;
- rank promotion priority;
- expose **uncertainty** that Active Learning will query.

The surrogate is **not** a substitute for paper-claim Stage III numbers. Final tables still come from real PPO / H-sweep.

---

## 2. Problem formulation

For candidate \(c\):

\[
\hat{y}(c) = f_\theta\bigl(x(c)\bigr), \quad
u(c) = U\bigl(f_\theta, x(c)\bigr)
\]

- \(x(c)\): feature vector (Score1 diagnostics, code/behavior fingerprints, optional short-rollout stats).
- \(y(c)\): label vector from a **labeling budget** (v1 default = Stage II short train+eval metrics).
- \(u(c)\): scalar uncertainty (ensemble std, distance to train set, or predictive variance).

**Decision hook (pipeline later):**

```text
if u(c) high           -> defer / enqueue Active Learning
elif ŷ weak + confident -> drop
elif ŷ strong           -> promote toward Stage II/III
else                    -> keep in normal evolution path
```

---

## 3. Features \(x(c)\) — schema v1

Implement in `features.extract_candidate_features`.

| Key | Type | Source | Notes |
|-----|------|--------|-------|
| `schema_version` | str | const | `"1"` |
| `candidate_id` | str | candidate | logging only; **do not** train on raw id |
| `code_hash` | str | SHA256 of normalized code | collision check / dedup |
| `code_len` | int | len(code) | cheap complexity proxy |
| `score1` | float | Score1Result.score | after gates (−inf → missing / reject flag) |
| `score1_raw` | float? | raw_score | before reject clamp |
| `score1_holdout` | float? | holdout_score | |
| `score1_train` | float? | train_score | |
| `score1_degen` | float | degenerate_fraction | |
| `score1_rejected` | bool | rejected | |
| `score1_reject_reason` | str? | | |
| `score1_worst_k` | list[{sid, rho}] | from scenario_scores | top-3 worst |
| `behavior_fingerprint` | list[float] | reward on fixed smoke states | length = n_smoke |
| `label_budget` | str | e.g. `stage2_short` | which y protocol was used when labeling |

**Rules:**

- All values JSON-serializable.
- Missing optional fields → `null`.
- Rejected Score1 candidates may still be labeled once for negative examples, or skipped (config flag `label_rejected`).

---

## 4. Labels \(y(c)\) — schema v1

Default labeling protocol: **`stage2_short`**.

| Key | Type | Meaning |
|-----|------|---------|
| `SR`, `CR`, `TR`, `NT`, `PL`, `ITR`, `SD` | float | last Stage II proxy metrics |
| `scalar` | float | `SR - CR - 0.5*TR` (engineering target for v1) |
| `env_steps` | int | actual env steps used |
| `k2_unit` | str | `env_steps` \| `gradient_steps` |
| `train_steps_config` | int | configured K2 |
| `eval_episodes` | int | E2 used |
| `wall_seconds` | float | wall time for this label |
| `seed` | int | | 
| `ok` | bool | trainer finished without crash |

Optional later protocol `stage3_short` (subset of candidates): same metric keys + `stage=stage3`.

**Primary regression target for v1:** `scalar`.  
Secondary (multi-output later): `SR`, `CR`, `TR`.

---

## 5. Bootstrap pipeline (automated)

Implement `bootstrap.run_bootstrap` + CLI `scripts/bootstrap_surrogate.py`.

### Steps (in order)

1. **Ensure Stage I dataset** exists at `stage1_dataset_path` (fail with message to run collector; do not silently collect M=100 inside bootstrap unless `--collect-stage1`).
2. **Idempotency:** if `model_dir/metrics.json` exists and `force=False` and manifest fingerprint matches current feature schema → return early.
3. **Build label population** (`n_candidates`):
   - always include seed / D5 reward if valid;
   - fill with LLM Gen0 (`--llm`) or scripted diverse rewards when `--llm seed`;
   - sandbox-validate; skip invalids; dedupe by `code_hash`.
4. **For each candidate:**
   - run Score1 via `make_score1_fn`;
   - `x = extract_candidate_features(...)`;
   - train+eval Stage II short via Domain Pack trainer (`use_stub` for `--fast`);
   - `y = metrics dict`;
   - `dataset_io.append_example`.
5. **Fit** `SurrogateModel` on all rows with train/val split (e.g. 80/20, grouped by `code_hash`).
6. **Save** model + `metrics.json` (val RMSE / MAE on `scalar`, Spearman of ranks) + update dataset `manifest.json`.

### Suggested CLI

```bash
python scripts/bootstrap_surrogate.py \
  --stage1-dataset data/stage1_dataset \
  --n-candidates 60 \
  --stage2-train-steps 8000 \
  --k2-unit env_steps \
  --out data/surrogate_dataset \
  --model-out artifacts/surrogate \
  --llm seed \
  --device cpu
```

`--fast`: stub trainer, `n_candidates<=4`, tiny steps (for tests).

### Resume

- Each example appended immediately to jsonl.
- Skip `example_id` already present in `features.jsonl`.
- `example_id` = `f"{code_hash[:12]}_{label_budget}"`.

---

## 6. On-disk layout

```text
data/surrogate_dataset/
  features.jsonl      # one JSON object per line
  labels.jsonl        # aligned by example_id
  manifest.json       # n, schema_version, budgets, created_at, git_commit?

artifacts/surrogate/
  model.joblib        # or model.pkl
  metrics.json
  feature_columns.json
  config.json         # hyperparams, target_keys
```

Both trees are local artifacts; prefer gitignoring large model blobs if needed later.

---

## 7. Model v1 (keep simple)

- **Library:** `sklearn` pipeline (or LightGBM if already easy to vendor). Prefer sklearn for fewer deps.
- **Model:** `HistGradientBoostingRegressor` or `RandomForestRegressor` on `scalar`.
- **Uncertainty v1:** std across an ensemble of 5 forests / bootstrap bags, **or** absolute residual magnitude from a holdout calibrator. Expose as `SurrogatePrediction.uncertainty`.
- **Do not** deep-learn transformers over code in v1.

API: `SurrogateModel.fit / predict / save / load` in `model.py`.

---

## 8. Pipeline integration (after bootstrap works)

**Do not block tomorrow’s first PR on this.** Second PR:

- `EvoNavRunConfig.surrogate_model_dir: Optional[str]`
- After Stage I (or after each Stage II round): call `predict`; write `surrogate_preds.json` under run output.
- Gate Stage III population: drop bottom fraction by ŷ if `u` below threshold.

Manifest fields: `surrogate_enabled`, `surrogate_metrics`, `n_dropped_by_surrogate`.

---

## 9. Tests (minimum)

| Test | Assert |
|------|--------|
| `test_feature_schema_serializable` | extract on fake candidate → json roundtrip |
| `test_dataset_io_roundtrip` | append + load_table length match |
| `test_bootstrap_fast_stub` | `--fast` writes model + metrics without GPU |
| `test_predict_shape` | loaded model returns finite scalar + uncertainty≥0 |

---

## 10. Acceptance criteria for “done v1”

1. One command bootstrap produces non-empty jsonl + loadable model.  
2. Val Spearman(rank ŷ, rank y) reported in `metrics.json` (even if low).  
3. Stubs replaced; `NotImplementedError` gone on happy path.  
4. No change to paper Stage III claim path unless flag explicitly set.  
5. Entry noted in `docs/WHATS_NEW.md`.

---

## 11. Non-goals (v1)

- Predicting full paper K3=1e7 outcomes accurately.
- Replacing Score1.
- Joint training with Active Learning loop (AL only **consumes** uncertainty later).
- Multi-domain packs beyond CrowdNav.

---

## 12. Suggested implementation order (tomorrow)

1. `dataset_io` + empty manifest writer  
2. `features` (Score1 + smoke fingerprint + code_hash)  
3. `model` sklearn fit/save/load  
4. `bootstrap` with stub Stage II  
5. CLI script + pytest `--fast`  
6. Real Stage II labeling pass on a small `n_candidates`  
7. (Later) pipeline gate + WHATS_NEW entry  

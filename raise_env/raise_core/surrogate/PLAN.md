# Surrogate — Technical Plan (LOCKED v1)

**Status:** implementation in progress (contract locked 2026-09-20).  
**Package:** `crowd_nav.reward_search.surrogate`  
**Data root:** `raise_env/data/surrogate_dataset/`  
**Artifact root:** `raise_env/artifacts/surrogate/`  
**Depends on:** Stage I Score1 (`scoring.py`) + Domain Pack Stage II trainer.  
**Sibling:** Active Learning consumes this API later — **out of scope for v1**.

This document is the **locked** implementation contract for Surrogate v1.  
Do not silently change schema / targets without bumping `FEATURE_SCHEMA_VERSION` / `LABEL_SCHEMA_VERSION` and a `docs/WHATS_NEW.md` entry.

---

## 0. Locked decisions (2026-09-20)

| # | Decision | Locked choice |
|---|----------|---------------|
| L1 | Role | Cheap predictor of **Stage II short** proxy metrics — **not** paper Stage III / K3 claims |
| L2 | Primary train targets | **`SR`, `CR`, `TR`** (multi-output). Not the engineering scalar |
| L3 | `scalar` field | Still **written** into labels as `SR - CR - 0.5*TR` for logging / back-compat; **not** the fit target |
| L4 | Ranking policy at predict time | Derive later from \(\hat{SR},\hat{CR},\hat{TR}\) (lex / thresholds). Do **not** train on LLM R2 rank ids |
| L5 | Model | `sklearn` + `joblib`; ensemble of `RandomForestRegressor` (default 5 bags) |
| L6 | Uncertainty | Mean of per-target ensemble std → `SurrogatePrediction.uncertainty ≥ 0` |
| L7 | Pipeline gate | **Wired (opt-in)** via `--surrogate` / `RaiseRunConfig.surrogate_*` — see `surrogate/gate.py` |
| L8 | Active Learning | **Wired (opt-in)** `--active-learning` after Stage I when surrogate model exists |
| L9 | `--fast` | Stub Stage II + smoke Score1; `n_candidates ≤ 4`; no GPU |
| L10 | Deps | Add `scikit-learn` + `joblib` to pinned requirements |

---

## 1. Goal

Train a **cheap regressor** that maps inexpensive features of a reward candidate to **Stage II short outcomes** so Algorithm 1 / later AMFRS can:

- drop hopeless candidates before long Stage II/III;
- rank promotion priority from predicted metrics;
- expose **uncertainty** that Active Learning will query.

The surrogate is **not** a substitute for paper-claim Stage III numbers. Final tables still come from real PPO / H-sweep.

---

## 2. Problem formulation

For candidate \(c\):

\[
\hat{y}(c) = f_\theta\bigl(x(c)\bigr), \quad
u(c) = U\bigl(f_\theta, x(c)\bigr)
\]

- \(x(c)\): feature vector (Score1 diagnostics, code/behavior fingerprints).
- \(y(c)\): label vector from labeling budget `stage2_short` — full proxy metrics; **fit on `SR,CR,TR`**.
- \(u(c)\): scalar uncertainty (ensemble std).

**Decision hook (pipeline later — not v1):**

```text
if u(c) high           -> defer / enqueue Active Learning
elif ŷ weak + confident -> drop
elif ŷ strong           -> promote toward Stage II/III
else                    -> keep in normal evolution path
```

Weak/strong may use lex on \((\hat{SR},-\hat{CR},-\hat{TR})\) or per-metric thresholds — configurable later.

---

## 3. Features \(x(c)\) — schema v1

Implement in `features.extract_candidate_features`.  
`FEATURE_SCHEMA_VERSION = "1"`.

| Key | Type | Source | Notes |
|-----|------|--------|-------|
| `schema_version` | str | const | `"1"` |
| `candidate_id` | str | candidate | logging only; **do not** train on raw id |
| `code_hash` | str | SHA256 of normalized code | collision check / dedup |
| `code_len` | int | len(code) | cheap complexity proxy |
| `score1` | float | Score1Result.score | after gates (−inf → train as NaN + rejected flag) |
| `score1_raw` | float? | raw_score | before reject clamp |
| `score1_holdout` | float? | holdout_score | |
| `score1_train` | float? | train_score | |
| `score1_degen` | float | degenerate_fraction | |
| `score1_rejected` | bool | rejected | |
| `score1_reject_reason` | str? | | |
| `score1_worst_k` | list[{sid, rho}] | from scenario_scores | top-3 worst |
| `score1_worst_mean` | float? | mean rho of worst_k | **trainable** numeric |
| `behavior_fingerprint` | list[float] | reward on fixed smoke states | length = n_smoke |
| `label_budget` | str | e.g. `stage2_short` | which y protocol was used when labeling |

**Rules:**

- All values JSON-serializable.
- Missing optional fields → `null`.
- Rejected Score1 candidates may still be labeled once for negative examples (`label_rejected=True` default in bootstrap).

**Trainable columns (model vectorizer):**  
`code_len`, `score1` (finite or NaN), `score1_raw`, `score1_holdout`, `score1_train`, `score1_degen`, `score1_rejected` (0/1), `score1_worst_mean`, `behavior_fingerprint[*]`.

---

## 4. Labels \(y(c)\) — schema v1

`LABEL_SCHEMA_VERSION = "1"`.  
Default labeling protocol: **`stage2_short`**.

| Key | Type | Meaning |
|-----|------|---------|
| `example_id` | str | same as features row |
| `SR`, `CR`, `TR`, `NT`, `PL`, `ITR`, `SD` | float | Stage II proxy metrics |
| `scalar` | float | `SR - CR - 0.5*TR` (**log only**; not fit target) |
| `env_steps` | int | actual env steps used (0 for stub) |
| `k2_unit` | str | `env_steps` \| `gradient_steps` |
| `train_steps_config` | int | configured K2 |
| `eval_episodes` | int | E2 used |
| `wall_seconds` | float | wall time for this label |
| `seed` | int | | 
| `ok` | bool | trainer finished without crash |

**Primary regression targets for v1:** `SR`, `CR`, `TR`.  
Stored but not fit: `NT`, `PL`, `ITR`, `SD`, `scalar`.

---

## 5. Bootstrap pipeline (automated)

Implement `bootstrap.run_bootstrap` + CLI `scripts/bootstrap_surrogate.py`.

### Steps (in order)

1. **Ensure Stage I dataset** exists at `stage1_dataset_path` when not `--fast` / smoke (fail with message to run collector; do not silently collect M=100 unless `--collect-stage1` — optional, not required in v1).
2. **Idempotency:** if `model_dir/metrics.json` exists and `force=False` and manifest fingerprint matches current feature schema → return early.
3. **Build label population** (`n_candidates`):
   - always include seed / D5 reward if valid;
   - fill with LLM Gen0 (`--llm`) or `SeedVariantLLMClient` when `--llm seed`;
   - sandbox-validate; skip invalids; dedupe by `code_hash`.
4. **For each candidate:**
   - run Score1 via domain `make_score_fn` (smoke under `--fast`);
   - `x = extract_candidate_features(...)`;
   - train+eval Stage II short via Domain Pack trainer (`use_stub` for `--fast`);
   - `y = metrics dict` (+ `scalar`);
   - `dataset_io.append_example`.
5. **Fit** `SurrogateModel` on all rows with train/val split (e.g. 80/20, grouped by `code_hash` when possible).
6. **Save** model + `metrics.json` (per-target val RMSE / MAE + Spearman of ranks for each of SR/CR/TR) + update dataset `manifest.json`.

### Suggested CLI

```bash
python scripts/bootstrap_surrogate.py \
  --stage1-dataset data/stage1_dataset \
  --n-candidates 60 \
  --stage2-train-steps 8000 \
  --k2-unit gradient_steps \
  --out data/surrogate_dataset \
  --model-out artifacts/surrogate \
  --llm seed \
  --device cpu
```

`--fast`: stub trainer, smoke Score1, `n_candidates<=4`, tiny steps (for tests).

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
  manifest.json       # n, schema versions, budgets, created_at

artifacts/surrogate/
  model.joblib
  metrics.json
  feature_columns.json
  config.json         # hyperparams, target_keys=["SR","CR","TR"]
```

Gitignore large `*.joblib` under `artifacts/surrogate/` if needed; keep `metrics.json` / `config.json` optional.

---

## 7. Model v1 (locked)

- **Library:** `sklearn` + `joblib`.
- **Model:** ensemble of `n_estimators_bags` (default **5**) `RandomForestRegressor` wrappers; each bag predicts all three targets (one multi-output RF or three single-output RFs — implementation may use `MultiOutputRegressor`).
- **Targets:** `("SR", "CR", "TR")`.
- **Uncertainty:** mean of per-target std across bags.
- **Do not** deep-learn transformers over code in v1.

API: `SurrogateModel.fit / predict / save / load` in `model.py`.  
`SurrogatePrediction.y_hat` must include at least `SR`, `CR`, `TR` (floats).

---

## 8. Pipeline integration (after bootstrap works)

**Out of scope for v1 first PR.** Second PR:

- `RaiseRunConfig.surrogate_model_dir: Optional[str]`
- After Stage I (or after each Stage II round): call `predict`; write `surrogate_preds.json`.
- Gate Stage III population: drop bottom fraction by predicted quality if `u` below threshold.

Manifest fields: `surrogate_enabled`, `surrogate_metrics`, `n_dropped_by_surrogate`.

---

## 9. Tests (minimum)

| Test | Assert |
|------|--------|
| `test_feature_schema_serializable` | extract on fake candidate → json roundtrip |
| `test_dataset_io_roundtrip` | append + load_table length match |
| `test_bootstrap_fast_stub` | `--fast` / `run_bootstrap(use_stub=True)` writes model + metrics without GPU |
| `test_predict_shape` | loaded model returns finite `SR/CR/TR` + `uncertainty≥0` |

---

## 10. Acceptance criteria for “done v1”

1. One command bootstrap (`--fast` or real) produces non-empty jsonl + loadable model.  
2. Val Spearman(rank \(\hat{y}_k\), rank \(y_k\)) reported in `metrics.json` for each of SR/CR/TR (even if low).  
3. Stubs replaced; `NotImplementedError` gone on happy path.  
4. No change to paper Stage III claim path.  
5. Entry noted in `docs/WHATS_NEW.md`.

---

## 11. Non-goals (v1)

- Predicting full paper K3=1e7 outcomes accurately.
- Replacing Score1.
- Training on LLM R2/R3 rank positions.
- Changing global `selection.navigation_scalar` / pipeline `final_rank`.
- Joint training with Active Learning loop.
- Multi-domain packs beyond CrowdNav.
- Wiring surrogate gates into `RaisePipeline`.

---

## 12. Implementation order (this pass)

1. Lock this document ← **done**  
2. `dataset_io` + empty manifest writer  
3. `features` (Score1 + smoke fingerprint + code_hash)  
4. `model` sklearn multi-output SR/CR/TR + uncertainty  
5. `bootstrap` with stub Stage II + CLI `--fast`  
6. pytest suite §9  
7. `docs/WHATS_NEW.md` entry  
8. (Later) real Stage II labeling on small `n_candidates`  
9. (Later) pipeline gate + AL  

# Active Learning — Technical Plan (LOCKED v1)

**Status:** implementation in progress (contract locked 2026-09-20).  
**Package:** `crowd_nav.reward_search.active_learning`  
**Queue / logs:** `evonav_env/data/active_learning/`  
**Hard dependency:** Surrogate v1 (`artifacts/surrogate` + uncertainty + `SR/CR/TR` targets).  
**Soft dependency:** Stage I collector; surrogate `dataset_io.append_example`.

Depends on Surrogate v1 being green. Pipeline flag `--active-learning` is **out of scope** for this pass.

---

## 0. Locked decisions (2026-09-20)

| # | Decision | Locked choice |
|---|----------|---------------|
| A1 | Surrogate targets | Disagreement / borderline use \(\hat{SR},\hat{CR},\hat{TR}\) (quality proxy = `SR−CR−0.5·TR` on \(\hat{y}\) only for ranking heuristics) |
| A2 | v1 query kinds | **`stage2_label` fully**; **`stage1_scenario`** = safe side-pool **request** under `stage1_extra/` (no mutation of paper Stage I npz; full collect optional later) |
| A3 | No model on disk | Loop returns `status=error` / CLI exit **2** with `Surrogate model required` |
| A4 | Weights default | `u=0.45`, `disagree=0.35`, `borderline=0.15`, `diversity=0.15` |
| A5 | Refit | After `refit_every` completed `stage2_label`s (or `--force-refit`), re-fit on full `surrogate_dataset` |
| A6 | Pipeline | Opt-in via `EvoNavPipeline` + `--active-learning` (requires `--surrogate`) |

---

## 1. Goal

Whenever Score1, surrogate, and/or short Stage II **disagree** or the surrogate is **uncertain**, acquire the most informative new data:

1. new Stage I scenario **requests** (cheap side pool), and/or  
2. new surrogate labels via short Stage II (medium cost),  

then append to the right dataset and periodically **re-fit** the surrogate.

---

## 2. Relationship to Surrogate

```text
Bootstrap (once)          Active Learning (recurring)
─────────────────         ───────────────────────────
build many labels   →     query only uncertain / disagreeing items
fit model           →     re-fit when n_new ≥ refit_every
save artifacts/     →     update same artifacts/surrogate
```

**Rule:** If no surrogate model is on disk, AL must no-op / exit 2 (`Surrogate model required`).

---

## 3. Query kinds

| `kind` | What we acquire | Writes to | Cost |
|--------|-----------------|-----------|------|
| `stage1_scenario` | request for scenario_ids (side pool) | `data/active_learning/stage1_extra/` | low |
| `stage2_label` | Score1 features + Stage II short metrics | `data/surrogate_dataset/*.jsonl` | medium |
| `stage3_label` | (v2+) | — | high |

---

## 4. Scoring (`query.score_queries`)

```text
priority = w_u * u_norm + w_d * d_1y + w_b * borderline - w_div * redundancy
```

- \(u\): surrogate uncertainty (batch min-max → [0,1])  
- \(d_{1\hat{y}}\): |rank_pct(score1) − rank_pct(quality(ŷ))|  
- borderline: closeness of quality(ŷ) to `promote_threshold`  
- redundancy: similarity of `behavior_fingerprint` to recently queued/labeled  

Default prefer `stage2_label`. Emit `stage1_scenario` when many uncertain candidates share weak Score1 `sid`s (from features / score1 worst_k).

---

## 5. Loop

1. Load surrogate (fail closed if missing).  
2. Load candidate batch (JSON path or in-memory).  
3. predict → score_queries → enqueue.  
4. dequeue_batch(limit=max_queries).  
5. execute_query → mark_done.  
6. Refit if needed.  
7. Append `steps.jsonl`.

---

## 6–13

See original sections for on-disk layout, tests, non-goals, and config knobs.  
**Acceptance:** PLAN §9–10. Pipeline integration remains later.

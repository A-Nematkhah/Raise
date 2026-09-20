# Closed-Loop Multi-Fidelity Evolution — Technical Plan (LOCK DRAFT)

**Status:** implementation in progress (innovation path — **not** paper Alg.1 linear).  
**Package (new):** `crowd_nav.reward_search.closed_loop`  
**Opt-in CLI:** `--closed-loop` on `run_evonav.py`  
**Default without flag:** keep current linear Stage I → II → III (unchanged).  
**Depends on:** Score1, StageIEvolver ops, Stage II short trainer, Surrogate v1 (`dataset_io` / `model` / `gate`), **Active Learning v1** (`query.score_queries` / `acquire` / queue).

This is the **innovation loop**: Gen0 → Score1 → Stage II labels → fit Surrogate → later gens:
**Score1 → Surrogate gate → AL (who else to label) → Stage II on selected set → refit**.
AL is **inside** the closed loop, not a separate afterthought.

---

## 0. Locked decisions

| # | Decision | Locked choice |
|---|----------|---------------|
| C1 | Paper fidelity | **Not required** for this mode. Manifest must set `mode=closed_loop_v1` and document deviation |
| C2 | Outer unit | **Generation epoch** \(g = 0..G-1\): propose → Score1 → (gate?) → Stage II short → append labels → refit |
| C3 | Gen0 | **No Surrogate gate.** Full pop → Score1 → Stage II short (all members) → first fit |
| C4 | Gen ≥ 1 | Propose → Score1 → **Surrogate gate** → **AL select** → Stage II on (survivors ∪ AL picks) |
| C5 | Stage II depth per epoch | **1 train/eval round per candidate** (no full G2=16 inside each gen). D.3 refine: **off by default** in loop; opt-in `--closed-loop-refine` |
| C6 | Label source | Same as surrogate bootstrap / AL `stage2_label`: `label_and_append_candidate` → `surrogate_dataset` |
| C7 | Refit rule | After Gen0 always; then every epoch if `n_new ≥ refit_every` (default **1** epoch or **≥8** new labels — whichever first) |
| C8 | Gate policy | Reuse `surrogate.gate.gate_population` (confident-weak drop). Until `n_labeled ≥ min_labels_for_gate` (default **24**), gate is **soft** (log only / drop_fraction=0) |
| C9 | Score1 | **Always on** before gate (cheap). Rejected Score1 can still be labeled optionally (`label_rejected=True` default for negative examples) |
| C10 | Population sizes | Default N=8, G=10 (same knobs as Stage I). Stage II K2 can be shorter in loop: default **4000** gradient steps (config `closed_loop_k2`) |
| C11 | What happens to “classic” Stage II/III | After loop finishes: optional **final Stage II polish** (few rounds on elite) + Stage III on gated elite — flags `--closed-loop-final-stage2` / normal Stage III |
| C12 | **Active Learning (in-loop)** | **On by default** when Surrogate is ready (`n_labeled ≥ min_labels_for_gate`). Each epoch after gate: `score_queries` on population (esp. dropped / uncertain) → acquire up to `al_max_per_epoch` extra `stage2_label`s. Optional `stage1_scenario` requests go to `stage1_extra/` (side pool; does not mutate paper npz). Disable with `--closed-loop-no-al` |
| C13 | `--fast` | Stub Stage II + smoke Score1; G≤2, N≤4, min_labels_for_gate=4 |
| C14 | Artifacts | `output_dir/closed_loop/` : `epochs.jsonl`, `al_steps.jsonl`, gate reports, surrogate refit log |

---

## 1. Goal (one sentence)

**Close the loop with Surrogate + AL:** Stage II outcomes train the Surrogate; the Surrogate gates who is “clearly weak”; AL picks the **informative** extras to label; together they decide the Stage II budget each generation.

---

## 2. Loop diagram

```text
                    ┌──────────────────────────────────────────────┐
                    │                                              │
                    ▼                                              │
              propose pop[g]                                       │
                    │                                              │
              Score1 (train/holdout gates)                         │
                    │                                              │
         g==0 or n_lab < L_min ?                                   │
              │ yes              │ no                              │
              │                  ▼                                 │
              │           Surrogate.predict + gate                 │
              │           survivors = keep; dropped = rest         │
              │                  │                                 │
              │           AL.score_queries(pop, preds)             │
              │           → up to al_max_per_epoch stage2_label    │
              │             (prefer uncertain / disagree / border) │
              │           optional stage1_scenario → stage1_extra/ │
              │                  │                                 │
              └──────────► to_label = unique(survivors ∪ AL picks) │
                    │         (+ ensure ≥ min_stage2_per_gen)        │
              Stage II short (1× train/eval each)                    │
                    │                                              │
              append → surrogate_dataset (+ AL queue done)         │
                    │                                              │
              refit Surrogate → artifacts/surrogate                │
                    │                                              │
              rank / reflect / build next proposers ───────────────┘
                    │
              after G epochs → optional final Stage II/III
```

**Roles (do not collapse):**

| Piece | Job |
|-------|-----|
| Score1 | Cheap analytic filter + evolution ranking |
| Surrogate gate | Drop **confident-weak** (save Stage II) |
| AL | Spend Stage II on **uncertain / disagree / borderline** (teach Surrogate) |
| Stage II short | Ground-truth labels for both gate learning and search |
---

## 3. Module layout

```text
crowd_nav/reward_search/closed_loop/
  PLAN.md          ← this file
  __init__.py
  config.py        ← ClosedLoopConfig dataclass
  epoch.py         ← run_epoch(...), propose helpers
  runner.py        ← ClosedLoopRunner.run() → final population + artifacts
  logging_io.py    ← epochs.jsonl helpers
```

**Reuse (do not fork):**

- `StageIEvolver.initialize_population` / `_next_generation` / `score_population`
- `label_and_append_candidate`, `SurrogateModel.fit/save/load`
- `gate_population`, `predict_population`
- Domain pack Stage II trainer

---

## 4. Config knobs (`ClosedLoopConfig` + CLI)

| Knob | Default | Meaning |
|------|---------|---------|
| `enabled` | False | `--closed-loop` |
| `population_size` | 8 | N |
| `generations` | 10 | G epochs |
| `min_labels_for_gate` | 24 | soft gate until then |
| `refit_every_new_labels` | 8 | min new labels between forced refits |
| `stage2_train_steps` | 4000 | K2 inside loop |
| `k2_unit` | gradient_steps | same as Stage2Config |
| `drop_fraction` | 0.25 | gate |
| `max_uncertainty_to_drop` | 0.15 | gate |
| `min_keep` | 2 | gate |
| `min_stage2_per_gen` | 4 | floor on Stage II slots / epoch |
| `al_enabled` | True | in-loop AL (C12) |
| `al_max_per_epoch` | 4 | max extra AL `stage2_label`s beyond survivors |
| `al_allow_stage1_requests` | True | emit `stage1_scenario` side-pool requests |
| `enable_refine` | False | D.3 inside loop |
| `final_stage2_rounds` | 0 | polish after loop (0 = skip) |
| `surrogate_model_dir` | artifacts/surrogate | shared with bootstrap |
| `surrogate_dataset` | data/surrogate_dataset | append-only jsonl |
| `al_root` | data/active_learning | queue / steps under closed-loop run or shared |

---

## 5. Epoch algorithm (pseudocode)

```python
def run_closed_loop(cfg, llm, score_fn, trainer, validator):
    evolver = StageIEvolver(llm, score_fn=score_fn, validator=validator, config=s1_cfg)
    pop = evolver.initialize_population()          # Gen0 propose
    known_ids = existing_example_ids(dataset)
    model = load_or_none(cfg.surrogate_model_dir)

    for g in range(cfg.generations):
        pop = evolver.score_population(pop)        # Score1
        evolver.update_reflection(pop)

        to_label = list(pop)
        gate_report = None
        al_report = None
        if g > 0 and model is not None and n_labeled >= cfg.min_labels_for_gate:
            preds = predict(pop, model, score_fn)
            survivors, gate_report = gate_population(pop, preds, ...)
            to_label = list(survivors)
            if cfg.al_enabled:
                queries = score_queries(pop, predictions=preds, ...)
                al_picks = take_stage2_labels(queries, limit=cfg.al_max_per_epoch)
                # also optional stage1_scenario → stage1_extra/
                to_label = unique_by_code(to_label + al_picks)
                al_report = {"n_al_stage2": len(al_picks), ...}
            if len(to_label) < cfg.min_stage2_per_gen:
                to_label = top_up_uncertain(pop, preds, to_label, cfg.min_stage2_per_gen)

        for c in to_label:
            label_and_append_candidate(...)  # same acquire path as AL stage2_label

        if should_refit(...):
            model = fit_and_save(dataset, cfg.surrogate_model_dir)

        log_epoch(g, pop, to_label, gate_report, al_report, n_labeled)

        if g + 1 < cfg.generations:
            pop = evolver._next_generation(ranked_pop(pop))

    return pop, model, history
```

**Ranking for next gen (v1):** Score1 order for proposers. Surrogate+AL only decide **who pays Stage II**.
---

## 6. Interaction with existing pipeline

```text
run_evonav:
  if not --closed-loop:
      linear Alg.1 path (current)
  else:
      ClosedLoopRunner → stage2_pop_out
      if final_stage2_rounds > 0: Stage2Runner on elite
      Stage III on result (reuse surrogate Stage III gate)
```

Bootstrap CLI remains for **warm-start** model before loop (`--surrogate DIR` if `model.joblib` exists → Gen0 can still skip gate until `min_labels`, but Gen1+ may gate earlier if warm-start already ≥ L_min labeled in dataset).

---

## 7. Failure modes & mitigations

| Risk | Mitigation |
|------|------------|
| Gate kills diversity | `min_keep`, AL picks, `min_stage2_per_gen` |
| AL doubles Stage II cost | cap `al_max_per_epoch` (default 4); gate still drops confident-weak |
| Surrogate overfits early | soft gate until 24 labels; Score1 still ranks evolution |
| Cost explosion | 1 Stage II pass/cand labeled; K2=4k; no refine by default |
| Distribution shift | append all labeled codes; periodic full refit on jsonl |
| Windows RAM | `num_processes=1` default in closed-loop Stage2Config |

---

## 8. Tests (must pass before merge)

| Test | Asserts |
|------|---------|
| `test_closed_loop_fast_two_epochs` | `--fast`: Gen0 labels → model exists → Gen1 gate+AL path |
| `test_gen0_never_gates` | with model on disk, g=0 still labels full pop |
| `test_soft_gate_until_min_labels` | n_lab < 24 → no drops |
| `test_al_inside_epoch` | Gen1 with ready model: `score_queries` called; AL picks ⊆ to_label |
| `test_refit_increments_manifest` | refit writes metrics.json |
| `test_linear_path_unchanged` | without `--closed-loop`, pipeline tests still green |

---

## 9. Implementation order

1. Package skeleton + `ClosedLoopConfig` + PLAN lock note in `WHATS_NEW.md`
2. `ClosedLoopRunner` Gen0+1 with stubs (`--fast`)
3. Wire `--closed-loop` in `run_evonav` / `EvoNavRunConfig`
4. Soft/hard gate + uncertain fill
5. Optional final Stage II + Stage III handoff
6. Smoke on real stub GPU-less; then short real cuda nproc=1

---

## 10. Non-goals (v1)

- Replacing Score1 with Surrogate for mutation parent choice  
- Interleaving Stage III inside the loop  
- Mutating paper Stage I npz as primary signal (AL `stage1_extra` remains optional later)  
- Claiming Table 3–6 paper numbers from closed-loop runs  

---

## 11. Acceptance

- [ ] `--closed-loop --fast` completes ≥2 epochs, writes `closed_loop/epochs.jsonl` + surrogate artifacts  
- [ ] Gen0 full Stage II labels; Gen1+ uses gate when labels ≥ threshold  
- [ ] Without flag, previous pipeline behavior unchanged  
- [ ] `docs/WHATS_NEW.md` entry describing innovation loop  

**Next step after lock:** implement §9.1–9.3.

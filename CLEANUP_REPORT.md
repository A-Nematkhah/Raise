# Deep cleanup — Phase 1 report (analysis only, nothing deleted)

Date: 2026-10-01 · Branch analysed: `simplify/highway` (HEAD `7ab229a` + uncommitted script cleanup, see §0)

Confidence legend: **high** = zero references anywhere in the repo (code, tests, docs, CI, configs, string lookups) and removal cannot change behaviour · **medium** = only referenced by tests/docs, or needs a small companion edit · **low** = probably dead but has an external/historical role.

Scope rule in force: **CrowdNav is frozen** (`.cursor/rules/highway-focus-crowdnav-frozen.mdc`). Every item under `raise_env/domains/crowdnav/`, CrowdNav-only scripts, root CrowdNav entry points (`train.py`, `test.py`, `arguments.py`, `plot.py`) and the vendored `baselines_openai/` is **report-only** and listed in "Needs my decision".

---

## 0. Stack, entry points, baseline

| Item | Value |
|------|-------|
| Language / runtime | Python 3.10.11 (system interpreter; no venv in repo) |
| Package manager | pip, no lockfile. `raise_env/requirements_pinned.txt` (CI), `requirements_highway.txt` (highway extras), `requirements.txt` (alias → pinned) |
| Build | none (pure Python); vendored `baselines_openai` installable via `setup.py` |
| Tests | pytest, `raise_env/pytest.ini` + `raise_env/conftest.py`; marker `slow` |
| CI | `.github/workflows/raise-ci.yml`: `pip install -r requirements_pinned.txt` + torch 1.12.1 cpu → `cd raise_env && python -m pytest -m "not slow" -q` |
| Lint | none configured; used ruff 0.16.9 + vulture 2.16 from a throw-away venv (not added to repo) |
| Highway entry points | `scripts/run_raise.py --domain highway`, `scripts/run_raise_highway.py` (budget = `presets.CLOSED_LOOP_PROFILES["highway"]`), `collect_highway_stage1_dataset.py`, `bootstrap_surrogate.py`, `eval_raise_checkpoint.py`, `plot_raise_run.py`, `print_raise_report.py`, `label_reliability_highway.py`, `analyze_label_reliability.py` |
| CrowdNav entry points (frozen) | `run_raise_1h/12h/18h(.ps1)`, `run_raise_paper_scale.py`, `run_p0_*`, `train_ds_rnn.py`, `report.py`, `visualize_raise.py`, `collect_stage1_dataset.py`, `run_stage2_smoke.py`, `run_stage3_smoke.py`, `run_active_learning_step.py`, root `train.py` / `test.py` |

**Pre-cleanup baseline (recorded 2026-10-01, working tree incl. §0 uncommitted changes):**

- `python -m pytest -m "not slow" -q` (CI command) → **360 passed, 1 skipped, 2 deselected**, 3 warnings, 234 s.
- ruff (F401/F841/F821/ERA001, excl. `results/`): core+highway+root 13 F401 · 1 F841 · 2 F821 · 3 ERA001; scripts 4 F401 · 3 F841 · 1 ERA001; tests 10 F401 · 2 F841; crowdnav 27 F401 · 3 F841 · 10 F821 · 1 F811 · 116 ERA001; baselines 19 ERA001.

**Pending uncommitted work on `simplify/highway`** (from the previous "delete duration-based run scripts" request; tests above already include it): 23 scripts `git rm`'d, `run_raise_highway_4h.py → run_raise_highway.py`, `highway_4h` profile renamed to `highway`, `highway_7h` removed, 8 docs updated. → see decision **D1**.

Method: AST import graph from every module (`from x import y`, relative imports, `importlib.import_module` strings) + whole-repo `git grep` for every candidate name (catches string/monkeypatch/doc refs) + ruff/vulture cross-checked by hand. Every vulture hit was opened and classified.

---

## a. Unused files

All `raise_core/` and `domains/highway/` modules are imported by runtime code (strict dotted-path import check: 0 unused).

| Path | Evidence | Confidence |
|------|----------|------------|
| `raise_env/artifacts/surrogate_fix_fix/` (`config.json`, `feature_columns.json`, `metrics.json`, `model.joblib` 225 KB) | `git grep surrogate_fix_fix` → 0 hits. Tests write surrogate models to tmp dirs. `.gitignore` whitelists a *different* name (`artifacts/surrogate_test_fix/`), so this dir looks committed by accident in `066e839`. Targets are SR/CR/TR. | high |
| `raise_env/domains/crowdnav/runtime/crowd_nav/reward_search/` (legacy shim, ~25 modules) + `crowd_nav/domains/` (3 `__init__.py`) | Strict import check: none imported by runtime; only `reward_search/prompts.py` is imported by 1 CrowdNav test. `README_LEGACY.md` documents it as a frozen compat shim. | medium — **CrowdNav frozen → D3** |
| `baselines_openai/` — 21 of 47 modules unreachable from any import: `common/{cg,cmd_util,distributions,input,models,mpi_adam,mpi_adam_optimizer,mpi_fork,mpi_moments,mpi_running_mean_std,plot_util,policies,retro_wrappers,runners,schedules,segment_tree}.py`, `common/vec_env/vec_video_recorder.py`, upstream tests `bench/test_monitor.py`, `common/test_mpi_util.py`, `common/vec_env/test_vec_env.py`, `common/vec_env/test_video_recorder.py` | Transitive closure from the 6 `baselines.*` imports used by CrowdNav runtime. Not collected by CI (pytest runs inside `raise_env/`). | medium — vendored third-party + CrowdNav-only → **D4** |
| Root CrowdNav scripts with ≤1 reference: `scripts/run_p0_eval_suite.py`, `run_p0_gst_baseline.py` (only `scripts/README.md`), `train_ds_rnn.py`, `visualize_raise.py`, `run_raise_18h.py` | Referenced only by `scripts/README.md` / docs; they are CLI entry points (not imports). | low — **CrowdNav frozen → D3** |

Kept on purpose (are entry points or referenced): every highway script listed in §0, `_prereqs.py` (imported by run scripts), `configs/paper_scale.yaml` (loaded by `presets.load_paper_scale_yaml`, tested), `groq_keys.json.example`, all `.gitkeep`, `run/*.sh` in `gst_updated` (referenced by `_prereqs.py` + README).

## b. Unused functions / classes / constants

Each item: 0 references outside its own definition (whole-repo grep, incl. tests/docs/strings).

| Path:line | Symbol | Confidence |
|-----------|--------|------------|
| `raise_env/raise_paths.py:21-31` | `CROWDNAV_DATA`, `HIGHWAY_DATA`, `CROWDNAV_STAGE1_DATASET`, `CROWDNAV_ACTIVE_LEARNING`, `CROWDNAV_SURROGATE_DATASET`, `HIGHWAY_STAGE1_DATASET`, `HIGHWAY_ACTIVE_LEARNING`, `HIGHWAY_SURROGATE_DATASET`, `GST_RUNTIME_PREFIX` (last importer was `scripts/_bootstrap.py`, deleted in §0). Module itself stays (imported for its `sys.path` side effect). | high (only after D1 lands) |
| `raise_env/domains/highway/env_wrapper.py:22` | `DEFAULT_TIME_STEP` | high |
| `raise_env/domains/highway/pareto_rank.py:45` | `LEGACY_CALIBRATION_MODE` | high |
| `raise_env/raise_core/presets.py:33` | `PAPER_DEFAULT_N_SEEDS` | high |
| `raise_env/raise_core/active_learning/loop.py:22` | class `SurrogateModelRequired` | high |
| `raise_env/raise_core/active_learning/queue.py:22` | `default_queue_root()` | high |
| `raise_env/raise_core/console.py:30` | `is_verbose()` | high |
| `raise_env/raise_core/surrogate/dataset_io.py:22` | `default_dataset_root()` | high |
| `raise_env/raise_core/surrogate/model.py:32` | `default_model_dir()` (its only reference is an unused import in `bootstrap.py:33`, see §d) | high |
| `raise_env/raise_core/surrogate/targets.py:98` | `normalize_labels()` | high |
| `raise_env/raise_core/explore.py:1121` | `seed_reward_code()` | high |
| `raise_env/raise_core/explore.py:311` | method `_llm_code()` | high |
| `raise_env/raise_core/stage3_checkpoint.py:41` | `stage3_checkpoint_dir()` | high |
| `raise_env/raise_core/scoring.py:51` | class `Score1Fn` | high |
| `raise_env/raise_core/scoring.py:438` | `make_constant_score_fn()` | high |
| `raise_env/raise_core/checkpointing.py:200` | method `load_stage_done()` | high |
| `raise_env/raise_core/rules.py:119` | `pairwise_rule_scores()` | high |
| `raise_env/raise_core/pipeline.py:277` | method `_best()` | high |
| `raise_env/raise_core/raise_loop/report.py:217` | `log_closed_loop_report()` (`print_raise_report.py` uses `build_/write_closed_loop_report`, not this) | high |
| `raise_env/raise_core/raise_loop/proxy_feedback.py:32` | `_is_highway_metrics()` — only mentioned in `VERIFICATION_REPORT.md` | high |
| `raise_env/domains/highway/pareto_rank.py:153` | `calibrate_from_reference_rollout()` + `rank_population(reference_speed_samples=, reference_cr=, reference_tr=)` branch (L471-493) — only caller was `collect_reference_rollout_stats`, removed in `e1e80bd`; now used only by 2 tests in `test_pareto_rank.py:104-125` | medium (removal deletes 2 tests) |
| `raise_env/raise_core/active_learning/queue.py:103` | `pending_items()` — used only by tests | medium |
| `raise_env/domains/highway/diagnostics_callback.py:61` | `format_component_trend_lines()` — used only by tests | medium |
| `raise_env/domains/highway/overtake.py:20,42` | attribute `passed_by` — written by tracker, read only by a test (the per-km metric was dropped in `e688e42`) | medium |

**Verified false positives (keep):** SB3 hooks `_on_step` / `_on_rollout_end` / `_on_training_end` (`adapter.py:416,424`, `diagnostics_callback.py:169,191,279,283`); `conftest.pytest_configure`; dataclass/pack fields `spec_path`, `description`, `selection_scalar_name`, `final_stage2_rounds`, `checkpointing.timestamp_utc` (serialized via `asdict`), `RaiseResult.stage2_population/stage3_population`; `DomainPack.d1/d3/d5_*` properties, `scoring.make_score1_fn`, `prompts.PROMPT_MEMORY_EXAMPLE` (used by CrowdNav tests/pack); `llm.generate`, `checkpointing.has`, `features.feature_schema_version` (protocol/API); `refine.py:600 human_num_arg` (callback signature); `pareto_rank.py:269 faster_speeds` (function parameter, positional call site at L704 — removing changes a signature → refactor, out of scope); CrowdNav attributes (`thisSeed`, `use_self_attn`, `benchmark`, `pause`, `clip_action`, …).

## c. Dead code

| Path:line | Finding | Confidence |
|-----------|---------|------------|
| `raise_env/raise_core/pipeline.py:460` | **Latent bug, not dead code:** `env_name_for_predict_method(...)` is called but never imported in this function (only imported locally at L523 in another method) → `NameError` whenever `closed_loop_final_stage2_rounds > 0` (default 0, so untested). Cleanup must not change behaviour → **D6**. | high (bug) |
| `raise_env/raise_core/validate.py:499, 562` | `last_ckpt_path` assigned twice, never read (pure `os.path.join` / alias, no side effects) | high |
| ERA001 `validate.py:64`, `scripts/report.py:453` | ruff "commented-out code" — actually prose/doc comments | false positive, keep |
| ERA001 `raise_env/test.py:119-120` | real commented-out code in root CrowdNav `test.py` | high — CrowdNav frozen → D3 |
| ERA001 116× in `domains/crowdnav/`, 19× in `baselines_openai/` | commented-out code in frozen/vendored trees | report only |
| Debug leftovers | `breakpoint()` / `pdb` / `ipdb`: 0. Bare `print(` in `raise_core` + `domains/highway` library code: 0 (logging goes through `raise_core.console`). `TODO/FIXME/XXX/HACK` in core/highway/scripts: 0. | — |

## d. Unused imports and variables (ruff F401/F841, each verified: not a re-export, not a monkeypatch target, no `__all__`)

**Highway / core (high):**
`conftest.py:8` pytest · `domains/highway/action_config.py:14` Sequence · `domains/highway/final_select.py:8` Optional · `raise_core/active_learning/queue.py:12` Optional · `raise_core/explore.py:45` RewardSandboxError · `raise_core/pipeline.py:34` RewardValidator · `raise_core/presets.py:13` Mapping · `raise_core/proxy_consistency.py:12` Iterable · `raise_core/raise_loop/parallel_label.py:8` Optional · `raise_core/raise_loop/runner.py:170` RewardValidator (function-local import) · `raise_core/surrogate/bootstrap.py:29` code_sha256, `:33` default_model_dir · `raise_core/surrogate/gate.py:15` SurrogatePrediction · `raise_core/validate.py` `last_ckpt_path` (see §c).

**Tests (high):**
`domains/highway/tests/test_calibration_modes.py:12` ReferenceStats · `test_eureka_components.py:5` SimpleNamespace, `:14` D2_MUTATION_PROMPT, `:24` format_reward_component_trends · `test_pack_loads.py:7` pytest · `test_pareto_bugfixes.py:17` survival_only_reference, `:61-62` unused locals `a`, `b` · `test_post_run_plots.py:11` os · `raise_core/tests/test_active_learning.py:5` math · `test_pipeline.py:6` os · `test_raise_loop.py:6` patch.

**Scripts:** `_prereqs.py:12` Optional (shared, high) · CrowdNav-only scripts `collect_stage1_dataset.py:37`, `report.py:51`, `run_p0_eval_suite.py:31`, `report.py:175 n_t / :457 name / :608 tr` → frozen, report only.

**Not a real issue:** `surrogate/bootstrap.py:60` F821 `"StageIConfig"` is a string-only type annotation.

**CrowdNav (report only):** 27 F401, 3 F841, 10 F821, 1 F811.

## e. Dependencies (requirements vs. real top-level imports)

| Package | Declared in | Used by | Verdict |
|---------|-------------|---------|---------|
| `Cython==0.29.36` | pinned | no import | low — build-time requirement of Python-RVO2 (`rvo2`, CrowdNav ORCA) → keep / **D5** |
| `protobuf==3.19.6` | pinned | no import | low — compat pin for `tensorflow==2.11` → keep while TF stays |
| `tensorflow==2.11.0` | pinned | only `baselines_openai` (12 files, mostly unreachable modules: `tf_util`, `policies`, `models`…) | medium — CrowdNav/vendored; heavy CI install → **D5** |
| `requirements.txt` | — | one-line alias `-r requirements_pinned.txt`; not referenced by CI or code; only a comment in `requirements_pinned.txt` | medium → **D5** |
| `openai` | **not declared** | imported in `raise_core` (1 file, guarded) | missing declaration (report) |
| `gymnasium`, `highway-env`, `stable-baselines3`, `Pillow` | `requirements_highway.txt` | highway | **CI does not install `requirements_highway.txt`** → highway tests rely on importorskip/stubs in CI (report, see §j) |
| numpy, pandas, gym, torch, matplotlib, scipy, cloudpickle, tqdm, groq, scikit-learn, joblib, pytest | pinned | used | keep |

Also report-only: `requirements_pinned.txt` pins `torch==2.11.0+cu128` but CI installs `torch==1.12.1` cpu — inconsistent, not a cleanup item. Undeclared imports inside vendored baselines (`mpi4py`, `cv2`, `retro`, `statsmodels`, `MPI`) are in unreachable/optional modules.

## f. Duplicates

- Identical tracked files (LF-normalised SHA-1, ≥50 B): **none**.
- Duplicated function bodies (AST-identical, ≥8 lines):
  - CrowdNav runtime copies of baselines vec-env code: `domains/crowdnav/runtime/rl/networks/{dummy_vec_env,shmem_vec_env}.py`, `rl/vec_env/vec_env.py` vs `baselines_openai/baselines/common/vec_env/*` (12 functions) — frozen, report only.
  - `crowd_sim*.py` `calcFOVLineEndPoint` ×4, `selfAttn_srnn_temp_node.create_attn_mask` ×2 — frozen.
  - Core: `raise_loop/checkpoint.py:30` vs `stage3_checkpoint.py:55` `_serialize_pop` (10 L); `raise_loop/gate_policy.py:12` vs `surrogate/targets.py:34` `_finite` (8 L); `refine.py:199` / `validate.py:196,209` stub `train_and_eval` bodies. Merging = refactor → **D7** (default: leave).
  - `domains/crowdnav/prompts.py` vs `domains/highway/prompts.py` `format_d2_crossover` / `format_d3_refinement`; `adapter.py __init__` / `evaluate_at_human_counts` — per-domain packs by design; touching CrowdNav forbidden → keep.
  - Test helpers `_invalid_import_code` ×3, `_make_population` ×3, `_cand` ×2, CrowdNav `_state`/`_traj` — could move to conftest; refactor → D7.

## g. Unused assets

- `raise_env/artifacts/surrogate_fix_fix/` — see §a (high).
- No images/fonts/CSS/i18n in repo. `configs/paper_scale.yaml` used. `baselines_openai/.benchmark_pattern` is upstream vendored metadata (keep with D4).

## h. Obsolete files

- Tracked junk (`*.bak/.old/.orig`, `*.pyc`, `__pycache__`, `.DS_Store`, `build/`, `dist/`, `*.egg-info`, logs): **none tracked**.
- Empty tracked files: only package `__init__.py` markers and `.gitkeep` (intentional).
- **Stale docs (reference removed code):**
  - `docs/WHATS_NEW.md:329,374` → `scripts/run_raise_highway_4h.py` (renamed in §0) — high, update text.
  - `raise_env/AUDIT_FIXES_REPORT.md:60` → `run_raise_highway_4h.py` — high, update text.
  - `raise_env/domains/highway/spec.md:103` lists `high_speed_frac` (metric dropped in `e688e42`) — high, update text.
  - `VERIFICATION_REPORT.md` (73 KB) cites deleted `_verify_*.py` scripts and `_is_highway_metrics` — historical report → **D2**.
  - `raise_env/REWARD_COMPONENTS_VERIFICATION.md` (26 KB) was generated by `_collect_reward_components_verification.py` / `_assemble_verification_report.py` (deleted in §0) — historical → **D2**.
  - Large planning/report docs whose freshness I can't judge: `PROJECT_TECHNICAL_REPORT.md`, `BASELINE_REPORT.md`, `docs/AMFRS_STAGE1_STAGE2_STAGE3_MASTER_PLAN.md` (61 KB), `docs/FIDELITY_AND_DOMAIN_PACK_CHANGE_REPORT.md`, `raise_env/AUDIT.md`, `raise_core/{raise_loop,surrogate,active_learning}/PLAN.md` → **D2**.
- **Untracked local clutter (not in git; deleting only frees disk):**

| Path | Size | Notes |
|------|------|-------|
| `raise_env/results/` (23 run dirs) | 296.5 MB | ignored; contains thesis run outputs incl. label-reliability runs → **D8** |
| `raise_env/artifacts/surr_warm/` | 230.6 MB | ignored; warm-start models (CrowdNav 12h flow) → **D8** |
| `raise_env/domains/highway/data/stage1_dataset_backup_20260925_151302/` | 3.5 MB | **not ignored** (shows in `git status`) → **D8** |
| `.../stage1_dataset_backup_20261001_144321/` | 34.1 MB | not ignored → **D8** |
| `.../stage1_dataset_removed_20260925_201408/` | 9.9 MB | not ignored → **D8** |
| `raise_env/artifacts/_hw_surr_smoke/` | 0.2 MB | ignored smoke output — safe to delete (high) |
| 33 `__pycache__/`, `.pytest_cache/` (root + raise_env), `baselines_openai/baselines.egg-info/` | 1.8 MB | regenerable, ignored — safe (high) |
| `raise_env/domains/crowdnav/runtime/gst_updated/results/` | 1.6 MB | **GST pretrained weights required by CrowdNav — do NOT delete** |

## i. Routes / endpoints / DB models

N/A — no web server, API routes, ORM or migrations in this repo.

## j. `.gitignore` gaps / redundancies

| Finding | Proposed fix | Confidence |
|---------|--------------|------------|
| Stage I backup dirs (`stage1_dataset_backup_*`, `stage1_dataset_removed_*`) are not ignored and pollute `git status` | add `domains/highway/data/stage1_dataset_*/` to `raise_env/.gitignore` | high |
| `raise_env/.gitignore:37-38` whitelists `artifacts/surrogate_test_fix/` which does not exist anywhere | drop the 2 lines (or rename to the dir actually kept, if D-a keeps `surrogate_fix_fix`) | high |
| Root `.gitignore:15` `raise_env/artifacts/surr_warm/` is already covered by `raise_env/.gitignore` `artifacts/**` | optional dedupe | low (harmless) |
| `.env*`: none present in repo; both `.gitignore`s ignore `.env`/`.env.*`; `raise_env/groq_keys.json` exists locally and is ignored ✔ (key contents not inspected/printed) | none | — |

---

## Needs my decision

| # | Item | Options |
|---|------|---------|
| **D1** | Uncommitted duration-script cleanup on `simplify/highway` (§0) | (a) commit it on `simplify/highway` first, then branch `chore/deep-cleanup` from there *(recommended — keeps one commit per step)*; (b) carry it into `chore/deep-cleanup` as its first commit |
| **D2** | Historical/large docs: `VERIFICATION_REPORT.md`, `REWARD_COMPONENTS_VERIFICATION.md`, `PROJECT_TECHNICAL_REPORT.md`, `BASELINE_REPORT.md`, `docs/AMFRS_..._MASTER_PLAN.md`, `docs/FIDELITY_...md`, `raise_env/AUDIT.md`, `AUDIT_FIXES_REPORT.md`, 3× `PLAN.md` | keep all (default) / delete selected / move to `docs/archive/` |
| **D3** | CrowdNav (frozen): legacy shim `crowd_nav/reward_search/` + `crowd_nav/domains/`, CrowdNav-only scripts, root `test.py` commented code, 27 F401/3 F841 | keep (default, rule) — only if you unfreeze CrowdNav |
| **D4** | Vendored `baselines_openai`: 21 unreachable modules incl. 4 upstream tests | keep (default) / prune unreachable modules (CrowdNav dependency → needs unfreeze) |
| **D5** | Deps: `requirements.txt` alias; `tensorflow`+`protobuf` (only for unreachable baselines code, but `baselines_openai/setup.py` asserts TF); `Cython` (RVO2 build) | keep (default) / drop `requirements.txt` alias / drop TF stack (needs D4 + unfreeze) |
| **D6** | Latent `NameError` at `raise_core/pipeline.py:460` | fix separately as a bug commit (outside cleanup, it's a behaviour change) / leave |
| **D7** | Small duplicated helpers in core/tests (§f) | leave (default; merging = refactor) / consolidate |
| **D8** | Local, untracked data: `results/` (297 MB), `artifacts/surr_warm/` (231 MB), 3 Stage I backup dirs (47.5 MB) | keep (default; irreversible) / delete selected |
| **D9** | Medium items in §b: `calibrate_from_reference_rollout` + `rank_population(reference_*)` branch (and its 2 tests), `pending_items`, `format_component_trend_lines`, `OvertakeTracker.passed_by` | remove with their tests / keep |

## Phase 2/3 results (2026-10-01)

Decisions applied: D1 commit first · D2 delete (but keep `AUDIT.md`: referenced by CLI help / error text / CrowdNav code) · D6 separate fix · D8 Stage I backups only · D9 remove · D3/D4/D5/D7 default (keep).

| Commit | Branch | Content |
|--------|--------|---------|
| `e3d870e` | `simplify/highway` | single `run_raise_highway.py`, 23 duration/one-off scripts removed (−4168 lines) |
| `f26fe55` | `simplify/highway` | fix `NameError` at `pipeline.py:460` + regression test |
| `7a3fb07` | `chore/deep-cleanup` | `.gitignore`: ignore `stage1_dataset_*/`, drop `surrogate_test_fix` whitelist |
| `6bca272` | `chore/deep-cleanup` | remove `artifacts/surrogate_fix_fix/` (4 files) |
| `5a5f0b5` | `chore/deep-cleanup` | delete 7 historical reports + 3 `PLAN.md`, `PLAN_PATH`, docstring pointers |
| `f3f8b37` | `chore/deep-cleanup` | unused functions/constants/imports + test-only code and its 3 tests (−298 lines) |
| (this) | `chore/deep-cleanup` | stale doc lines (`WHATS_NEW.md`, `spec.md`) + this report |

- Cleanup branch vs `f26fe55`: 63 files changed, +29 / −5378 lines; 14 tracked files deleted. With `e3d870e`: 37 files deleted, ≈9.5k lines removed.
- Local only (untracked): 3 Stage I backups (47.5 MB), `_hw_surr_smoke`, 33 `__pycache__`, 2 `.pytest_cache`, empty `artifacts/highway_surr_warm/`, empty `raise_env/data/`.
- Steps 5 (deps) and 6 (duplicates): no change per decisions. Step 7: no tracked empty folders.
- Tests: baseline 360 passed / 1 skipped → final **358 passed / 1 skipped** (+1 new bug-fix test, −3 tests removed with their functions). ruff F401/F841 outside CrowdNav: 30 → 7 (all remaining are frozen CrowdNav scripts, the CrowdNav trainer, or a string annotation). vulture hits: 76 → 42 (all verified false positives / frozen).
- Entry points: `run_raise.py --help` and `run_raise_highway.py --help` exit 0 (with `PYTHONIOENCODING=utf-8`).
- Skipped on purpose: `validate.py` `last_ckpt_path` (inside the frozen CrowdNav GST/SRNN trainer); `baselines.egg-info` (backs the editable `baselines` install).
- Found, not fixed (pre-existing): `--help` crashes with `UnicodeEncodeError` when stdout is a cp1252 pipe (`≤` in `run_raise.py:313`, from `0094e6c`); `test_raise_loop.py::test_select_gen0_never_gates` failed once in a full run and passed on rerun (flaky).

### Run wrappers → profiles (user-approved, CrowdNav scripts)

Deleted `run_raise_1h.py`, `run_raise_1h.ps1`, `run_raise_12h.py`, `run_raise_18h.py`, `run_stage2_smoke.py`, `run_stage3_smoke.py`, `run_raise_paper_scale.py`. Their settings now live in `presets.CLOSED_LOOP_PROFILES` and run via `python scripts/run_raise.py --profile smoke|1h|12h|18h|paper_scale`:

- Profiles with `output_dir_prefix` get a stamped `--output-dir` and nested `surrogate_model/` + `surrogate_dataset/` (as the wrappers did); resume = rerun with the same `--output-dir`.
- `12h`: `--warm-surrogate DIR` moved into `run_raise.py`. `18h`: profile now matches what the wrapper actually ran (`num_processes=1`, `--final-rank llm`, `--no-h-sweep`).
- `smoke`: one tiny real A2C/PPO pipeline run on CPU (seed LLM, Score1 smoke, K2=K3=200) replaces the two stage-level smokes (~20 s).
- `paper_scale`: CLI (CI guard, seed gate, `PaperScaleRunner`) moved to `raise_core.paper_scale.run_paper_scale_cli`; tests now target `--profile paper_scale`.
- Dropped: the 12h/18h preflight banners and `_prereqs` checks (failures still surface at run time) and the 18h legacy `artifacts/surrogate_closed_loop_18h_*` resume fallback.
- Tests: **363 passed / 1 skipped** (+5 in `test_run_profiles.py`).

## Proposed Phase 2 plan (after approval, on branch `chore/deep-cleanup`)

1. **Junk + .gitignore** — add the `stage1_dataset_*/` ignore, drop stale `surrogate_test_fix` whitelist; delete local `__pycache__`, `.pytest_cache`, `egg-info`, `artifacts/_hw_surr_smoke/` (untracked, so no diff besides `.gitignore`).
2. **Unused assets** — `git rm -r raise_env/artifacts/surrogate_fix_fix/`.
3. **Unused files** — nothing on the highway path beyond §0 (CrowdNav/baselines only if D3/D4 change).
4. **Unused code/imports** — all **high** items in §b, §c (`last_ckpt_path`), §d (core + tests + `_prereqs.py`); plus D9 items if approved.
5. **Dependencies** — only what D5 decides (default: no change).
6. **Duplicates** — only what D7 decides (default: no change).
7. **Empty folders** — re-scan after step 4.

After each step: `python -m pytest -m "not slow" -q` (must match 360 passed / 1 skipped, minus any tests removed with D9) + ruff F401/F841/F821 count must not increase. Phase 3: update the 3 stale doc references in §h, rerun the full baseline, smoke-import `scripts/run_raise_highway.py --help`, and write the final summary.

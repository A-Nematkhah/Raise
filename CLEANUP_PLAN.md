# CLEANUP_PLAN (Phase 0 inventory)

Verified 2026-10-21 against the working tree. Evidence from grep, file reads, and vulture.
Delete this file at the end of Phase 3.

## Phase action map

| Phase | Commit scope |
|-------|----------------|
| 0 | This plan only |
| 1 | aise_config.py\, factories, CLI shrink, AST test, docs Configuration section |
| 2 | CrowdNav-only pack accessor; remove multi-env / highway guards / \_crowdnavpp_src\ |
| 3 | Dead code, \scripts/dev/\, hygiene, remove this plan |

## Kinematics decision (locked for Phase 1)

| Path | Effective today | New fields |
|------|-----------------|------------|
| Default train/eval (un_raise\, no env file override) | **unicycle** (\CrowdNavSettings\ / \RaiseRunConfig\; \make_crowd_sim\ overwrites file) | \env.kinematics\ |
| Stage I dataset collect CLI default | **holonomic** | \udget.stage1.collect_kinematics\ |
| Hard config file alone | holonomic in file; ignored unless CLI infers kinematics from file | folded into \env\ + \evaluation_variants\ |

Behavior-neutral: keep train/eval = unicycle, Stage I collect = holonomic.

## Acceptance grep targets (Phase 2+)

\highway|mobile_robot|--domain|load_domain\(|available_domains|DEFAULT_DOMAIN|crowdnav_16h(except migration table mentioning removed flags)

---

# Agent 4404e000-8cb4-498a-b462-e5d17752dce1

## 1. Multi-environment machinery inventory (`raise_env/`)

**Scope:** Grep over `F:/Code/New folder/Raise/raise_env` (plus `docs/ARCHITECTURE.md` where the registry is referenced). Counts are **unique logical sites** (duplicate path spellings from the index collapsed).

**Ground truth:** Only one pack exists on disk (`domains/crowdnav/`). Registry holds a single entry; docstrings state that explicitly.

```18:20:F:/Code/New folder/Raise/raise_env/raise_core/domains/__init__.py
_REGISTRY = {
    "crowdnav": "domains.crowdnav.pack",
}
```

```4:6:F:/Code/New folder/Raise/raise_env/raise_core/domains/__init__.py
Each environment family registers a module (``domains/<name>/pack.py``)
exposing ``get_pack() -> DomainPack``. The only registered domain is ``crowdnav``.
```

---

### 1.1 Registry symbols (define + call sites)

| Symbol | Defining location | Call / reference sites (file:line) | Total refs (excl. def/`__all__`) |
|--------|-------------------|-------------------------------------|----------------------------------|
| `_REGISTRY` | `raise_core/domains/__init__.py:18` | `:25`, `:35`, `:38`, `:100` (same file only) | **4** internal |
| `DEFAULT_DOMAIN` | `raise_core/domains/__init__.py:16` | `raise_core/pipeline.py:19` (import), `:120` (default field); `domains/crowdnav/tests/test_no_highway.py:25` (import), `:29` (arg); `load_domain` default param `:28` | **4** external + 1 default param |
| `available_domains()` | `raise_core/domains/__init__.py:23` | `scripts/run_raise.py:38` (import), `:66` (`choices=`); `domains/crowdnav/tests/test_no_highway.py:40`; internal `:36` (error msg) | **3** external + **1** internal |
| `load_domain()` | `raise_core/domains/__init__.py:28` | `raise_core/pipeline.py:381`; `raise_core/raise_loop/runner.py:366`; `domains/crowdnav/viz.py:273`; `scripts/bench_speed.py:1277`; `domains/crowdnav/tests/test_no_highway.py:29`; `domains/crowdnav/tests/test_end_to_end.py:90`; `domains/crowdnav/tests/test_sandbox_and_prompts.py:31`; `raise_core/tests/test_sandbox_comprehension_ban.py:34`, `:45`, `:59`; `explore.py:258` (string in error only); `domains/README.md:10`, `docs/ARCHITECTURE.md:44` (docs) | **10** code calls + **2** doc mentions |
| `register_domain()` | `raise_core/domains/__init__.py:97` | *(none)* | **0** callers |
| `DomainPack` | `raise_core/domains/base.py:34` (class) | `raise_core/domains/__init__.py:13`, `:28`, `:44–45`, `:54`, `:72`, `:81`, `:89`, `:105`; `raise_core/pipeline.py:20`, `:89`, `:376`; `domains/crowdnav/pack.py:7`, `:30–32`; `raise_core/raise_loop/proxy_feedback.py:4` (doc); `domains/crowdnav/settings.py:111` (doc) | **~15** type/construct/doc |
| `EnvAdapter` | `raise_core/domains/base.py:16` (Protocol) | `raise_core/domains/base.py:39`, `:48`; `raise_core/domains/__init__.py:13`, `:106` | **No** `isinstance(..., EnvAdapter)` or structural checks in repo |
| `with_adapter` | `load_domain(..., with_adapter=)` `__init__.py:28`, `:43`; `domains/crowdnav/pack.py:30–31`, `:53–54` | Same files only | **6** sites, **2** files |

**CrowdNav adapter (de facto env surface, not via `EnvAdapter` typing):** `CrowdNavAdapter` at `domains/crowdnav/adapter.py:57`; wired in `domains/crowdnav/pack.py:31`, `:43`. Core uses **`make_stage2_trainer` / `make_stage3_trainer` factories**, not `pack.adapter` — **zero** reads of `pack.adapter` / `domain_pack.adapter` anywhere.

---

### 1.2 `make_*_for_domain` helpers

| Helper | Defined | Runtime calls | Test / monkeypatch string refs |
|--------|---------|---------------|----------------------------------|
| `make_score_fn_for_domain` | `raise_core/domains/__init__.py:53` | `raise_core/pipeline.py:520` (**1**) | — |
| `make_stage2_trainer_for_domain` | `:71` | `pipeline.py:523`, `raise_loop/runner.py:376` (**2**) | — |
| `make_stage3_trainer_for_domain` | `:80` | `pipeline.py:563` (**1**) | `test_pipeline.py:134`, `:155`; `test_stage3_part8_fixes.py:68`; `test_pipeline_stage3_selection.py:30`, `:102`, `:132`, `:177` (patch targets) |
| `make_validator_for_domain` | `:89` | `pipeline.py:384`, `runner.py:378`, `viz.py:274`, `bench_speed.py:1278`, `test_sandbox_and_prompts.py:31`, `test_end_to_end.py:90`, `test_sandbox_comprehension_ban.py:35`, `:45`, `:59` (**9**) | `explore.py:249` (error string) |

---

### 1.3 `--domain` CLI / config plumbing

| Location | Evidence |
|----------|----------|
| CLI flag | `F:/Code/New folder/Raise/raise_env/scripts/run_raise.py:66` — `"--domain", choices=available_domains(), default=_DEFAULTS.domain`; wired at `:568` `domain=args.domain` |
| Config fields | `RaiseRunConfig.domain` `raise_core/pipeline.py:120` (`DEFAULT_DOMAIN`); `ClosedLoopConfig.domain` `raise_core/raise_loop/config.py:51` (literal `"crowdnav"`, **not** imported from `DEFAULT_DOMAIN`) |
| Runtime load | `pipeline.py:381` `load_domain(self.config.domain)`; `runner.py:366` `load_domain(cfg.domain)` |
| Consistency check | `pipeline.py:492–495` `pack.name` vs `cfg.domain` |
| Test | `raise_core/tests/test_stage3_part8_fixes.py:228` `"--domain", "crowdnav"` in `--print-config` argv |

With one registered name, `--domain` is effectively **no-op choice** (`crowdnav` only).

---

### 1.4 `domains/README.md` (summary)

Path: `F:/Code/New folder/Raise/raise_env/domains/README.md`

- States RAISE has **one domain**: `crowdnav` → `domains.crowdnav`, CrowdSimVarNum + SB3 PPO.
- Points registry to `raise_core.domains` (`load_domain`, trainer / validator / Score1 factories).
- Layout tree: `domains/crowdnav/{pack,prompts,env_wrapper,adapter,…}`, `tests/`, `data/`.
- No “add a new domain” procedure (phrase **not found** in `raise_env/`).

Package stub: `F:/Code/New folder/Raise/raise_env/domains/__init__.py:1` — `"""Environment domain packs (CrowdNav)."""`

---

### 1.5 Highway / legacy-env guards

**`test_no_highway.py`** — `F:/Code/New folder/Raise/raise_env/domains/crowdnav/tests/test_no_highway.py`

| Test | Lines | Role |
|------|-------|------|
| `test_core_and_pack_import_without_highway_env` | `:45–50` | Subprocess import guard; expects `DOMAINS ['crowdnav'] crowdnav`, empty `HIGHWAY_MODULES` |
| `test_requirements_and_ci_do_not_install_highway_env` | `:53–60` | No `highway` in root reqs / CI; pins gym + RVO2 |
| `test_no_source_file_imports_highway` | `:64–76` | Walks `raise_core`, `domains`, `scripts` for `highway_env` / `domains.highway` imports |

**Prompt lexicon ban (Highway-era wording):** `test_sandbox_and_prompts.py:185–186` — asserts `highway`, `lane`, `vehicle`, `overtake` **not** in lowered prompt text (`test_prompts_describe_the_crowdnav_state_and_objective`, from `:175`).

**Residual Highway artifacts (not live code):**

- `raise_env/.gitignore:9–10` — `domains/highway/` “removed HighwayEnv domain”
- `raise_env/NOTICE.md:54–55` — historical HighwayEnv / vita-epfl CrowdNav
- `test_no_highway.py:65` — still scans for `domains.highway` imports (no such package today)

---

### 1.6 `mobile_robot_env`, domain-agnostic, other environments

| Pattern | Hits | Notes |
|---------|------|-------|
| `mobile_robot_env` | `raise_core/sandbox/__init__.py:2`; `config.py:4`; `runtime.py:4`; `validator.py:4`; `ast_policy.py:4` | Provenance comments only; **reward sandbox**, not env registry |
| `domain-agnostic` | `raise_core/stats.py:1`; `domains/crowdnav/metrics.py:26` | Stats / re-exports — **keep** as shared math |
| `every domain` | `raise_core/feedback.py:2` | Doc wording only |
| `pluggable` / `environment family` | `raise_core/domains/base.py:2`, `:5`; registry module doc `:4–5` | Contract language for future packs |
| `other environments` | `third_party/crowdnavpp/crowd_sim/envs/crowd_sim.py:22` | Upstream CrowdSim comment, unrelated to RAISE registry |
| `add a new domain` | **0** matches in `raise_env/` |

---

### 1.7 Is `raise_core/domains/` multi-domain ready?

**Files:** only `F:/Code/New folder/Raise/raise_env/raise_core/domains/__init__.py` and `base.py`.

| Aspect | Assessment |
|--------|------------|
| Registry | Dict + `importlib` + `get_pack()` — **extension-shaped**, but **one** entry |
| `register_domain` | Public API, **never called** — dead unless tests/extensions added |
| `DomainPack` | Rich dataclass (prompts, Score1, trainers, validator, configure, post_run) — **real integration boundary** for CrowdNav today |
| `EnvAdapter` | Protocol defined and stored on `DomainPack`, but **unused** by pipeline/runner (factories used instead); **no** explicit `CrowdNavAdapter(EnvAdapter)` |
| Disk layout | `raise_env/domains/` contains **only** `crowdnav/` (+ README) |

**Verdict:** **CrowdNav-only in practice**, with **scaffolding** (registry, dynamic import, CLI `--domain`, `register_domain`, pluggable docstrings) left from multi-env / Highway era. The valuable part to keep is **`DomainPack` + factories + `domains/crowdnav/pack.py`**, not the multi-name registry UX.

---

### 1.8 Delete vs keep (CrowdNav-only simplification)

| Item | Recommendation | Rationale |
|------|----------------|-----------|
| `DomainPack`, `get_pack()`, `make_*_for_domain`, `load_domain("crowdnav")` | **Keep** (optionally rename to `load_crowdnav_pack`) | Active pipeline/runner/test path |
| `_REGISTRY`, `available_domains()`, `register_domain` | **Delete or collapse** | Single entry; `register_domain` has 0 callers |
| `DEFAULT_DOMAIN`, `RaiseRunConfig.domain`, CLI `--domain` | **Simplify** | Redundant with one env; duplicate default in `raise_loop/config.py:51` |
| `pack.adapter` / `with_adapter=False` paths | **Candidate delete** | Adapter field never read; only factories matter |
| `EnvAdapter` export + Protocol | **Optional delete** | No enforcement; CrowdNav uses concrete trainers |
| Domain mismatch check `pipeline.py:492–495` | **Keep** if `domain` field kept; else drop with field | Guards injected `domain_pack` vs config |
| `test_no_highway.py`, prompt highway-word test | **Keep** | Regression guard for removed HighwayEnv dependency |
| `.gitignore` `domains/highway/` | **Delete** when cleaning docs/artifacts | No code references |
| `mobile_robot_env` sandbox comments | **Keep** (or shorten) | Attribution, not multi-env machinery |
| `raise_core/feedback.py` “every domain” wording | **Reword** on cleanup | Cosmetic |

---

### 1.9 Related paths (absolute)

- Registry: `F:/Code/New folder/Raise/raise_env/raise_core/domains/__init__.py`
- Contract: `F:/Code/New folder/Raise/raise_env/raise_core/domains/base.py`
- CrowdNav pack: `F:/Code/New folder/Raise/raise_env/domains/crowdnav/pack.py`
- Domain docs: `F:/Code/New folder/Raise/raise_env/domains/README.md`
- Architecture (external to `raise_env/`): `F:/Code/New folder/Raise/docs/ARCHITECTURE.md:18–20`, `:44`

[REDACTED]


---


# Agent 2c479a78-6f98-44ad-b5ee-567d33335bac

# CLEANUP_PLAN.md — Section 2: Config surface inventory

Base path: `F:/Code/New folder/Raise/raise_env`

---

## 2.1 Dataclass / settings summaries (fields + defaults)

### `RaiseRunConfig` — `F:/Code/New folder/Raise/raise_env/raise_core/pipeline.py`

| Field | Default |
|-------|---------|
| `output_dir` | `"results/raise_run"` |
| `seed` | `425` |
| `domain` | `"crowdnav"` (`DEFAULT_DOMAIN`) |
| `llm_provider` | `"seed"` |
| `llm_model` | `None` |
| `llm_max_tokens` | `None` |
| `llm_reasoning_effort` | `None` |
| `llm_seed` | `None` |
| `score1_mode` | `"dataset"` |
| `stage1_dataset_path` | `"domains/crowdnav/data/stage1_dataset"` |
| `population` | `6` (`POPULATION_SIZE`) |
| `generations` | `4` (`GENERATIONS`) |
| `stage2_train_steps` | `12000` |
| `stage2_eval_episodes` | `30` |
| `stage2_eval_split` | `"val"` |
| `stage2_paired_seeds` | `True` |
| `stage2_seeds` | `1` |
| `stage3_train_steps` | `70000` (`STAGE3_STEPS`) |
| `stage3_eval_episodes` | `30` |
| `stage3_warm_start` | `False` |
| `stage3_skip_collapsed` | `True` |
| `collapse_cr` | `0.8` |
| `stage3_seeds` | `1` |
| `stage3_top_k` | `None` |
| `final_eval_episodes` | `100` |
| `eval_tied_finalists` | `True` |
| `use_stub` | `False` |
| `elitism` | `True` |
| `n_crossover` / `n_mutation` / `n_random` | `None` each |
| `mutation_parents` | `"lower"` |
| `mutation_exclude_collapsed` | `False` |
| `stage2_successive_halving` | `False` |
| `resume` | `True` |
| `allow_config_drift` | `False` |
| `infra_error_policy` | `"retry_then_abort"` |
| `max_fallback_fraction` | `0.5` |
| `strict_fallback` | `False` |
| `ppo_normalize_reward` | `False` |
| `n_envs` | `1` |
| `label_workers` | `1` |
| `warm_start` | `False` |
| `eval_mode` | `"holdout_only"` |
| `action_mode` | `"continuous"` |
| `robot_kinematics` | `"unicycle"` |
| `action_speed_samples` | `5` |
| `action_rotation_samples` | `16` |
| `env_config_path` | `None` |
| `feedback` | `FeedbackCardConfig()` |
| `selection` | `SelectionConfig()` |
| `device` | `"cuda"` |
| `torch_threads` | `1` |
| `fast` | `False` |

**Not on dataclass:** `env_config_obj`, `human_num_override` (only via `CrowdNavSettings` / `settings_from_config` if ever attached).

**`apply_fast_profile()` overrides:** `fast=True`, `score1_mode="smoke"`, `population=2`, `generations=2`, `stage2_train_steps=8`, `stage2_eval_episodes=2`, `stage3_train_steps=8`, `stage3_eval_episodes=2`, `final_eval_episodes=2`, `use_stub=True`, `llm_provider="seed"`, operator counts `None`.

---

### `ClosedLoopConfig` — `F:/Code/New folder/Raise/raise_env/raise_core/raise_loop/config.py`

| Field | Default | Notes vs `RaiseRunConfig` |
|-------|---------|---------------------------|
| `population_size` | `6` | renamed from `population` |
| `generations` | `4` | |
| `stage2_train_steps` | `12000` | |
| `stage1_dataset_path` | `"domains/crowdnav/data/stage1_dataset"` | |
| `seed` | `425` | |
| `device` | **`"cpu"`** | pipeline passes `cfg.device` (`"cuda"`) |
| `use_stub` | `False` | |
| `llm_provider` | `"seed"` | |
| `output_dir` | `"results/closed_loop"` | pipeline uses run `output_dir` |
| `keep_runtime_elite` | `True` | maps from `RaiseRunConfig.elitism` |
| `n_crossover` / `n_mutation` / `n_random` | `None` | |
| `mutation_parents` | `"lower"` | |
| `mutation_exclude_collapsed` | `False` | |
| `collapse_cr` | `0.8` | |
| `resume` | `True` | |
| `domain` | `"crowdnav"` | |
| `eval_episodes` | **`None`** | runner: `8` if stub else **`30`** when `None` |
| `n_envs` | `1` | |
| `label_workers` | `1` | |
| `warm_start` | `False` | |
| `stage2_successive_halving` | `False` | |
| `stage2_paired_seeds` | `True` | |
| `stage2_seeds` | `1` | |
| `eval_mode` | `"holdout_only"` | runner Stage2 uses **`"both"`** fallback if empty; pipeline passes `holdout_only` |
| `stage2_eval_split` | `"val"` | |
| `action_mode` | `"continuous"` | pipeline `_space_knobs` uses **`or "discrete"`** (latent bug if falsy) |
| `robot_kinematics` | `"unicycle"` | |
| `action_speed_samples` | `5` | |
| `action_rotation_samples` | `16` | |
| `env_config_path` | `None` | |
| `infra_error_policy` | `"retry_then_abort"` | |
| `allow_config_drift` | `False` | |
| `max_fallback_fraction` | `0.5` | |
| `strict_fallback` | `False` | |
| `ppo_normalize_reward` | `False` | |
| `feedback` | `FeedbackCardConfig()` | |
| `selection` | `SelectionConfig()` | |

---

### `Stage2Config` — `F:/Code/New folder/Raise/raise_env/raise_core/refine.py`

| Field | Default |
|-------|---------|
| `train_env_steps` | `12000` |
| `eval_episodes` | `30` |
| `seed` | `425` |
| `output_root` | `"trained_models/stage2"` |
| `device` | `"cpu"` |
| `n_envs` | `1` |
| `warm_start` | `False` |
| `eval_mode` | **`"both"`** |
| `eval_split` | `"val"` |
| `action_mode` | `"continuous"` |
| `robot_kinematics` | `"unicycle"` |
| `action_speed_samples` | `5` |
| `action_rotation_samples` | `16` |
| `env_config_path` | `None` |
| `ppo_ent_coef` | `0.0` |
| `eval_deterministic` | `False` |
| `ppo_n_steps` | `256` |
| `ppo_n_epochs` | `10` |
| `ppo_gamma` | `None` → domain `PPO_GAMMA` (`0.99`) |
| `eval_seed_base` | `None` → per-candidate seed |
| `record_training_dynamics` | `False` (runner sets from feedback card) |
| `record_reward_components` | `False` (runner sets from feedback card) |
| `ppo_normalize_reward` | `False` |

---

### `Stage3Config` — `F:/Code/New folder/Raise/raise_env/raise_core/validate.py`

| Field | Default |
|-------|---------|
| `train_env_steps` | `70000` |
| `eval_episodes` | `30` |
| `seed` | `425` |
| `output_root` | `"trained_models/stage3"` |
| `device` | `"cpu"` |
| `resume` | `True` |
| `action_mode` | `"continuous"` |
| `robot_kinematics` | `"unicycle"` |
| `action_speed_samples` | `5` |
| `action_rotation_samples` | `16` |
| `env_config_path` | `None` |
| `eval_mode` | `"both"` |
| `eval_split` | `"val"` |
| `warm_start` | `False` |
| `n_envs` | `1` |
| `ppo_n_steps` | `256` |
| `ppo_gamma` | `None` → `0.99` |
| `eval_deterministic` | `False` |
| `eval_seed_base` | `None`; pipeline sets `seed + STAGE3_EVAL_SEED_OFFSET` |
| `skip_collapsed` | `True` |
| `collapse_cr` | `0.8` |
| `n_seeds` | `1` |
| `top_k` | `None` |

---

### `StageIConfig` — `F:/Code/New folder/Raise/raise_env/raise_core/explore.py`

| Field | Default | Closed-loop mapping |
|-------|---------|---------------------|
| `population_size` | **`8`** | overridden → `ClosedLoopConfig.population_size` (`6`) |
| `generations` | **`10`** | overridden → `ClosedLoopConfig.generations` (`4`) |
| `n_crossover` / `n_mutation` / `n_random` | `None` | from closed loop |
| `max_invalid_replacements` | `16` | **not** wired from CLI |
| `func_name` | `"compute_reward"` | |
| `include_external_knowledge` | `True` | |
| `keep_runtime_elite` | `True` | from `keep_runtime_elite` / elitism |
| `mutation_parents` | `"lower"` | |
| `mutation_exclude_collapsed` | `False` | |
| `collapse_cr` | `0.8` | |
| `seed` | **`0`** | closed loop uses `ClosedLoopConfig.seed` (`425`) |
| `behavioral_dedup` | `True` | **not** on CLI |
| `max_fallback_fraction` | `0.5` | |
| `strict_fallback` | `False` | |

Standalone Stage I scripts/tests use `8/10`; end-to-end RAISE uses `6/4` from `defaults.py`.

---

### `SelectionConfig` — `F:/Code/New folder/Raise/raise_env/raise_core/evo_selection/config.py`

| Field | Default |
|-------|---------|
| `mode` | `"legacy"` |
| `top_k` | `2` |
| `n_votes` | `3` |
| `temperature` | `0.1` |
| `p_restart` | `0.1` |
| `guards` | `False` |
| `guard_success_threshold` | `0.9` |
| `guard_min_path_length` | `0.1` |
| `reflection_note_cap` | `3` |
| `seed` | `425` |

---

### `FeedbackCardConfig` — `F:/Code/New folder/Raise/raise_env/raise_core/feedback.py`

| Field | Default |
|-------|---------|
| `enabled` | `True` |
| `max_input_tokens` | `6000` |
| `target_input_tokens` | `5500` |
| `history_entries` | `3` |
| `max_trajectories` | `3` |
| `random_calls` | `True` |
| `training_dynamics` | `True` |
| `component_instrumentation` | `True` |

---

### `SandboxConfig` — `F:/Code/New folder/Raise/raise_env/raise_core/sandbox/config.py`

| Field | Default |
|-------|---------|
| `timeout_seconds` | `5.0` |
| `startup_timeout_seconds` | `120.0` |
| `max_code_length` | `20000` |
| `required_function_name` | `"compute_reward"` |
| `required_arg_name` | `"state"` |
| `required_memory_arg_name` | `"memory"` |
| `allowed_modules` | `("math",)` |
| `forbidden_names` | long tuple (see file) |
| `allow_while` | `False` |
| `allow_imports` | `False` |
| `allowed_attributes` | `None` (CrowdNav pack sets allowlist) |
| `restrict_top_level` | `False` (CrowdNav: `True` via `crowdnav_sandbox_config()`) |
| `max_call_seconds` | `1.0` |
| `max_mean_call_seconds` | `0.02` |
| `mean_call_window` | `100` |
| `smoke_max_call_seconds` | `0.1` |
| `max_abs_reward` | `1e4` |
| `max_range_length` | `10000` |
| `max_memory_keys` | `64` |
| `max_memory_value_len` | `10000` |
| `smoke_rlimit_as_bytes` | `2 * 1024**3` |
| `max_pow_exponent` | `64` |
| `max_pow_bits` | `64` |
| `max_shift_bits` | `64` |
| `max_sequence_repeat` | `10000` |
| `max_factorial_arg` | `1000` |

---

### `CrowdNavSettings` — `F:/Code/New folder/Raise/raise_env/domains/crowdnav/settings.py`

| Field | Default |
|-------|---------|
| `kinematics` | `"unicycle"` |
| `action_mode` | `"continuous"` |
| `speed_samples` | `5` |
| `rotation_samples` | `16` |
| `env_config_path` | `None` |
| `env_config_obj` | `None` |
| `human_num_override` | `None` |

**Default env file when path/obj unset:** `F:/Code/New folder/Raise/raise_env/third_party/crowdnavpp/crowd_nav/configs/config.py` (`DEFAULT_ENV_CONFIG_PATH`).

**Mapping from run configs:** `robot_kinematics` → `kinematics`; `action_mode`; `action_speed_samples` → `speed_samples`; `action_rotation_samples` → `rotation_samples`; `env_config_path` / optional `env_config_obj`; optional `human_num_override` (no first-class field on `RaiseRunConfig`).

---

## 2.2 `raise_core/defaults.py`

| Constant | Value |
|----------|-------|
| `POPULATION_SIZE` | `6` |
| `GENERATIONS` | `4` |
| `STAGE2_TRAIN_STEPS` | `12000` |
| `STAGE3_TRAIN_STEPS` | `70000` |
| `STAGE2_EVAL_EPISODES` | `30` |
| `STAGE3_EVAL_EPISODES` | `30` |
| `FINAL_EVAL_EPISODES` | `100` |
| `PPO_N_STEPS` | `256` |
| `PPO_N_EPOCHS` | `10` |
| `STAGE2_SEEDS` | `1` |
| `STAGE2_PAIRED_SEEDS` | `True` |
| `INFRA_ERROR_POLICY` | `"retry_then_abort"` |
| `ALLOW_CONFIG_DRIFT` | `False` |
| `MAX_FALLBACK_FRACTION` | `0.5` |
| `STRICT_FALLBACK` | `False` |
| `PPO_NORMALIZE_REWARD` | `False` |
| `STAGE3_STEPS` | alias of `STAGE3_TRAIN_STEPS` |

---

## 2.3 Domain Python env configs

### `F:/Code/New folder/Raise/raise_env/configs/crowdnav_hard.py`

Clones upstream `Config` sections, then sets:

| Section | Fields set |
|---------|------------|
| `env` | `time_limit=30`, `time_step=0.25`, `val_size=100`, `test_size=500` |
| `reward` | `success_reward=1.0`, `collision_penalty=-0.25`, `discomfort_dist=0.2`, `discomfort_penalty_factor=0.5` |
| `sim` | `circle_radius=6`, `human_num=10`, `human_num_range=0`, `predict_method="const_vel"`, `predict_steps=5` |
| `data` | `pred_timestep=0.25` |
| **`action_space`** | **`kinematics="holonomic"`** |
| `robot` | `visible=True` |

All other section values remain from vendored upstream defaults at import time.

### `F:/Code/New folder/Raise/raise_env/configs/crowdnav_generalization_17.py`

Deep-copies `crowdnav_hard.Config`, then:

- `sim.human_num = 17`
- `sim.human_num_range = 0`

(inherits holonomic + all hard overrides)

---

## 2.4 `env_config_path` / `env_config_obj` / `human_num_override` plumbing

### Read path (effective env file)

1. **`CrowdNavSettings.env_config_source`** — `env_config_obj` if set, else `env_config_path`, else `DEFAULT_ENV_CONFIG_PATH`.
2. **`settings_from_config(config)`** — reads `env_config_path`, `env_config_obj`, `human_num_override` from any stage/run config; maps robot/action fields.
3. **`load_env_config(source)`** (`env_wrapper.py`) — loads/clones Python config module or object.
4. **`make_crowd_sim`** — loads config, applies overrides (below), builds `CrowdSimVarNum`.
5. **`prompts._configured_env()`** — `load_env_config(settings.env_config_source)` for task text + human counts.

### Write / override sites

| Location | What happens |
|----------|----------------|
| `make_crowd_sim` | If `human_num_override` set → `config.sim.human_num`, `human_num_range=0` |
| `make_crowd_sim` | **`config.action_space.kinematics = active.kinematics`** (RAISE wins over file) |
| `make_crowd_sim` | `robot.policy="srnn"`, `env.use_wrapper=False`, prediction method normalization |
| `set_settings` / `configure_from_config` | Process-wide `_SETTINGS` (prompts, smoke states) |
| `CrowdNavAdapter._train_and_eval` | `set_settings(settings_from_config(config))` each train/eval |

### Who sets `env_config_path` (production)

| File | Usage |
|------|--------|
| `scripts/run_raise.py` | `RaiseRunConfig.env_config_path = args.env_config` |
| `raise_core/pipeline.py` | `_closed_loop_config` / `_stage3_config` via `_space_knobs` |
| `raise_core/raise_loop/runner.py` | `Stage2Config.env_config_path` |
| `raise_core/validate.py` | Stage III config dict persistence |
| `scripts/collect_crowdnav_stage1_dataset.py` | CLI `--env-config` |
| `scripts/evaluate_crowdnav_generalization.py` | generalization config path |
| `scripts/calibrate_crowdnav_native.py`, `check_orca_reference.py`, `run_baselines.py` | read from saved run or CLI |

### `env_config_obj`

- **Production:** never set on run configs.
- **Tests only:** `domains/crowdnav/tests/test_env_wrapper.py`, `test_sandbox_and_prompts.py`.
- **Plumbing:** `settings_from_config` passes through; `env_config_source` prefers obj over path.

### `human_num_override`

- **Production:** no CLI; always `None` unless tests or manual `CrowdNavSettings(...)`.
- **Read:** `settings_from_config`; **write to sim:** `make_crowd_sim`; **prompts:** human count text in `prompts.py`.
- **Not** in resume drift keys (only `env_config_path` is).

### Process-wide configure hooks

- `RaisePipeline.__init__` → `domain_pack.apply_config(RaiseRunConfig)` → `configure_from_config`.
- `ClosedLoopRunner.run` → `pack.apply_config(ClosedLoopConfig)` again before Stage I/II.

---

## 2.5 Module-level constants (beyond `defaults.py`)

| Symbol | Value | File |
|--------|-------|------|
| `PPO_GAMMA` | `0.99` | `domains/crowdnav/objective_constants.py` |
| `CROWDNAV_FITNESS_OBJECTIVE` | `"mean_outcome_score"` | same |
| `CROWDNAV_EVAL_GAMMA` | `1.0` | same |
| `STAGE3_EVAL_SEED_OFFSET` | `10_000_000` | `raise_core/validate.py` |
| `FINAL_EVAL_SEED_OFFSET` | `20_000_000` | `raise_core/validate.py` |
| `FINAL_EVAL_SPLIT` | `"test"` | `raise_core/validate.py` |
| `TRAIN_EVAL_SEED_OFFSET` | `10_003` | `domains/crowdnav/adapter.py` |
| `HOLDOUT_SEED_OFFSET` | `50_003` | `domains/crowdnav/adapter.py` |
| `MIN_SEEDS_FOR_ACROSS_SEEDS_SE` | `3` | `raise_core/seed_aggregate.py` |
| `DEFAULT_ENV_CONFIG_PATH` | vendored `config.py` | `domains/crowdnav/settings.py` |
| `UNICYCLE_*_DELTA_*` | speed/rotation clip bounds | `settings.py` |
| `DEFAULT_STAGE1_DATASET` | path string | `domains/crowdnav/stage1.py` |
| `OUTCOME_SCORES` | success/collision/timeout | `domains/crowdnav/metrics.py` |
| Closed-loop gen eval | `seed + g * 1000` | comments in `validate.py` / adapter |

Vendored upstream default: `action_space.kinematics = "holonomic"` in `third_party/crowdnavpp/crowd_nav/configs/config.py`.

---

## 2.6 `scripts/run_raise.py` CLI → config (74 `add_argument` calls)

Effective defaults: dataclass defaults unless noted. `robot_kinematics` resolved in `main()` before `RaiseRunConfig` build.

| CLI flag | dest | Config field | Default (effective) |
|----------|------|--------------|---------------------|
| `--print-config` | `print_config` | *(exit)* | `False` |
| `--output-dir` | `output_dir` | `RaiseRunConfig.output_dir` | `None` → `results/crowdnav_<timestamp>` |
| `--seed` | `seed` | `seed` | `425` |
| `--domain` | `domain` | `domain` | `"crowdnav"` |
| `--llm` | `llm` | `llm_provider` | `"seed"` |
| `--llm-model` | `llm_model` | `llm_model` | `None` |
| `--llm-max-tokens` | `llm_max_tokens` | `llm_max_tokens` | `None` |
| `--llm-reasoning-effort` | `llm_reasoning_effort` | `llm_reasoning_effort` | `None` |
| `--allow-seed-llm` | `allow_seed_llm` | *(gate only)* | `False` |
| `--verbose` | `verbose` | *(logging)* | `False` |
| `--score1` | `score1` | `score1_mode` | `"dataset"` (`"smoke"` if `--fast`) |
| `--stage1-dataset` | `stage1_dataset` | `stage1_dataset_path` | `"domains/crowdnav/data/stage1_dataset"` |
| `--fast` | `fast` | `use_stub` + `apply_fast_profile()` | `False` |
| `--device` | `device` | `device` | `"cuda"` |
| `--population` | `population` | `population` | `6` |
| `--generations` | `generations` | `generations` | `4` |
| `--stage2-train-steps` | `stage2_train_steps` | `stage2_train_steps` | `12000` |
| `--stage2-eval-episodes` | `stage2_eval_episodes` | `stage2_eval_episodes` | `30` |
| `--stage2-eval-split` | `stage2_eval_split` | `stage2_eval_split` | `"val"` |
| `--stage3-train-steps` | `stage3_train_steps` | `stage3_train_steps` | `70000` |
| `--stage3-eval-episodes` | `stage3_eval_episodes` | `stage3_eval_episodes` | `30` |
| `--stage3-warm-start` | `stage3_warm_start` | `stage3_warm_start` | `False` |
| `--stage3-skip-collapsed` | `stage3_skip_collapsed` | `stage3_skip_collapsed` | `True` |
| `--collapse-cr` | `collapse_cr` | `collapse_cr` | `0.8` |
| `--stage3-seeds` | `stage3_seeds` | `stage3_seeds` | `1` |
| `--stage3-top-k` | `stage3_top_k` | `stage3_top_k` | `None` |
| `--final-eval-episodes` | `final_eval_episodes` | `final_eval_episodes` | `100` |
| `--eval-tied-finalists` | `eval_tied_finalists` | `eval_tied_finalists` | `True` |
| `--elitism` | `elitism` | `elitism` → `ClosedLoopConfig.keep_runtime_elite` | `True` |
| `--n-crossover` | `n_crossover` | `n_crossover` | `None` |
| `--n-mutation` | `n_mutation` | `n_mutation` | `None` |
| `--n-random` | `n_random` | `n_random` | `None` |
| `--mutation-parents` | `mutation_parents` | `mutation_parents` | `"lower"` |
| `--mutation-exclude-collapsed` | `mutation_exclude_collapsed` | `mutation_exclude_collapsed` | `False` |
| `--stage2-successive-halving` | `stage2_successive_halving` | `stage2_successive_halving` | `False` |
| `--stage2-seeds` | `stage2_seeds` | `stage2_seeds` | `1` |
| `--stage2-paired-seeds` | `stage2_paired_seeds` | `stage2_paired_seeds` | `True` |
| `--infra-error-policy` | `infra_error_policy` | `infra_error_policy` | `"retry_then_abort"` |
| `--allow-config-drift` | `allow_config_drift` | `allow_config_drift` | `False` |
| `--max-fallback-fraction` | `max_fallback_fraction` | `max_fallback_fraction` | `0.5` |
| `--strict-fallback` | `strict_fallback` | `strict_fallback` | `False` |
| `--ppo-normalize-reward` | `ppo_normalize_reward` | `ppo_normalize_reward` | `False` |
| `--llm-seed` | `llm_seed` | `llm_seed` | `None` |
| `--n-envs` | `n_envs` | `n_envs` | `1` |
| `--label-workers` | `label_workers` | `label_workers` | `1` |
| `--torch-threads` | `torch_threads` | `torch_threads` | `1` |
| `--eval-mode` | `eval_mode` | `eval_mode` | `"holdout_only"` |
| `--warm-start` | `warm_start` | `warm_start` (Stage II only) | `False` |
| `--action-mode` | `action_mode` | `action_mode` | `"continuous"` |
| `--robot-kinematics` | `robot_kinematics` | `robot_kinematics` | `None` → see §2.7 |
| `--action-speed-samples` | `action_speed_samples` | `action_speed_samples` | `5` |
| `--action-rotation-samples` | `action_rotation_samples` | `action_rotation_samples` | `16` |
| `--env-config` | `env_config` | `env_config_path` | `None` |
| `--feedback-card` | `feedback_card` | `feedback.enabled` | `True` |
| `--max-input-tokens` | `max_input_tokens` | `feedback.max_input_tokens` | `6000` |
| `--target-input-tokens` | `target_input_tokens` | `feedback.target_input_tokens` | `5500` |
| `--feedback-history` | `feedback_history` | `feedback.history_entries` | `3` |
| `--feedback-trajectories` | `feedback_trajectories` | `feedback.max_trajectories` | `3` |
| `--feedback-random-calls` | `feedback_random_calls` | `feedback.random_calls` | `True` |
| `--feedback-training-dynamics` | `feedback_training_dynamics` | `feedback.training_dynamics` | `True` |
| `--feedback-components` | `feedback_components` | `feedback.component_instrumentation` | `True` |
| `--selection-mode` | `selection_mode` | `selection.mode` | `"legacy"` |
| `--selection-top-k` | `selection_top_k` | `selection.top_k` | `2` |
| `--selection-votes` | `selection_votes` | `selection.n_votes` | `3` |
| `--selection-temperature` | `selection_temperature` | `selection.temperature` | `0.1` |
| `--selection-restart-probability` | `selection_restart_probability` | `selection.p_restart` | `0.1` |
| `--selection-guards` | `selection_guards` | `selection.guards` | `False` |
| `--selection-guard-success-threshold` | `selection_guard_success_threshold` | `selection.guard_success_threshold` | `0.9` |
| `--selection-guard-min-path-length` | `selection_guard_min_path_length` | `selection.guard_min_path_length` | `0.1` |
| `--selection-reflection-note-cap` | `selection_reflection_note_cap` | `selection.reflection_note_cap` | `3` |
| `--selection-seed` | `selection_seed` | `selection.seed` | `None` → `--llm-seed` or `--seed` |
| `--resume` / `--no-resume` | `resume` | `resume` | `True` |
| `--baselines` | `baselines` | *(post-run script)* | `False` |

**Not exposed on CLI:** `Stage2Config`/`Stage3Config` PPO knobs (`ppo_n_steps`, `ppo_gamma`, `eval_deterministic`, …), `StageIConfig.max_invalid_replacements`, `behavioral_dedup`, `human_num_override`, `env_config_obj`.

---

## 2.7 Kinematics: `CrowdNavSettings.kinematics` vs `configs/crowdnav_hard.py`

**Authority:** At sim build time, **`make_crowd_sim` always sets `config.action_space.kinematics = CrowdNavSettings.kinematics`**. The value in the Python env file is loaded then overwritten. Policy/action clipping uses `CrowdNavSettings` (`continuous_to_velocity`, discrete tables, prompts).

**`run_raise.py` resolution** (`main`):

- If `--robot-kinematics` omitted and `--env-config` **set** → `robot_kinematics = load_env_config(path).action_space.kinematics` (hard file → **holonomic**).
- If both omitted → **`unicycle`** (`RaiseRunConfig` default).
- Explicit `--robot-kinematics` wins.

### Effective kinematics by path

| Path | Env config file | Effective kinematics | Notes |
|------|-----------------|----------------------|--------|
| **Default closed-loop / Stage II / Stage III** (`run_raise`, no `--env-config`) | Vendored upstream (`holonomic` in file) | **`unicycle`** | RAISE default + overwrite in `make_crowd_sim` |
| **Recommended hard run** (`--env-config configs/crowdnav_hard.py`, no `--robot-kinematics`) | hard: `holonomic` | **`holonomic`** | CLI infers from file; matches file intent |
| **Hard env + forced unicycle** | hard: `holonomic` | **`unicycle`** | Only if `--robot-kinematics unicycle` |
| **Stage I dataset collect** (`collect_crowdnav_stage1_dataset.py`) | optional; default vendored | **`holonomic`** (CLI default `--kinematics`) | Independent of `run_raise`; not auto-synced with training kinematics |
| **Stage I Score1 in loop** | N/A (dataset/smoke) | **Same as `apply_config` on run** | Prompts/smoke use global `get_settings()` after `pack.apply_config` |
| **Stage III in pipeline** | from `env_config_path` | Same as `robot_kinematics` on run | `_stage3_config` → adapter |
| **`evaluate_crowdnav_generalization.py`** | `crowdnav_generalization_17.py` (holonomic in file) | **`unicycle` hardcoded** in `Stage3Config` | **Mismatch** with env file and with hard-run defaults |

### Cleanup-relevant drift points

1. Three “defaults”: `CrowdNavSettings` (`unicycle`), vendored env file (`holonomic`), `crowdnav_hard` (`holonomic`), Stage I collect CLI (`holonomic`).
2. `pipeline._space_knobs` uses `action_mode or "discrete"` while `RaiseRunConfig` default is `"continuous"`.
3. `ClosedLoopConfig.device` default `cpu` vs `RaiseRunConfig` `cuda` (masked when using pipeline).
4. `Stage2Config.eval_mode` default `"both"` vs run default `"holdout_only"`.
5. `human_num_override` / `env_config_obj` only in tests—no run-level surface.
6. Generalization eval script ignores env-file kinematics.

---

## 2.8 Suggested cleanup themes (for later sections)

- Single kinematics default + document `--env-config` + `--robot-kinematics` interaction.
- Align `StageIConfig` standalone defaults with `defaults.py` or document split.
- Unify `eval_mode` / `device` defaults across stage dataclasses.
- Expose or remove dead fields (`human_num_override` on run config, `env_config_obj`).
- Fix `evaluate_crowdnav_generalization.py` to use run manifest / `settings_from_config` kinematics.
- Consider `_space_knobs` `action_mode` fallback (`"discrete"` → `"continuous"`).

[REDACTED]


---


# Agent dc9f197a-b955-4d39-bcc4-902232fcbb73

## 3. Dead code and CLI surface

### 3.1 Vulture (`min-confidence 80`)

**Command:** `cd F:/Code/New folder/Raise/raise_env; python -m vulture raise_core domains scripts --min-confidence 80`

**Result:** **No findings** (exit code 0, empty output).

At **60%** confidence, vulture reports **42 items**, mostly false positives (SB3 callback hooks, config attributes set for upstream CrowdNav++, pytest hooks, dataclass/protocol stubs). The few **plausibly real** unused symbols (not ≥80%):

| Location | Symbol | Notes |
|----------|--------|--------|
| `F:/Code/New folder/Raise/raise_env/domains/crowdnav/adapter.py` | `eval_act_fn_metrics` | Defined only; **no callers** in repo |
| `F:/Code/New folder/Raise/raise_env/domains/crowdnav/metrics.py` | `OUTCOMES` | Unused constant |
| `F:/Code/New folder/Raise/raise_env/domains/crowdnav/prompts.py` | `PROMPT_MEMORY_EXAMPLE`, `DOMAIN_NAME` | Unused prompt fragments |
| `F:/Code/New folder/Raise/raise_env/raise_core/key_manager.py` | `_current_key` | Unused method |
| `F:/Code/New folder/Raise/raise_env/raise_core/pipeline.py` | `stage2_population` | Unused variable |

Treat 60% hits as **review queue**, not auto-delete.

---

### 3.2 `raise_env/scripts/` inventory

**Index:** `F:/Code/New folder/Raise/raise_env/scripts/README.md` (12 Python CLIs + README).

| Script | Classification | Documented in | Runtime coupling |
|--------|----------------|---------------|------------------|
| `run_raise.py` | **Core workflow** | README, `README_RAISE.md`, root `README.md`, CI/docs | Primary entry |
| `collect_crowdnav_stage1_dataset.py` | **Core workflow** | Same + `domains/crowdnav/data/README.md` | Score1 dataset |
| `print_raise_report.py` | **Core workflow** | `scripts/README.md` | Post-run reporting |
| `plot_raise_run.py` | **Core workflow** | `scripts/README.md` | Imported from `domains/crowdnav/post_run.py` |
| `run_baselines.py` | **Core workflow** | `scripts/README.md`, `README_RAISE.md` | Post-run analysis |
| `run_gamma_sweep.py` | **Core workflow** | Same | Post-run analysis |
| `evaluate_crowdnav_generalization.py` | **Core workflow** | Same | Holdout eval |
| `calibrate_crowdnav_native.py` | **Core workflow** | `README_RAISE.md` | Native-reward calibration |
| `check_orca_reference.py` | **Core workflow** | Same | ORCA sanity |
| `analyze_score1_vs_fitness.py` | **Documented analysis** | `scripts/README.md` | Imported by `raise_core/raise_loop/report.py`; tested in `raise_core/tests/test_analyze_score1.py` |
| `bench_speed.py` | **Dev / profiling** | `scripts/README.md` (“optional”) | Large standalone bench harness; not CI-critical |
| `smoke_crowdnavpp.py` | **Dev / smoke** | `scripts/README.md` | Minimal vendor import check |

**Summary:** No orphan scripts—all are either indexed or imported/tested. Only `bench_speed.py` and `smoke_crowdnavpp.py` are clearly **non-production** (keep for ops/debug).

---

### 3.3 Git-tracked Python caches

**Command:** `git ls-files "**/__pycache__/**" "**/*.pyc" "**/.pytest_cache/**"` from `F:/Code/New folder/Raise`

**Result:** **Empty** — none of these paths are committed.

**Note:** `.gitignore` at repo root and `F:/Code/New folder/Raise/raise_env/.gitignore` already ignore `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.ruff_cache/`. Working tree may still contain untracked cache dirs (see §4.4).

---

## 4. Stray artifacts, vendored leftovers, and owner decisions

### 4.1 `raise_env/third_party/_crowdnavpp_src/`

| Question | Answer |
|----------|--------|
| **What is it?** | A **nested directory shell** with **3 entries**, ending in a **git submodule gitlink** (mode `160000`, commit `390773137be04ed14e27620dc5fd7c5e1a5b1f62`) at `F:/Code/New folder/Raise/raise_env/third_party/_crowdnavpp_src/raise_env/third_party/_crowdnavpp_src`. No source files—only empty nested folders + submodule pointer. |
| **Active vendor?** | **No.** Production vendor is **`F:/Code/New folder/Raise/raise_env/third_party/crowdnavpp/`** (normal `100644` blobs; documented in `UPSTREAM.md` @ same commit). Wired via `raise_paths.CROWDNAVPP_VENDOR_DIR` and `domains/crowdnav/settings.py`. |
| **Code references to `_crowdnavpp_src`?** | **None** (grep across repo: zero matches). |
| **Nested stray?** | **Yes** — path literally nests `raise_env/third_party/_crowdnavpp_src` inside itself. |
| **Git health** | **Broken submodule:** `git submodule status` fails with *“no submodule mapping found in .gitmodules”*; **no `.gitmodules`** in repo. Git **does track** the gitlink: `raise_env/third_party/_crowdnavpp_src/raise_env/third_party/_crowdnavpp_src`. |

**Recommended action:** Remove gitlink + empty tree; rely solely on `third_party/crowdnavpp/` (see §4.5).

---

### 4.2 `results/crowdnav_16h` and run-output references

| Item | Detail |
|------|--------|
| **On disk** | `F:/Code/New folder/Raise/raise_env/results/crowdnav_16h` — **not present** (`Test-Path` → false). Expected: `raise_env/results/` is **gitignored**. |
| **Code reference** | `F:/Code/New folder/Raise/raise_env/domains/crowdnav/tests/test_adapter_train_cleanup.py` — Windows-style path string in a test fixture only. |
| **Docs reference** | `F:/Code/New folder/Raise/docs/NEXT.md` — title *“Post–crowdnav_16h follow-ups”*; body describes implemented CLI flags (not a pointer to a results tree). |
| **Default run dirs** | CLIs use `results/crowdnav_<timestamp>`, `results/crowdnav_run`, etc. (`run_raise.py`, `README_RAISE.md`) — generic, not `crowdnav_16h`. |

**Recommended action:** No repo cleanup for `crowdnav_16h` unless local results exist outside git; optionally rename `docs/NEXT.md` title to drop the run-specific name.

---

### 4.3 Duplicated / overlapping documentation

| Paths | Relationship |
|-------|----------------|
| `F:/Code/New folder/Raise/README.md` vs `F:/Code/New folder/Raise/raise_env/README.md` vs `F:/Code/New folder/Raise/raise_env/README_RAISE.md` | **Layered, not byte-duplicates:** root = repo map; `raise_env/README.md` = short pointer; `README_RAISE.md` = full ops guide. Overlap is **shared quickstart one-liners** (intentional). |
| `F:/Code/New folder/Raise/docs/ARCHITECTURE.md` vs above | Tree diagram + architecture; overlaps `README_RAISE.md` on layout/scripts. |
| `F:/Code/New folder/Raise/raise_env/scripts/README.md` | Canonical script index; aligns with `README_RAISE.md` tables. |
| `docs/NEXT.md` vs implemented flags in code | **Historical naming** (`crowdnav_16h`); content still valid as feature “why” doc. |

**Not duplicates:** `third_party/crowdnavpp/crowd_sim/README.md` (upstream sim doc).

---

### 4.4 Local untracked / gitignored junk (safe to delete locally)

Present in working tree per git status / glob; **not** in `git ls-files` for caches:

- `F:/Code/New folder/Raise/raise_env/**/__pycache__/`, `*.pyc`
- `F:/Code/New folder/Raise/raise_env/.pytest_cache/`
- `F:/Code/New folder/Raise/raise_env/.ruff_cache/`
- `F:/Code/New folder/Raise/raise_env/pytest-cache-files-jq8kyn2v/` (orphan pytest cache dir)
- `F:/Code/New folder/Raise/raise_env/followup_full.txt`, `followup_full2.txt`, `followup_full3.txt`, `followup_pytest.txt` — **pytest session captures**, not referenced by code/docs

---

### 4.5 Ambiguous keep-or-delete (owner decision)

| Item | Path(s) | Recommendation | Owner question |
|------|---------|----------------|----------------|
| **Broken nested submodule gitlink** | `F:/Code/New folder/Raise/raise_env/third_party/_crowdnavpp_src/` | **Delete** from git + filesystem | Confirm no workflow still runs `git submodule update` expecting this path? |
| **`eval_act_fn_metrics`** | `domains/crowdnav/adapter.py` | Delete or wire to tests/callers | Planned public API or dead experiment? |
| **`bench_speed.py`** (~1.9k lines) | `raise_env/scripts/bench_speed.py` | Keep if perf regression tracking matters; else move to `devtools/` or drop | Still used for hardware tuning? |
| **`followup_*.txt`** | `raise_env/followup_*.txt` | Delete | Agent/debug logs only? |
| **`docs/NEXT.md` title** | `F:/Code/New folder/Raise/docs/NEXT.md` | Rename / merge into `README_RAISE.md` or ARCHITECTURE | Keep as changelog-style doc or fold into main docs? |
| **`README` triad** | root + `raise_env/README.md` + `README_RAISE.md` | Keep split or consolidate | Reduce duplication of quickstart blocks? |
| **Upstream CrowdNav++ files not used by RAISE** | e.g. `crowd_sim_pred*.py`, `ros_turtlebot2i_env.py`, `srnn.py` under `third_party/crowdnavpp/` | **Keep** if “unmodified vendor” policy holds; **trim** only with documented UPSTREAM.md diff | Strict MIT vendor snapshot vs minimal import closure? |
| **Local `raise_env/results/`** | gitignored | Delete old run trees locally | Any run dirs (e.g. past `crowdnav_16h`) worth archiving off-machine? |

---

### 4.6 Suggested cleanup commands (reference only)

```powershell
# Local cache / stray logs (does not affect git)
Remove-Item -Recurse -Force "F:/Code/New folder/Raise/raise_env/.pytest_cache", `
  "F:/Code/New folder/Raise/raise_env/.ruff_cache", `
  "F:/Code/New folder/Raise/raise_env/pytest-cache-files-jq8kyn2v", `
  "F:/Code/New folder/Raise/raise_env/followup_*.txt" -ErrorAction SilentlyContinue

# Broken submodule gitlink (after owner OK — requires git commit)
cd "F:/Code/New folder/Raise"
git rm "raise_env/third_party/_crowdnavpp_src/raise_env/third_party/_crowdnavpp_src"
# Remove empty parent dirs if they remain
```

[REDACTED]


---


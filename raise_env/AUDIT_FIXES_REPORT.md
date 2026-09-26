# AUDIT FIXES REPORT

**Date:** 2026-09-26  
**Scope:** Parts 1–4 from the full-repo audit task

---

## PART 1 — Legacy `reward_search/` documentation (no deletion)

### 1.1 / 1.2 Created

- [`domains/crowdnav/runtime/crowd_nav/reward_search/README_LEGACY.md`](domains/crowdnav/runtime/crowd_nav/reward_search/README_LEGACY.md)
- One-line `# LEGACY / FROZEN` header on every top-level `.py` in that directory (25 files), pointing at `raise_core/<same>` when a counterpart exists; CrowdNav-only shims (`dataset.py`, `dsrnn_baseline.py`, `regime.py`, `reporting.py`, `state.py`) note “No current counterpart under raise_core/”.

### 1.3 `test_domain_pack_baseline.py` import — **intentional**

`import crowd_nav.reward_search.prompts as legacy_prompts` is a **deliberate** Algorithm-1 Score1 lock: tests assert `legacy_prompts.D5_SEED_FUNCTION is pack_prompts.D5_SEED_FUNCTION` (and string parity of formatters). Not an accidental leftover. README_LEGACY wording documents this.

### 1.4 Stale path docstrings updated

| File | Now points to |
|------|----------------|
| `raise_core/tests/test_validate.py` | `raise_core/tests/test_validate.py` |
| `raise_core/tests/test_refine.py` | `raise_core/tests/test_refine.py` |
| `domains/crowdnav/tests/test_stage1_real_dataset.py` | `domains/crowdnav/tests/...` (+ comment path) |
| `scripts/run_stage2_smoke.py` | `raise_core/tests/test_refine.py` |
| `scripts/run_stage3_smoke.py` | `raise_core/tests/test_validate.py` |
| `scripts/run_active_learning_step.py` | `raise_core/active_learning/PLAN.md` (exists) |

### 1.5 `raise_env/README.md`

Created with an Architecture note: `raise_core/` is the only live pipeline; legacy shim → `README_LEGACY.md`.

---

## PART 2 — Sandbox timeout kill

### Granularity finding

`run_with_timeout` / smoke timeout is invoked **once per candidate validation** (`RewardValidator.validate_code` → `smoke_test_compute` on a handful of smoke states), **not** per env frame. `SandboxedReward.compute()` has no timeout wrapper.

### Fix chosen: **full process kill** (not orphan-thread stopgap)

- `run_with_timeout` uses `multiprocessing` spawn + `Process.terminate()` / `.kill()`.
- Preferred smoke path: recompile `source_code` inside `_smoke_worker` (picklable).
- Validator passes `source_code=` into `smoke_test_compute`.
- Default `SandboxConfig.timeout_seconds` raised **1.0 → 5.0** (Windows spawn
  cold-start otherwise false-timeouts valid rewards).
- Test: `raise_core/tests/test_sandbox_timeout_kill.py` (3 passed).

---

## PART 3 — Profile consolidation

### What changed

- `raise_core/presets.py`: `CLOSED_LOOP_PROFILES` + `get_closed_loop_profile` + `profile_to_run_raise_argv` for `1h`, `12h`, `18h`, `highway_4h`, `paper_scale` (redirect).
- `scripts/run_raise.py`: strips `--profile`, prepends profile argv (CLI overrides win), `--print-config` dumps args and exits.
- `run_raise_1h.py`: **thin wrapper** → `run_raise.py --profile 1h` (+ stamped output/surrogate paths).
- `run_raise_12h.py` / `run_raise_highway_4h.py`: `PROFILE = dict(CLOSED_LOOP_PROFILES[...])` (orchestration for warm/resume/prereqs kept — moving that into `run_raise.py` would be a larger behavior change than this pass).
- `run_raise_18h.py`: argparse defaults from `CLOSED_LOOP_PROFILES['18h']`.
- `run_raise_paper_scale.py`: unchanged entry; profile redirects to it.

### `--print-config` / help proof (profile 1h)

```text
# Effective flags after --profile 1h (excerpt)
python scripts/run_raise.py --profile 1h --print-config
```

Observed (excerpt): `closed_loop=True`, `stage1_population=4`, `stage1_generations=3`,
`closed_loop_k2=2000`, `k2_unit='gradient_steps'`, `closed_loop_min_labels_gate=4`,
`stage3_use_stub` path via `--stage3-stub`, `regime='without_random'`.

**Caveat:** Full byte-diff of old `run_raise_1h.py --help` vs new wrapper is not identical (wrapper is thinner; help text preserved in the module docstring). Budget fields match the extracted PROFILE.

### Tests referencing wrappers

`test_paper_scale.py` / `test_seed_llm_gate.py` only check script path existence / CI refuse — still valid.

---

## PART 4 — Small fixes

| Item | Action |
|------|--------|
| **4a** | `evidence_block`: cast `n_eval_episodes` via `int(float(...))` → `"8 episodes"` |
| **4b** | Removed `reset_num_timesteps=True` from `model.learn()`; comment explains fresh PPO ⇒ SB3 resets anyway |
| **4c** | Renamed gate to `should_attach_crowdnav_proxy_feedback` (+ legacy alias); docstring: highway attach path does not use it |
| **4d** | Canonical `raise_core.selection.is_highway_metrics`; used from `proxy_feedback.py` and `selection.navigation_scalar_from_dict`. Explore still uses pack-level `_is_highway_pack()` (prompt module / path — different predicate). |

Call sites for metrics-domain check now:

- `raise_core/selection.py` (`is_highway_metrics`, `navigation_scalar_from_dict`)
- `raise_core/raise_loop/proxy_feedback.py` (all highway branches)

---

## Test suites

| Suite | Result |
|-------|--------|
| `raise_core` + `domains/highway` + `domains/crowdnav` (`not slow/live`) | **274 passed**, 1 skipped, 2 deselected |
| `test_sandbox_timeout_kill.py` | 3 passed |

Anything not fully verified: 12h/highway orchestration still lives in the named scripts (PROFILE centralized; not a pure `--profile`-only thin wrap for those two). Paper-scale multi-seed path unchanged. First smoke validation after process switch is slower (~process spawn); budgeted via 5s default timeout.

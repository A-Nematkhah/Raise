# REWARD_COMPONENTS_VERIFICATION.md

Verification of the EUREKA-style `reward_components` feature (Phases 1–4) and the global comprehension-ban side fix. Evidence collected under `results/_reward_components_verification/` on 2026-09-29 unless noted. CrowdNav remains frozen; all product diffs are outside `domains/crowdnav/`.

---

## SECTION 1 — Diff audit against the five resolved decisions

### 1.1 Compatibility shim (bare float → `(float, {})`)

**Decision match:** Highway `SandboxedReward.compute` unpacks via `unpack_reward_return(..., allow_components=True)` (set only by `HighwayRewardValidator`); bare floats become `(v, {})`; RL still receives a float.

Relevant hunk (`git diff HEAD -- raise_core/sandbox/runtime.py`):

```diff
+def unpack_reward_return(
+    value: object,
+    *,
+    allow_components: bool = False,
+) -> Tuple[float, Dict[str, float]]:
+    ...
+    if isinstance(value, (int, float)):
+        return require_finite_float(value), {}
+    if not allow_components:
+        raise RewardSandboxError(...)
+    ...
+    total = require_finite_float(value[0])
+    comps = require_reward_components(value[1])
+    return total, comps

     def compute(self, state: RewardState) -> float:
         ...
-        return require_finite_float(value)
+        total, comps = unpack_reward_return(
+            value, allow_components=self._allow_components
+        )
+        # Stash last step only — never accumulate here (diagnostics callback owns trends).
+        self._last_components = comps
+        return total
```

Env wrapper exposes last-step only (`domains/highway/env_wrapper.py`):

```diff
         reward = float(self.reward_fn.compute(state))
+        comps: Dict[str, float] = {}
+        getter = getattr(self.reward_fn, "last_reward_components", None)
+        if callable(getter):
+            raw = getter()
+            if isinstance(raw, dict):
+                comps = {str(k): float(v) for k, v in raw.items()}
         ...
+        info["raise_reward_components"] = comps
```

**Runtime confirmation** (`pytest …::test_highway_accepts_tuple_dict_and_shim_float` + PPO micro-runs):

- OLD scalar: validates; `compute` → `1.5`; `last_reward_components() == {}`.
- NEW tuple (D5 seed): validates; `compute` → float; components populated (`progress_term` / `collision_penalty` / …).
- PPO 3000 steps: `OFF_bare_float_shim` → `component_history_keys=0`; `ON_tuple_components` → 3 keys with 47 rollout checkpoints (`section4_throughput.txt`).

Pytest excerpt (`section1_pytest_key_tests.txt`):

```
domains/highway/tests/test_eureka_components.py::test_highway_accepts_tuple_dict_and_shim_float PASSED
...
============================= 14 passed in 42.52s =============================
exit=0
```

### 1.2 CrowdNav isolation

```text
$ git diff --stat HEAD -- domains/crowdnav/

(empty — no output)
```

**Match:** No CrowdNav files changed. Isolation is by domain pack wiring (`HighwayRewardValidator(allow_components=True)` only), not by flipping a shared CLI flag.

### 1.3 Every-step return, snapshot-only aggregation

**Match:** Reward path returns components every step into `info`; aggregation only at `RolloutDiagnosticsCallback._on_rollout_end` via `_flush_rollout_component_means`. GT callback component snapshots deferred (not wired).

Env + diagnostics diffs: `results/_reward_components_verification/diff_env_diag.patch` (callback accumulates mid-rollout into `_comp_sums` only; `component_history` grows solely on flush).

**Full test** (`domains/highway/tests/test_eureka_components.py`):

```python
def test_components_not_aggregated_outside_rollout_callback():
    """Env / SandboxedReward keep last-step only; history grows only on flush."""
    v = make_highway_validator(smoke_states=default_smoke_states())
    rw = v.validate_code(D5_SEED_FUNCTION)
    st = default_smoke_states()[0]
    for _ in range(5):
        rw.compute(st)
    assert not hasattr(rw, "component_history")
    last = rw.last_reward_components()
    assert isinstance(last, dict)

    cb = RolloutDiagnosticsCallback(candidate_id="c0", log_path="/tmp/x.jsonl")
    for i in range(4):
        cb._ingest_step_components({"const_term": 1.0, "vary_term": float(i)})
        assert cb.component_history == {}
    means = cb._flush_rollout_component_means()
    assert means["const_term"] == pytest.approx(1.0)
    assert means["vary_term"] == pytest.approx(1.5)
    assert len(cb.component_history["const_term"]) == 1
    for _ in range(2):
        cb._ingest_step_components({"const_term": 1.0})
    cb._flush_rollout_component_means()
    assert len(cb.component_history["const_term"]) == 2
    assert abs(max(cb.component_history["const_term"]) - min(cb.component_history["const_term"])) < 1e-9
```

Pass output:

```
domains/highway/tests/test_eureka_components.py::test_components_not_aggregated_outside_rollout_callback PASSED
```

### 1.4 Runtime-only shape validation (no AST Return matching)

**Match:** Shape checks live in `require_reward_components` / `unpack_reward_return` (runtime/smoke). No AST visitor on `Return` nodes was added. Locals-then-return style passes.

Hunk: see §1.1 `require_reward_components` / `unpack_reward_return` (full file: `diff_runtime.patch`).

**Concrete EUREKA-style candidate** (from `section2_adversarial.txt`):

```python
def compute_reward(state, memory):
    progress_term = float(state.progress)
    speed_term = float(0.08 * state.speed)
    collision_penalty = -20.0 if state.collision else 0.0
    total_reward = progress_term + speed_term + collision_penalty
    reward_components = {
        "progress_term": progress_term,
        "speed_term": speed_term,
        "collision_penalty": collision_penalty,
    }
    return total_reward, reward_components
```

```
=== build_locals_then_return_eureka_style ===
PASS total=4.0 comps={'progress_term': 2.0, 'speed_term': 2.0, 'collision_penalty': 0.0}
```

### 1.5 Global comprehension ban

**Match:** Shared denylist in `ast_policy.py` rejects `ListComp`/`SetComp`/`DictComp`/`GeneratorExp` for both domains.

```diff
+    def visit_ListComp(self, node: ast.ListComp) -> None:
+        self.reasons.append(
+            "list comprehensions are forbidden "
+            "(use an explicit for-loop; unbounded comps are a DoS risk)"
+        )
+        ...
+    def visit_SetComp(...)
+    def visit_DictComp(...)
+    def visit_GeneratorExp(...)
```

`rg` (`section1_comprehension_rg.txt`):

```
58:    def visit_ListComp(self, node: ast.ListComp) -> None:
65:    def visit_SetComp(self, node: ast.SetComp) -> None:
72:    def visit_DictComp(self, node: ast.DictComp) -> None:
79:    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
exit=0
```

**Suite counts after the ban + feature (re-runs 2026-09-29):**

| Suite | Result |
|-------|--------|
| `domains/highway/tests` | **118 passed**, 1 warning (143.15s) |
| `domains/crowdnav/tests` | **79 passed**, 1 warning (4.57s) |
| `raise_core/tests` (includes explore) | **150 passed**, 1 skipped, 1 warning (52.86s) |
| Focused new tests (eureka + comprehension ban + evidence_feedback) | **30 passed** (35.43s) |

---

## SECTION 2 — Sandbox enforcement audit (adversarial cases)

Literal messages from `section2_adversarial.txt` (highway validator smoke):

1. **15 keys:** `FAIL: reward components dict has 15 keys; max allowed is 12`
2. **Non-string key `{1: 2.0}`:** `FAIL: reward component keys must be str, got int`
3. **Nested dict value:** `FAIL: reward component 'a' must be a finite float, got nested dict`
4. **Dict comprehension:** `FAIL: dict comprehensions are forbidden (use an explicit for-loop; unbounded comps are a DoS risk)`
5. **NaN:** `FAIL: reward component 'a' is non-finite: nan`  
   **Inf:** `FAIL: reward component 'a' is non-finite: inf`
6. **Non-numeric total:** `FAIL: compute_reward must return a finite int or float, got str`
7. **Bare float shim:** `PASS total=2.5 comps={}`

---

## SECTION 3 — Rendered prompt and evidence_block audit

### 3.1 D1 example + D5 seed (rendered)

**Episode-memory example as rendered by `format_d1_initial()`** (`section3_d1_rendered.txt`):

```python
def compute_reward(state, memory):
    prev = memory.get("prev_x")
    progress = 0.0 if prev is None else float(state.ego.x) - float(prev)
    memory["prev_x"] = float(state.ego.x)
    total = float(progress)
    return total, {"progress_term": total}
```

Signature line in same render:  
`Signature: exactly one function def compute_reward(state, memory): returning (finite float, dict[str, float]).`

**`D5_SEED_FUNCTION`** (`section3_d5_seed.txt`):

```python
def compute_reward(state, memory):
    """Highway seed: dense progress/speed shaping with collision/off-road costs."""
    collision_penalty = -20.0
    off_road_penalty = -10.0
    speed_coef = 0.08
    progress_coef = 1.0
    if state.collision:
        total = float(collision_penalty)
        return total, {
            "collision_penalty": total,
            "off_road_penalty": 0.0,
            "progress_term": 0.0,
            "speed_term": 0.0,
        }
    if state.off_road:
        total = float(off_road_penalty)
        return total, {
            "collision_penalty": 0.0,
            "off_road_penalty": total,
            "progress_term": 0.0,
            "speed_term": 0.0,
        }
    progress_term = float(progress_coef * state.progress)
    speed_term = float(speed_coef * state.speed)
    total = float(progress_term + speed_term)
    return total, {
        "collision_penalty": 0.0,
        "off_road_penalty": 0.0,
        "progress_term": progress_term,
        "speed_term": speed_term,
    }
```

(Note: seed uses monotonic `speed_coef * state.speed` — no fixed speed threshold.)

### 3.2 Full `evidence_block()` (3 components × 5 checkpoints)

From `section3_evidence_block.txt`:

```
Holdout eval (20 episodes): SR=0.45 CR=0.35 TR=0.05 mean_speed=23.4m/s speed_p10=19.0 progress=95m lane_change_rate=0.120 overtakes_per_km=0.80 overtake_episode_frac=0.25
Pareto rank: 2/8 (front-relative; lower is better; feasible=True; front=0)
Feasibility this run: survival only (SR > 0 on holdout). No speed floor and no CR/TR ceilings — Pareto objectives: -CR, -TR, progress, mean_speed, lane_change_rate, overtakes_per_km.
Reachable action speed range this run: [20.0, 30.0] m/s (from action config; not a preferred target)
Score1=0.612
Reward component trends (this candidate's own reward function):
collision_penalty: ['0.00','0.00','-2.50','0.00','0.00'], Max: 0.00, Mean: -0.50, Min: -2.50
progress_term: ['0.40','0.55','0.52','0.61','0.58'], Max: 0.61, Mean: 0.53, Min: 0.40
speed_term: ['0.12','0.31','0.30','0.34','0.28'], Max: 0.34, Mean: 0.27, Min: 0.12
```

Confirmations: ground-truth Holdout/Pareto/Score1 still present; one `Reward component trends` section; keys sorted alphabetically; EUREKA `Max/Mean/Min` format; no truncation.

### 3.2.1 Clarification — feasibility / Pareto lines (follow-up)

The Holdout / Pareto / Feasibility / “Reachable action speed range” lines above are **not** from this reward_components task.

(a) **Already implemented before this task:** primarily commit `8eb45dc` (2026-09-28, survival-only feasibility + overtake Pareto objectives), with earlier evidence-shape neighbors in `1fbdc55` / `0f6d981`.

(b) **No dedicated Pareto/feasibility verification report** exists for that earlier work (related: `AUDIT.md`, `AUDIT_FIXES_REPORT.md`, `docs/PIPELINE_AUDIT.md`, plus unit tests `test_calibration_modes.py`, `test_overtake_objectives.py`, `test_seed_no_speed_threshold.py`, `test_pareto_*`). Track separately if needed — **outstanding, out of scope here**.

(c) **This task did not alter that logic.** Uncommitted `git diff HEAD -- raise_core/raise_loop/proxy_feedback.py` only inserts `format_reward_component_trends(...)` after Score1 and adds that helper; `_feasibility_lines` appears only as unchanged context.

### 3.3 D2 mutation + EUREKA principles

Full render: `section3_d2_mutation_full.txt` (includes evidence above + diagnosis block).

Three principle lines (verbatim):

```
- If the survival/success rate is always near zero, rewrite the entire reward function.
- If a component's value is nearly identical across all checkpoints, RL was not able to optimize it as written — consider changing its scale/temperature, rewriting it, or discarding it.
- If one component's magnitude is significantly larger than the others, rescale it to a comparable range.
```

Grep of **those three lines only** for `speed|traffic|lane|highway|collision|overtake`: **NONE**.

Honest note: the *preceding* diagnosis bullet still gives a highway-flavored example (`safety/speed trade-off`); that text is outside the three EUREKA principle bullets.

---

## SECTION 4 — Throughput re-check (PPO + RolloutDiagnosticsCallback)

Short real PPO trains, `n_steps=64`, CPU, **3000** env steps, callback attached (`section4_throughput.txt`):

| Mode | wall | steps/s | component_history |
|------|------|---------|-------------------|
| **ON** `(float, dict)` + ingest/flush | 123.730s | **24.2** | 3 keys, 47 rollout means |
| **OFF** bare float shim (empty comps) | 128.621s | **23.3** | 0 keys |

ON was not slower than OFF in this run (noise / Windows scheduling). Absolute ~24 steps/s is env-bound; component plumbing did not create a visible PPO wall-clock regression vs the Phase 0 prediction that hundreds of ns on the reward path are negligible vs ms-scale steps.

---

## SECTION 5 — Synthetic plateau detection

### 5.1 Reward source

```python
def compute_reward(state, memory):
    """One component fixed; one varies with speed."""
    fixed_term = 0.42
    speed_term = float(0.05 * state.speed)
    collision_penalty = -20.0 if state.collision else 0.0
    total = fixed_term + speed_term + collision_penalty
    return total, {
        "fixed_term": fixed_term,
        "speed_term": speed_term,
        "collision_penalty": collision_penalty,
    }
```

### 5.2 Component trends (flat `fixed_term`)

From `section5_plateau.txt`:

```
fixed_term: values all 0.42 → Max=Mean=Min=0.42
speed_term: 0.82 → 1.22 across checkpoints (Max≠Min)
```

Evidence block excerpt:

```
Holdout eval (20 episodes): SR=0.55 CR=0.30 TR=0.05 mean_speed=24.0m/s ...
Reward component trends (this candidate's own reward function):
collision_penalty: ['-4.00','-4.00','-4.00','-4.00','-4.00'], Max: -4.00, Mean: -4.00, Min: -4.00
fixed_term: ['0.42','0.42','0.42','0.42','0.42'], Max: 0.42, Mean: 0.42, Min: 0.42
speed_term: ['0.82','0.92','1.02','1.12','1.22'], Max: 1.22, Mean: 1.02, Min: 0.82
```

### 5.3 Distinguishability

In the same block, holdout **SR/CR** are ordinary task metrics (`SR=0.55 CR=0.30`), while **`fixed_term` is visibly flat (max≈mean≈min)** and **`speed_term` is not** — so component plateaus are separable from task-level numbers, not a duplicate of SR/CR.

Unit test: `test_constant_component_flat_in_trend_summary PASSED`.

---

## SECTION 6 — Live run with a real LLM

**Completed** 2026-09-30 (not `--llm seed`). Artifacts: `results/_reward_components_live_verify/`; extract: `results/_reward_components_verification/section6_live_extract.txt`; log: `live_run.log`.

```text
python scripts/run_raise.py --domain highway --llm groq --device cuda --closed-loop \
  --stage1-population 4 --stage1-generations 3 --closed-loop-k2 3000 \
  --closed-loop-proxy-feedback --closed-loop-proxy-d3 1 \
  --closed-loop-evolve-rank pareto --stage3-stub \
  --output-dir results/_reward_components_live_verify --no-resume
```

**403 diagnosis (why the first attempt failed):** from this egress IP, direct `POST /chat/completions` sometimes returned `{"error":{"message":"Forbidden"}}` while keys themselves were valid. The successful §6 live run used an external network path (local VPN/proxy outside the repo). No in-repo HTTP_PROXY / explicit httpx proxy client is required or retained for normal runs — turn on VPN if Groq 403 / connection errors reappear.

**Run outcome:** closed-loop finished successfully (~33 min wall). Epochs 0–2 labeled; final pick Stage II/III `mut_0019` (SR=1.0). Gen0 had many `getattr`/truncation rejects and some seed fallbacks (`see_*`); later epochs produced real LLM crossover/mutation/D3 (`cro_*`, `mut_*`, `cro_0017_d3`).

### 6.1 Real component names (≥3 candidates, ≥2 generations)

From trained rollouts + returned component dicts (extract file):

| candidate | parents | component keys observed |
|-----------|---------|-------------------------|
| `cro_0017` | mut_0016, mut_0015 | `collision_penalty`, `off_road_penalty`, `progress_term`, `speed_term`, `survival_bonus` |
| `cro_0018` | mut_0016, mut_0015 | same five |
| `mut_0019` | cro_0013 | above + `lane_change_term`, `overtake_term`, `proximity_penalty` |
| `mut_0020` | cro_0014 (**verified** `parent_ids`) | `collision_penalty`, `off_road_penalty`, `progress_term`, `speed_term`, `proximity_penalty`, `lane_change_term`, `overtake_term` |

Diagnostics show non-constant trends for several keys (e.g. `mut_0019` `progress_term` 4.28→4.92 across rollouts; `collision_penalty` −1.64→−0.23).

### 6.2 Diagnosis naming a component + EUREKA principle

Verbatim `llm_diagnosis` for `mut_0020` (names **collision penalty** / **proximity penalty**; applies **principle 3 — magnitude/rescale**):

> The parent reward heavily favors forward progress while giving only a modest penalty for collisions, so the agent learns to chase distance at the expense of safety, resulting in a 100 % collision rate and zero survival. Increasing the magnitude of the collision penalty (and the proximity penalty) and giving a slightly stronger weight to speed will shift the trade-off toward safer, faster driving without changing the overall structure of the reward.

Also `mut_0019` names collision/proximity penalties and says keep other terms unchanged (principle-2 flavor: leave non-problem terms alone / flat shaping). Note: diagnoses usually say “collision penalty” in prose, not the exact identifier `collision_penalty`.

### 6.3 Before/after mutation editing a named component

**Lineage note:** §6.2’s diagnosis belongs to `mut_0020` with **`parent_ids=['cro_0014']`**. The D3 example below is a **different** lineage (`cro_0017` → `cro_0017_d3`) and must not be mixed with that diagnosis for faithfulness scoring (see §7.2 item 10 correction).

D3 refine `cro_0017` → `cro_0017_d3` (this pair’s own diagnosis called out tiny collision vs large progress; mutation raised the collision term):

```diff
-    collision_penalty = -30.0
-    off_road_penalty = -15.0
-    progress_coef = 1.0
-    speed_coef = 0.1
+    COLLISION_PENALTY = -100.0
+    OFF_ROAD_PENALTY = -80.0
+    PROGRESS_COEF = 0.5
+    SPEED_COEF = 0.05
...
-        total = float(collision_penalty)
+        total = float(COLLISION_PENALTY)
...
-    progress_term = float(progress_coef * state.progress)
+    progress_delta = max(state.progress - last_progress, 0.0)
+    progress_term = PROGRESS_COEF * progress_delta
```

Full unified diff in `section6_live_extract.txt`.

---

## SECTION 7 — Test suite and honest caveats

### 7.1 Suite output (summaries; full logs under `results/_reward_components_verification/`)

**Focused new tests** (`suite_new_focused.txt` / prior verbose):

```
30 passed in 35.43s
```

(Includes `test_eureka_components.py`, `test_sandbox_comprehension_ban.py`, `test_evidence_feedback.py`.)

**Highway full suite** (`suite_highway_rerun.txt`):

```
118 passed, 1 warning in 143.15s (0:02:23)
```

**raise_core / explore suite** (`suite_raise_core_rerun.txt`):

```
150 passed, 1 skipped, 1 warning in 52.86s
```

**CrowdNav full suite** (`suite_crowdnav_rerun.txt`):

```
79 passed, 1 warning in 4.57s
```

Adversarial + key eureka tests also recorded in `section1_pytest_key_tests.txt` / `section2_adversarial.txt`.

### 7.2 Not fully verified / residual risks

1. **Live Groq closed-loop (§6)** completed after egress to Groq was available (VPN/network outside the repo). Direct chat POSTs from a blocked egress can still return **403 Forbidden** — turn on VPN if that happens. Diagnoses name components mostly in prose (“collision penalty”), not always as exact dict keys. Gen0 still wasted attempts on `getattr`/truncation; some early slots fell back to seed variants.
2. **`MAX_REWARD_COMPONENT_KEYS = 12`** was chosen as a bound from the task brief, **not** empirically tuned against LLM outputs.
3. **Bare-float shim has no expiry plan** — it persists indefinitely for highway resume/checkpoints; prompts push the tuple contract, but old genomes still validate.
4. **Phase 0 float call-site table:** Score1 / surrogate / `env_wrapper` remain safe because `SandboxedReward.compute` still returns a **float**. CrowdNav never sets `allow_components`. Residual risk: any third-party code that calls the raw `compute_fn` (bypass wrapper) would see a tuple on highway — not observed in-repo.
5. **GroundTruthCheckpointCallback** does **not** snapshot components (deferred by design); only rollout-end aggregation is live.
6. **Diagnosis prompt item (1)** still mentions `safety/speed` as an example *outside* the three generic EUREKA principles — principles themselves are clean.
7. **Throughput ON vs OFF** was a single paired 3k-step CPU run; not a multi-seed statistical comparison. Absolute SPS is low (~24) due to highway-env cost, not components.
8. **CrowdNav isolation** verified by empty `git diff --stat domains/crowdnav/` **and** CrowdNav suite green + `test_crowdnav_rejects_tuple_return` / comprehension-ban CrowdNav cases — not merely assumed.
9. Prompt fix applied during verification: D1 example braces escaped (`{{`/`}}`) so `.format()` works; also wired `_RETURN_CONTRACT` into `D1_SYSTEM_PROMPT` and clarified the D1 task description for the tuple return.
10. **Diagnosis↔diff lineage hygiene (correction of a prior mispaired claim).** An earlier draft of this item compared §6.2’s `mut_0020` diagnosis to the §6.3 `cro_0017` → `cro_0017_d3` diff. That comparison is **invalid**: verified `mut_0020.parent_ids == ['cro_0014']` (`origin="mutation"`), not `cro_0017`. **Withdrawn:** the claim that “proximity penalty” was absent from the parent, and that speed/structure claims contradicted the mutation, when those contradictions came from the wrong lineage (`cro_0017` has no `proximity_penalty` and the D3 diff shrinks `speed_coef` / rewrites progress). **Corrected check against `cro_0014` → `mut_0020`:**
    - **Parent source:** full `cro_0014` reward **code was not retained** in any checkpoint/snapshot/population JSON (snapshotter started after that candidate left the live population). Recoverable evidence: `surrogate_dataset/features.jsonl` (`code_len=2455`, `code_hash=b7860151…`) and Stage-II `diagnostics_rollout.jsonl` for `r1002_cro_0014`.
    - **Parent components (from diagnostics):** `collision_penalty`, `off_road_penalty`, `progress_term`, `speed_term`, **`proximity_penalty`**, `lane_change_term`, `overtake_term`. So naming “proximity penalty” in the diagnosis is **in-vocabulary for the true parent**, not a hallucination relative to `cro_0014`.
    - **Child `mut_0020`:** full source retained; same seven component keys; `speed_coef = 0.1`; collision/off-road constants `-80` / `-40`; docstring says same shaping terms as parent with stronger safety and slightly higher speed incentive.
    - **Directional claims:** same component key set parent↔child supports “overall structure” at the component-dict level. Stronger collision magnitude is **consistent** with more-negative `collision_penalty` rollout means (parent ≈ −0.12…−0.82 vs child ≈ −1.25…−1.56). “Slightly stronger weight to speed” is **not coef-verified** without parent source (child `speed_term` means are higher, but policy-confounded). `proximity_penalty` means were ~0 for both during logged rollouts — magnitude increase of that term is **not** numerically confirmed from diagnostics alone.
    - **Net:** the previous “diagnosis was unfaithful” conclusion does **not** stand for the correct pair; available evidence is compatible with a partially faithful component-aware edit, with coef-level proof incomplete because the parent body is missing. Full before/after unified diff for this pair cannot be pasted.
11. **Data-hygiene rule for future reports:** whenever a diagnosis-vs-diff (or diagnosis-vs-code) comparison is presented, **explicitly state and verify** `child.parent_ids` (and `origin`) **before** presenting the comparison. Do not juxtapose a diagnosis from one candidate with a diff from another lineage.
12. **Recommended follow-on metric (do not implement now):** across a larger sample of **verified** `(parent, child)` pairs (linked by `parent_ids`), measure automatically (1) what fraction of diagnosis mentions name a component that exists in the **parent’s** recorded `reward_components` / trend keys, and (2) what fraction of directional claims match the sign of change in a retained before/after diff (or, if only diagnostics exist, in comparable component means — with policy confounding noted). Also **retain parent source code** (or hash-addressable archive) across generations so coef-level checks are possible. Report a **faithfulness rate**; this remains the right rigorous test, but only on correctly paired lineages.

---

*Artifacts directory: `raise_env/results/_reward_components_verification/` (see also `sec6_corrected_pair_report.txt`).*

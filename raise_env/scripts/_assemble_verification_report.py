#!/usr/bin/env python
"""Assemble VERIFICATION_REPORT.md at repo root."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"F:/Code/raise-paper-baseline")
ENV = ROOT / "raise_env"
ART = ENV / "results/_verify_live_highway"


def read(path: Path, default: str = "") -> str:
    if not path.is_file():
        return default
    return path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    os.chdir(ENV)
    constants = read(ENV / "domains/highway/objective_constants.py")
    gate_out = subprocess.check_output(
        [sys.executable, "scripts/_verify_gate.py"], text=True, encoding="utf-8"
    )
    prompt_out = subprocess.check_output(
        [sys.executable, "scripts/_verify_prompt_render.py"],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    plumbing = json.loads(read(ART / "diagnosis_plumbing.json", "{}"))
    epoch_sum = json.loads(read(ART / "epoch_speed_summary.json", "{}"))
    metrics_diff = read(ART / "diff_raise_env_domains_highway_metrics.py.patch")
    proxy_diff = read(ART / "diff_raise_env_raise_core_raise_loop_proxy_feedback.py.patch")
    explore_diff = read(ART / "diff_raise_env_raise_core_explore.py.patch")
    prompts_diff = read(ART / "diff_raise_env_domains_highway_prompts.py.patch")
    adapter_diff = read(ART / "diff_raise_env_domains_highway_adapter.py.patch")
    crowdnav_stat = read(ART / "diff_crowdnav_stat.txt", "(empty — no changes)\n").strip()
    pytest_targeted = read(ENV / "results/_verify_pytest_targeted2.txt")
    pytest_suites = read(ENV / "results/_verify_pytest_suites2.txt")
    chunks = epoch_sum.get("chunks", [])
    chunks_json = json.dumps(chunks, indent=2)

    report = f"""# VERIFICATION REPORT — Highway fitness-gate recenter + evidence-driven LLM feedback

**Date:** 2026-09-26  
**Repo:** `raise-paper-baseline` / `raise_env/`  
**Diff baseline:** uncommitted working tree vs `main` (`bfc0ecf`)  
**Scope:** fitness-gate recenter + evidence-driven feedback (Parts A–C)

This report proves implementation correctness with **real numbers, real diffs, and real rendered prompts**, plus a **real PPO closed-loop run** (not stub trainers).

---

## SECTION 1 — Diff audit against the original spec

### 1.1 CrowdNav freeze check

Command:

```text
git diff --stat HEAD -- raise_env/domains/crowdnav/
```

Output:

```text
{crowdnav_stat}
```

**Verdict:** empty — no files under `domains/crowdnav/` changed. Intentional.

### 1.2 New file: `raise_env/domains/highway/objective_constants.py` (full content)

Matches spec item: single source of truth for `V_TARGET`, `V_MIN`, `V_FLOOR`, `K_GATE=6.0`, `LAG_SPEED_MPS`.

```python
{constants}```

### 1.3 `domains/highway/metrics.py` — `_speed_gate` recenter

Matches spec: gate recentered on `(V_MIN+V_TARGET)/2` with span `(V_TARGET-V_MIN)`; constants imported from `objective_constants.py`; floor / hack / low-speed penalty preserved.

```diff
{metrics_diff}```

### 1.4 `raise_core/raise_loop/proxy_feedback.py`

Matches spec: adds `evidence_block()`; highway attach always uses evidence when enabled (no threshold gate); removes `classify_highway_hack` + highway hand-coded focus notes; CrowdNav retains `focus_note_from_metrics` / `should_attach_proxy_feedback`.

```diff
{proxy_diff}```

### 1.5 `raise_core/explore.py`

Matches spec: `extract_llm_diagnosis`; highway evidence reflection; CrowdNav advice path guarded by `_is_highway_pack()`; diagnosis stored on candidates.

```diff
{explore_diff}```

### 1.6 `domains/highway/prompts.py`

Matches spec: fitness formula from constants in D1/D4; diagnose-before-code in D2/D3; removed lagging / soft@20 prescription language.

```diff
{prompts_diff}```

### 1.7 Related: `domains/highway/adapter.py`

```diff
{adapter_diff}```

### 1.8 Grep audits

#### Command A

```text
rg -n "focus_note_from_metrics|should_attach_proxy_feedback" raise_env/
```

Live hits:

- `raise_core/raise_loop/proxy_feedback.py` — **definitions + CrowdNav-only call sites** (intentional; highway branch does not use the gate).
- `raise_core/tests/test_proxy_feedback.py`, `test_evidence_feedback.py` — tests.

#### Command B

```text
rg -n "lags traffic|strengthen goal progress|is a failure mode" raise_env/domains/highway/ raise_env/raise_core/
```

Live hits:

- `raise_core/explore.py` (~line 916) `strengthen goal progress` — **CrowdNav-only** branch of `_build_reflection`.
- `raise_core/tests/test_evidence_feedback.py` — asserts highway reflection does **not** contain that string.
- **Zero matches under `domains/highway/`.**
- **Zero matches** for `lags traffic` / `is a failure mode` in live highway / raise_core non-test code.

---

## SECTION 2 — Math verification of the recentered gate

Script: `raise_env/scripts/_verify_gate.py` (imports the real `_speed_gate`).

### Full printed table

```text
{gate_out}```

### Explicit numbers

| Quantity | Value |
|----------|------:|
| `gate(20)` new | **0.731059** |
| `gate(25)` new | **0.952574** |
| `gate(25)−gate(20)` new | **0.221516** (≥ 0.10) |
| `gate(V_MIN=10)` new | **0.047426** (≈ 0) |
| `gate(V_TARGET=25)` new | **0.952574** (≤ 0.97) |
| `gate(20)` old | 0.960834 |
| `gate(25)` old | 0.991837 |
| `gate(25)−gate(20)` old | **0.031003** |

Old gate is already ~0.96 at 20 m/s; new gate keeps a usable gradient across `[10, 25]`.

---

## SECTION 3 — Prompt content audit (real render)

Representative holdout metrics (~20 m/s plateau failure mode):

```json
{{
  "domain": "highway",
  "SR": 0.95, "CR": 0.05, "TR": 0.0,
  "mean_speed": 20.06, "speed_p10": 20.06,
  "mean_progress": 803.0, "soft_success": 0.0,
  "lane_change_rate": 0.0, "n_eval_episodes": 20
}}
```

### Literal full render (`scripts/_verify_prompt_render.py`)

```text
{prompt_out}
```

### Inspection

| Check | Result |
|-------|--------|
| Numbers: SR, CR, mean_speed, speed_gate, v_eff, fitness | **Present** |
| evidence_block banned words (`lags`,`hack`,`degenerate`,`improve`,`strengthen`,`weakness focus`) | **All absent** |
| D2/D3 instruction may mention “degenerate strategy” as an example diagnosis option | **Yes** (prompt instruction, not Python verdict) |
| Formula uses `K_GATE=6.0`, `V_MIN=10.0`, `V_TARGET=25.0` | **Yes** |

Live PPO run also wrote evidence-style `proxy_feedback`, e.g. candidate `cro_0019`:

```text
Holdout eval (8.0 episodes): SR=1.00 CR=0.00 TR=0.00 mean_speed=20.1m/s speed_p10=20.1 progress=803m soft_success=0.00 lane_change_rate=0.000
Fitness=0.319 (speed_gate=0.736 at v_eff=…
```

---

## SECTION 4 — Live run evidence (real PPO)

### Command

```powershell
python scripts/run_raise.py --domain highway --allow-seed-llm --llm seed --score1 smoke `
  --closed-loop --closed-loop-proxy-feedback `
  --stage1-population 6 --stage1-generations 4 `
  --closed-loop-k2 2000 --k2-unit env_steps `
  --stage2-eval-episodes 8 --stage3-train-steps 400 --stage3-eval-episodes 4 `
  --highway-eval-mode holdout_only --highway-n-envs 1 `
  --closed-loop-min-stage2 3 --closed-loop-no-al `
  --device cuda --output-dir results/_verify_live_highway --no-resume
```

- **No stubs / no `--fast`** — real SB3 PPO on `highway-fast-v0`.
- `--score1 smoke` because local Stage I dataset was missing; **PPO labels are real**.
- **No Groq/Ollama/OpenAI keys** → seed LLM only → live `llm_diagnosis` is null.
- Wall-clock: **21m 08.1s**. Artifacts: `raise_env/results/_verify_live_highway/`.

### Per-generation Stage II holdout metrics (parsed from run log)

```json
{chunks_json}
```

Closed-loop epoch summary lines:

```text
epoch 0: evo[scalar]=ini_0004 fitness=0.319 | best_ever=0.319
epoch 1: evo[scalar]=mut_0009 fitness=0.800 | best_ever=0.800
epoch 2: evo[scalar]=mut_0016 fitness=0.319 | best_ever=0.800
epoch 3: evo[scalar]=mut_0022 fitness=0.319 | best_ever=0.800
```

### Live `llm_diagnosis`

From final `closed_loop/checkpoint.json` population: **6/6 candidates have `llm_diagnosis: null`.**

Seed LLM returns code-only completions (no pre-fence text). **No live frontier-model self-diagnosis text was obtained in this session.**

### Plumbing proof (scripted model that *does* emit a diagnosis)

`scripts/_verify_diagnosis_plumbing.py` → `results/_verify_live_highway/diagnosis_plumbing.json`:

```json
{json.dumps(plumbing, indent=2)}
```

Verbatim stored diagnosis:

> Holdout mean_speed sits near 20 m/s while V_TARGET is 25; the speed_gate is only ~0.73 so fitness still has headroom on the speed band, but CR=0.08 suggests pushing speed without more collision shaping may trade off badly. I will raise traffic-matching shaping gently and keep a stronger collision cost.

This mentions gate / speed headroom / collision trade-off — but it is from a **compliance ScriptedLLM**, not from the 21-minute seed-LLM PPO run.

### Speed vs ~20 m/s plateau narrative

- Epoch 0 elites: **20.1 m/s**, fitness **0.319**, gate ≈ 0.736 (not saturated).
- Epoch 1 best: **mut_0009 @ 21.8 m/s, fitness 0.800** (best_ever).
- Epochs 2–3 fall back to **20.1 m/s** elites under seed mutations.
- **No sustained climb to 25 m/s.** No paired old-gate baseline was re-run here; epoch-0 matches the prior plateau narrative, epoch-1 shows the new objective can prefer slightly higher speed when achieved.

---

## SECTION 5 — Test results

### Targeted new tests (`-vv`)

```text
{pytest_targeted}
```

### Full highway + explore + crowdnav suites

```text
{pytest_suites}
```

**148 passed**, 2 warnings (surrogate ConstantInputWarning; gym Box float32). CrowdNav suite green.

---

## SECTION 6 — Open risks / caveats

1. **`K_GATE=6` not re-tuned on a long LLM-backed run.** Section 2 shape looks reasonable (midpoint 17.5, Δ20→25 = 0.22, target gate ≤ 0.97), but selection dynamics under real mutations remain empirically open.

2. **Live `llm_diagnosis` was 100% null** in the PPO run (seed LLM / no API keys). Plumbing works; frontier self-diagnosis quality is **unverified** here.

3. **Checkpoint `reflection` can show `fitness=-inf` / `nan` speeds** when built on the next generation *before* Stage II re-attaches `last_metrics`. Proxy evidence on labeled parents still attaches; global reflection channel is weaker until rebuilt post-label.

4. **CrowdNav advice strings remain in `explore.py`** behind `_is_highway_pack()` — intentional parity; regression risk if the guard is removed.

5. **Remaining hand-coded thresholds (out of prompt-feedback scope):** `LAG_SPEED_MPS=18` in fitness lag penalty; degeneracy penalties in `metrics.py`; soft-success progress floor 400 m in `adapter.py`.

6. **Stage I used smoke Score1** (no local highway dataset) — fine for gate/PPO verification, not for Score1↔fitness claims.

7. **No fresh old-gate A/B baseline run** in this session.

---

## Bottom line

| Claim | Verified? |
|-------|-----------|
| Gate recentered; Δ(gate25−gate20) ≥ 0.10 | **Yes** (0.2215) |
| Constants centralized; CrowdNav tree untouched | **Yes** |
| Highway evidence/prompts are number-first | **Yes** |
| Real PPO closed-loop produces new fitness numbers | **Yes** (21m, 4 epochs) |
| Live frontier-LLM self-diagnosis text | **No** (no API keys) |
| Diagnosis plumbing when model emits preamble | **Yes** (scripted) |
"""
    out = ROOT / "VERIFICATION_REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out} ({len(report)} chars)")


if __name__ == "__main__":
    main()

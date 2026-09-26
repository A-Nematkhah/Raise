# VERIFICATION REPORT - Highway fitness-gate recenter + evidence-driven LLM feedback

**Date:** 2026-09-26  
**Repo:** `raise-paper-baseline` / `raise_env/`  
**Diff baseline:** uncommitted working tree vs `main` (`bfc0ecf`)  
**Scope:** fitness-gate recenter + evidence-driven feedback (Parts A-C)

This report proves implementation correctness with **real numbers, real diffs, and real rendered prompts**, plus a **real PPO closed-loop run** (not stub trainers).

---

## SECTION 1 - Diff audit against the original spec

### 1.1 CrowdNav freeze check

Command:

```text
git diff --stat HEAD -- raise_env/domains/crowdnav/
```

Output:

```text
(empty - no changes)
```

**Verdict:** empty - no files under `domains/crowdnav/` changed. Intentional.

### 1.2 New file: `raise_env/domains/highway/objective_constants.py` (full content)

Matches spec item: single source of truth for `V_TARGET`, `V_MIN`, `V_FLOOR`, `K_GATE=6.0`, `LAG_SPEED_MPS`.

```python
"""Single source of truth for highway fitness / soft-success speed scales.

Import these constants everywhere (metrics, prompts, proxy feedback). Do not
redefine or hand-pick parallel literals downstream.
"""

from __future__ import annotations

# Nominal traffic / cruise target (m/s). Soft-success and fitness speed terms
# are expressed relative to this value.
V_TARGET = 25.0

# Below this, survival terms are gated down (m/s).
V_MIN = 0.4 * V_TARGET  # 10.0

# Hard disqualify crawl / parked hack (m/s).
V_FLOOR = 0.15 * V_TARGET  # 3.75

# Sigmoid steepness for the recentered speed gate spanning [V_MIN, V_TARGET].
K_GATE = 6.0

# Lag penalty kicks in when mean_speed is below this (m/s).
LAG_SPEED_MPS = 18.0
```

### 1.3 `domains/highway/metrics.py` - `_speed_gate` recenter

Matches spec: gate recentered on `(V_MIN+V_TARGET)/2` with span `(V_TARGET-V_MIN)`; constants imported from `objective_constants.py`; floor / hack / low-speed penalty preserved.

```diff
diff --git a/raise_env/domains/highway/metrics.py b/raise_env/domains/highway/metrics.py
index bb6f7d0..d3cb627 100644
--- a/raise_env/domains/highway/metrics.py
+++ b/raise_env/domains/highway/metrics.py
@@ -13,21 +13,40 @@ from __future__ import annotations
 import math
 from typing import Any, Dict, Mapping, Optional
 
+from domains.highway.objective_constants import (
+    K_GATE,
+    LAG_SPEED_MPS,
+    V_FLOOR,
+    V_MIN,
+    V_TARGET,
+)
+
+# Re-export so existing ``from domains.highway.metrics import V_TARGET`` keeps working.
+__all__ = [
+    "V_TARGET",
+    "V_MIN",
+    "V_FLOOR",
+    "K_GATE",
+    "LAG_SPEED_MPS",
+    "HACK_PENALTY",
+    "highway_fitness",
+    "highway_navigation_scalar",
+    "fitness_components",
+    "attach_fitness",
+    "attach_selection_scalar",
+    "format_highway_metrics_line",
+]
+
+# Alias kept for callers that imported the private name.
+_LAG_SPEED_MPS = LAG_SPEED_MPS
 
 # Reference scales for ~40s highway-fast episodes at traffic cruise.
 _REF_PROGRESS_M = 800.0  # ~20 m/s * 40 s
-_REF_SPEED_MPS = 25.0
+_REF_SPEED_MPS = float(V_TARGET)
 
-# Speed gate (relative to target cruise).
-V_TARGET = 25.0  # m/s - nominal traffic / cruise target
-V_MIN = 0.4 * V_TARGET  # 10 m/s - below this, survival terms are gated down
-V_FLOOR = 0.15 * V_TARGET  # 3.75 m/s - hard disqualify (crawl / parked hack)
-K_GATE = 8.0  # sigmoid steepness around v_min
 HACK_PENALTY = 10.0  # magnitude when v_eff < v_floor (F = −HACK_PENALTY)
 LOW_SPEED_WEIGHT = 0.50  # weight on (v_min − v_eff)+ / v_target
 
-# Lag vs surrounding traffic (~≥20 m/s historically; soft@25 uses V_TARGET).
-_LAG_SPEED_MPS = 18.0
 # Constant-cruise detector extras (progress_std ≈ 0 + flat speed band).
 _CRUISE_PROGRESS_STD_MAX = 1.0
 _CRUISE_SPEED_SPREAD_MAX = 0.35
@@ -114,8 +133,10 @@ def _v_eff(src: Mapping[str, Any]) -> float:
 
 
 def _speed_gate(v_eff: float) -> float:
-    """sigmoid(k · (v_eff − v_min) / v_target) ∈ (0, 1)."""
-    return _sigmoid(K_GATE * (float(v_eff) - V_MIN) / V_TARGET)
+    """Continuous gate spanning V_MIN → V_TARGET (not saturated near V_MIN)."""
+    center = (V_MIN + V_TARGET) / 2.0
+    span = max(1e-6, V_TARGET - V_MIN)
+    return _sigmoid(K_GATE * (float(v_eff) - center) / span)
 
 
 def _low_speed_penalty(v_eff: float) -> float:
@@ -127,8 +148,8 @@ def _low_speed_penalty(v_eff: float) -> float:
 
 
 def _lag_penalty(sr: float, mean_speed: float) -> float:
-    if sr >= 0.5 and mean_speed < _LAG_SPEED_MPS:
-        return 0.40 * (_LAG_SPEED_MPS - mean_speed) / _LAG_SPEED_MPS
+    if sr >= 0.5 and mean_speed < LAG_SPEED_MPS:
+        return 0.40 * (LAG_SPEED_MPS - mean_speed) / LAG_SPEED_MPS
     return 0.0
 
 
@@ -141,11 +162,12 @@ def highway_fitness(metrics: Optional[Mapping[str, Any]]) -> float:
         v_eff = speed_p10 if present else mean_speed
         if v_eff < v_floor:  return −hack_penalty   # hard disqualify
 
-        gate = sigmoid(k · (v_eff − v_min) / v_target)
+        center = (v_min + v_target) / 2
+        gate = sigmoid(k · (v_eff − center) / (v_target − v_min))
 
         F = gate · (SR − CR − 0.5·TR)
           + 0.35 · tanh(progress / 800)
-          + 0.25 · tanh(mean_speed / 25)
+          + 0.25 · tanh(mean_speed / v_target)
           + 0.15 · soft_success · gate
           − lag_penalty − degeneracy_penalty − low_speed_penalty
     """
@@ -200,6 +222,10 @@ def fitness_components(metrics: Optional[Mapping[str, Any]]) -> Dict[str, float]
             "gate": 0.0,
             "disqualified": 1.0,
             "hack_penalty": float(HACK_PENALTY),
+            "v_min": float(V_MIN),
+            "v_floor": float(V_FLOOR),
+            "v_target": float(V_TARGET),
+            "low_speed_penalty": 0.0,
         }
     gate = _speed_gate(v_eff)
     return {
```

### 1.4 `raise_core/raise_loop/proxy_feedback.py`

Matches spec: adds `evidence_block()`; highway attach always uses evidence when enabled (no threshold gate); removes `classify_highway_hack` + highway hand-coded focus notes; CrowdNav retains `focus_note_from_metrics` / `should_attach_proxy_feedback`.

```diff
diff --git a/raise_env/raise_core/raise_loop/proxy_feedback.py b/raise_env/raise_core/raise_loop/proxy_feedback.py
index c11c82e..1d029df 100644
--- a/raise_env/raise_core/raise_loop/proxy_feedback.py
+++ b/raise_env/raise_core/raise_loop/proxy_feedback.py
@@ -1,9 +1,7 @@
 """Attach Stage-II proxy metrics to LLM prompts inside the RAISE loop.
 
-Score1 remains the cheap evolutionary signal; SR/CR/TR (+ highway continuous
-fields) feedback is selective so short-horizon noise does not dominate every
-mutation - but known *reward-hacking* fingerprints always surface so the LLM
-sees what the policy actually did.
+Highway: always attach raw evidence (numbers + fitness decomposition) - the LLM
+diagnoses failure modes itself. CrowdNav: keep selective attach + focus notes.
 """
 
 from __future__ import annotations
@@ -12,10 +10,6 @@ from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
 
 from raise_core.explore import RewardCandidate
 
-# Align with domains.highway.metrics.V_TARGET (avoid hard import cycles in CrowdNav).
-_HIGHWAY_TRAFFIC_MPS = 25.0
-_SOFT_FLOOR_LEGACY_MPS = 20.0
-
 
 def nav_scalar(metrics: Mapping[str, Any]) -> float:
     """Selection scalar: highway-aware when continuous fields are present."""
@@ -34,129 +28,69 @@ def _f(metrics: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
     return float(default)
 
 
-def classify_highway_hack(metrics: Mapping[str, Any]) -> Optional[str]:
-    """
-    Detect named reward-hacking / fail modes from holdout Stage-II metrics.
+def _is_highway_metrics(metrics: Mapping[str, Any]) -> bool:
+    return str(metrics.get("domain", "")).strip().lower() == "highway"
 
-    Returns a short mode id or None when behavior looks non-degenerate.
-    """
-    is_highway = str(metrics.get("domain", "")).lower() == "highway"
-    has_speed = "mean_speed" in metrics or "ITR" in metrics
-    if not (is_highway or has_speed):
-        return None
 
-    sr = _f(metrics, "SR", "sr")
-    cr = _f(metrics, "CR", "cr")
-    tr = _f(metrics, "TR", "tr")
-    speed = _f(metrics, "mean_speed", "ITR")
-    soft = _f(metrics, "soft_success") if "soft_success" in metrics else None
-    lc = metrics.get("lane_change_rate")
-    p10 = metrics.get("speed_p10")
-    p90 = metrics.get("speed_p90")
-    pstd = metrics.get("progress_std")
-    spread = None
-    if p10 is not None and p90 is not None:
+def evidence_block(
+    metrics: Mapping[str, Any],
+    *,
+    score1: Optional[float] = None,
+) -> str:
+    """Raw evaluation evidence + fitness decomposition - no interpretation."""
+    from domains.highway.metrics import fitness_components
+
+    src = metrics
+    holdout = metrics.get("holdout")
+    if isinstance(holdout, Mapping) and holdout:
+        src = holdout
+    comps = fitness_components(metrics)
+    p10 = src.get("speed_p10", metrics.get("speed_p10", "n/a"))
+    if p10 is not None and p10 != "n/a":
         try:
-            spread = abs(float(p90) - float(p10))
+            p10 = f"{float(p10):.1f}"
         except (TypeError, ValueError):
-            spread = None
-
-    # Crash-loving / unsafe speed shaping.
-    if cr >= 0.5:
-        return "crash_attract"
-
-    # Constant cruise / soft-threshold lock (the soft@20 exploit fingerprint).
-    flat_speed = spread is not None and spread < 0.5
-    flat_progress = pstd is not None and float(pstd) < 1.0
-    no_lane = lc is not None and float(lc) < 0.02
-    near_soft_floor = (
-        _SOFT_FLOOR_LEGACY_MPS - 1.0 <= speed <= _SOFT_FLOOR_LEGACY_MPS + 1.5
-    )
-    below_traffic = speed + 0.5 < _HIGHWAY_TRAFFIC_MPS
-    if (
-        sr >= 0.85
-        and cr < 0.15
-        and below_traffic
-        and (flat_speed or near_soft_floor)
-        and (no_lane or flat_progress or soft is not None and soft < 0.35)
-    ):
-        return "constant_cruise_hack"
-
-    if soft is not None and sr >= 0.5 and soft < 0.25 and below_traffic:
-        return "below_traffic_soft"
-
-    if sr >= 0.5 and speed < 18.0:
-        return "lag_traffic"
-
-    if sr >= 0.5 and _f(metrics, "mean_progress", "PL") < 300.0:
-        return "low_progress"
-
-    if tr >= 0.5:
-        return "offroad_timeout"
-
-    return None
+            p10 = str(p10)
+    lc = src.get("lane_change_rate", metrics.get("lane_change_rate", "n/a"))
+    if lc is not None and lc != "n/a":
+        try:
+            lc = f"{float(lc):.3f}"
+        except (TypeError, ValueError):
+            lc = str(lc)
+    n_ep = src.get("n_eval_episodes", metrics.get("n_eval_episodes", "?"))
+    progress = _f(src, "mean_progress", "PL", default=_f(metrics, "mean_progress", "PL"))
+    lines = [
+        (
+            f"Holdout eval ({n_ep} episodes): "
+            f"SR={_f(src, 'SR', 'sr'):.2f} CR={_f(src, 'CR', 'cr'):.2f} "
+            f"TR={_f(src, 'TR', 'tr'):.2f} mean_speed={_f(src, 'mean_speed', 'ITR'):.1f}m/s "
+            f"speed_p10={p10} "
+            f"progress={progress:.0f}m "
+            f"soft_success={_f(src, 'soft_success'):.2f} "
+            f"lane_change_rate={lc}"
+        ),
+        (
+            f"Fitness={float(comps.get('fitness', 0.0)):.3f} "
+            f"(speed_gate={float(comps.get('gate', 0.0)):.3f} at "
+            f"v_eff={float(comps.get('v_eff', 0.0)):.1f}, "
+            f"v_min={comps.get('v_min', '?')}, v_target={comps.get('v_target', '?')}, "
+            f"low_speed_penalty={float(comps.get('low_speed_penalty', 0.0)):.3f})"
+        ),
+    ]
+    if score1 is not None and _finite(score1):
+        lines.append(f"Score1={float(score1):.3f}")
+    return "\n".join(lines)
 
 
 def focus_note_from_metrics(metrics: Mapping[str, Any]) -> str:
-    """One-line editing hint for mutation / D.3 - name the hack when present."""
-    mode = classify_highway_hack(metrics)
+    """CrowdNav-only editing hint (highway uses evidence_block instead)."""
+    if _is_highway_metrics(metrics):
+        # Highway path never interprets; keep a neutral stub for legacy callers.
+        return evidence_block(metrics)
+
     sr = _f(metrics, "SR", "sr")
     cr = _f(metrics, "CR", "cr")
     tr = _f(metrics, "TR", "tr")
-    speed = _f(metrics, "mean_speed", "ITR")
-    progress = _f(metrics, "mean_progress", "PL")
-    soft = _f(metrics, "soft_success") if "soft_success" in metrics else None
-
-    if mode == "crash_attract":
-        return (
-            "REWARD HACKING / FAIL: crash attractor - PPO learned high speed then "
-            "collides (CR high). Strengthen collision/off-road penalties relative "
-            "to speed/progress; keep clearance shaping; still target ~25 m/s traffic "
-            "when safe - do NOT only maximize speed"
-        )
-    if mode == "constant_cruise_hack":
-        return (
-            "REWARD HACKING DETECTED: constant-cruise exploit - policy locks near "
-            f"~{speed:.1f} m/s with almost zero speed variance / lane changes / "
-            "progress variance (soft-threshold gaming). Break the plateau: reward "
-            f"matching traffic near {_HIGHWAY_TRAFFIC_MPS:.0f} m/s, penalize flat "
-            "constant speed just above a soft floor, and do not treat SR=1 alone "
-            "as success without throughput diversity"
-        )
-    if mode == "below_traffic_soft":
-        return (
-            "REWARD HACKING RISK: soft_success low while surviving - ego stays "
-            f"below traffic (~{speed:.1f} m/s vs target ~{_HIGHWAY_TRAFFIC_MPS:.0f}). "
-            "Raise shaping toward traffic cruise; soft_success needs survive AND "
-            f">={_HIGHWAY_TRAFFIC_MPS:.0f} m/s AND meaningful progress"
-        )
-    if mode == "lag_traffic":
-        return (
-            "lags traffic - surrounding flow is ~20-30 m/s (target ~25); increase "
-            "reward for matching that band and forward progress; penalize ego "
-            "much slower while on-road"
-        )
-    if mode == "low_progress":
-        return (
-            "low forward progress - strengthen state.progress / speed terms; "
-            "avoid rewarding idle/lane-hold forever"
-        )
-    if mode == "offroad_timeout":
-        return (
-            "high off-road/timeout share - keep on_road shaping; still reward "
-            "forward progress at traffic speed (~25 m/s)"
-        )
-
-    if str(metrics.get("domain", "")).lower() == "highway" or "mean_speed" in metrics:
-        if soft is not None and soft < 0.3 and sr >= 0.4:
-            return (
-                "soft_success low - survive AND match traffic (~>=25 m/s) with "
-                "meaningful progress; balance safety with throughput"
-            )
-        return (
-            "improve safe throughput: higher soft_success (survive+~25 m/s+"
-            "progress) without raising CR; avoid constant-speed lane-keep hacks"
-        )
 
     if tr >= 0.5 and tr >= cr:
         return (
@@ -184,6 +118,9 @@ def format_proxy_feedback_block(
     score1: Optional[float] = None,
 ) -> str:
     """Compact block embedded in mutation weakness / D.3 feedback."""
+    if _is_highway_metrics(metrics):
+        return evidence_block(metrics, score1=score1)
+
     sr = _f(metrics, "SR", "sr")
     cr = _f(metrics, "CR", "cr")
     tr = _f(metrics, "TR", "tr")
@@ -192,39 +129,9 @@ def format_proxy_feedback_block(
     if score1 is not None and _finite(score1):
         s1_bit = f" | Score1={float(score1):.3f}"
     focus = focus_note_from_metrics(metrics)
-    mode = classify_highway_hack(metrics)
-    mode_bit = f" | mode={mode}" if mode else ""
-    extra = ""
-    if "mean_speed" in metrics or str(metrics.get("domain", "")).lower() == "highway":
-        spd = _f(metrics, "mean_speed", "ITR")
-        pl = _f(metrics, "mean_progress", "PL")
-        soft = _f(metrics, "soft_success") if "soft_success" in metrics else 0.0
-        lc = _f(metrics, "lane_change_rate") if "lane_change_rate" in metrics else None
-        p10 = metrics.get("speed_p10")
-        p90 = metrics.get("speed_p90")
-        pstd = metrics.get("progress_std")
-        extra = f" | speed={spd:.1f}m/s progress={pl:.0f}m soft={soft:.2f}"
-        if lc is not None:
-            extra += f" laneΔ={lc:.3f}"
-        if p10 is not None and p90 is not None:
-            try:
-                extra += f" spd_band=[{float(p10):.1f},{float(p90):.1f}]"
-            except (TypeError, ValueError):
-                pass
-        if pstd is not None:
-            try:
-                extra += f" prog_std={float(pstd):.2f}"
-            except (TypeError, ValueError):
-                pass
-        fit = metrics.get("fitness", metrics.get("selection_scalar"))
-        if fit is not None:
-            try:
-                extra += f" fitness={float(fit):.3f}"
-            except (TypeError, ValueError):
-                pass
     return (
         f"ProxyRefine: SR={sr:.2f} CR={cr:.2f} TR={tr:.2f} "
-        f"scalar={sc:.2f}{extra}{s1_bit}{mode_bit}\nFocus: {focus}"
+        f"scalar={sc:.2f}{s1_bit}\nFocus: {focus}"
     )
 
 
@@ -236,33 +143,18 @@ def should_attach_proxy_feedback(
     population_scalars: Optional[Sequence[float]] = None,
 ) -> bool:
     """
-    Attach when proxy looks bad, reward-hacking fingerprint is present, or
-    Score1 looks strong while proxy scalar is weak.
+    CrowdNav: attach when proxy looks bad or Score1/proxy mismatch.
+
+    Highway: always True (callers should prefer ``evidence_block`` attach).
     """
+    if _is_highway_metrics(metrics):
+        return True
+
     sr = _f(metrics, "SR", "sr")
     cr = _f(metrics, "CR", "cr")
     tr = _f(metrics, "TR", "tr")
     if sr < 0.10 or tr >= 0.50 or cr >= 0.50:
         return True
-    # Named highway hacks (including high-SR constant cruise) always surface.
-    if classify_highway_hack(metrics) is not None:
-        return True
-    is_highway = str(metrics.get("domain", "")).lower() == "highway"
-    has_speed = "mean_speed" in metrics or "ITR" in metrics
-    if is_highway or has_speed:
-        soft = (
-            float(metrics["soft_success"])
-            if "soft_success" in metrics and metrics.get("soft_success") is not None
-            else None
-        )
-        # Clear lag only; mild shortfall vs 25 m/s is covered by classify_* /
-        # soft_success - do not flag every 24 m/s survivor as a hack.
-        speed = _f(metrics, "mean_speed", "ITR")
-        lag_floor = 18.0 if is_highway else 12.0
-        if sr >= 0.5 and speed < lag_floor:
-            return True
-        if soft is not None and sr >= 0.5 and soft < 0.25:
-            return True
     if (
         score1 is not None
         and _finite(score1)
@@ -293,8 +185,8 @@ def attach_proxy_feedback(
     """
     Write ``proxy_feedback`` (+ related keys) onto ``candidate.metadata``.
 
-    Always stores ``last_metrics`` if missing. Returns True when a prompt-facing
-    ``proxy_feedback`` block was attached.
+    Highway: when enabled, always attach ``evidence_block`` (no threshold gate).
+    CrowdNav: selective attach after epoch/label gates (unchanged).
     """
     md = dict(candidate.metadata or {})
     md["last_metrics"] = dict(metrics)
@@ -304,26 +196,39 @@ def attach_proxy_feedback(
             score1 = float(md["score1_train"])
         except (TypeError, ValueError):
             score1 = None
+    score1_f = float(score1) if score1 is not None and _finite(score1) else None
+
+    if _is_highway_metrics(metrics):
+        if not bool(enabled):
+            md.pop("proxy_feedback", None)
+            md.pop("proxy_feedback_focus", None)
+            md.pop("proxy_hack_mode", None)
+            candidate.metadata = md
+            return False
+        block = evidence_block(metrics, score1=score1_f)
+        md["proxy_feedback"] = block
+        md["proxy_nav_scalar"] = nav_scalar(metrics)
+        md.pop("proxy_feedback_focus", None)
+        md.pop("proxy_hack_mode", None)
+        candidate.metadata = md
+        return True
 
-    attach = bool(enabled) and int(n_labeled_dataset) >= int(min_labels) and int(epoch) >= 1
+    attach = (
+        bool(enabled)
+        and int(n_labeled_dataset) >= int(min_labels)
+        and int(epoch) >= 1
+    )
     if attach and should_attach_proxy_feedback(
         metrics,
-        score1=float(score1) if score1 is not None and _finite(score1) else None,
+        score1=score1_f,
         population_score1=population_score1,
         population_scalars=population_scalars,
     ):
-        block = format_proxy_feedback_block(
-            metrics,
-            score1=float(score1) if score1 is not None and _finite(score1) else None,
-        )
+        block = format_proxy_feedback_block(metrics, score1=score1_f)
         md["proxy_feedback"] = block
         md["proxy_nav_scalar"] = nav_scalar(metrics)
         md["proxy_feedback_focus"] = focus_note_from_metrics(metrics)
-        mode = classify_highway_hack(metrics)
-        if mode:
-            md["proxy_hack_mode"] = mode
-        else:
-            md.pop("proxy_hack_mode", None)
+        md.pop("proxy_hack_mode", None)
         candidate.metadata = md
         return True
 
@@ -338,7 +243,7 @@ def attach_proxy_feedback(
 
 
 def proxy_summary_for_reflection(candidates: Sequence[RewardCandidate]) -> Optional[str]:
-    """One-line population summary for the global reflection string."""
+    """One-line population summary for the global reflection string (CrowdNav)."""
     rows: List[Tuple[str, float, float, float, float]] = []
     for c in candidates:
         md = c.metadata or {}
@@ -409,6 +314,7 @@ def apply_in_loop_d3(
     """
     from dataclasses import replace
 
+    from raise_core.explore import extract_llm_diagnosis
     from raise_core.llm import extract_python_code, normalize_to_compute_reward
 
     if prompts is not None:
@@ -445,6 +351,7 @@ def apply_in_loop_d3(
     try:
         raw = llm.complete(full_prompt)
         new_code = normalize_to_compute_reward(extract_python_code(raw))
+        diagnosis = extract_llm_diagnosis(raw)
     except Exception:  # noqa: BLE001
         return candidate
     reward_fn, err = validator.try_validate(new_code)
@@ -453,6 +360,7 @@ def apply_in_loop_d3(
     new_md = dict(md)
     new_md["in_loop_d3"] = True
     new_md["in_loop_d3_parent"] = candidate.candidate_id
+    new_md["llm_diagnosis"] = diagnosis
     return replace(
         candidate,
         candidate_id=f"{candidate.candidate_id}_d3",
```

### 1.5 `raise_core/explore.py`

Matches spec: `extract_llm_diagnosis`; highway evidence reflection; CrowdNav advice path guarded by `_is_highway_pack()`; diagnosis stored on candidates.

```diff
diff --git a/raise_env/raise_core/explore.py b/raise_env/raise_core/explore.py
index c096bf4..945d92f 100644
--- a/raise_env/raise_core/explore.py
+++ b/raise_env/raise_core/explore.py
@@ -18,6 +18,7 @@ from __future__ import annotations
 import json
 import logging
 import os
+import re
 from dataclasses import dataclass, field, replace
 from datetime import datetime, timezone
 from typing import Any, Callable, Dict, List, Optional, Sequence
@@ -45,6 +46,20 @@ from domains.crowdnav.state import RewardFunction
 
 logger = logging.getLogger(__name__)
 
+_DIAGNOSIS_FENCE_RE = re.compile(r"```(?:python)?\s*\n", re.IGNORECASE)
+
+
+def extract_llm_diagnosis(raw: str, *, max_chars: int = 500) -> Optional[str]:
+    """Plain text preceding the first code fence, or None if absent."""
+    text = str(raw or "")
+    match = _DIAGNOSIS_FENCE_RE.search(text)
+    if match is None:
+        return None
+    preamble = text[: match.start()].strip()
+    if not preamble:
+        return None
+    return preamble[: int(max_chars)]
+
 
 @dataclass
 class RewardCandidate:
@@ -134,6 +149,9 @@ class StageIEvolver:
         self.reflection: str = ""
         self.history: List[GenerationRecord] = []
         self.global_best: Optional[RewardCandidate] = None
+        # Highway evidence trends (numbers only; fed into reflection).
+        self._highway_best_fitness_trend: List[float] = []
+        self._highway_best_mean_speed_trend: List[float] = []
 
         n = self.config.population_size
         if self.config.n_crossover + self.config.n_mutation + self.config.n_random != n:
@@ -186,6 +204,15 @@ class StageIEvolver:
         code = extract_python_code(raw)
         return normalize_to_compute_reward(code, self.config.func_name)
 
+    def _is_highway_pack(self) -> bool:
+        prompts = self._prompts
+        if prompts is None:
+            return False
+        if str(getattr(prompts, "DOMAIN_NAME", "")).strip().lower() == "highway":
+            return True
+        path = str(getattr(prompts, "__file__", "") or "").replace("\\", "/")
+        return "/domains/highway/" in path
+
     @staticmethod
     def _batch_max_tokens(population_size: int) -> int:
         return max(4000, int(population_size) * 500)
@@ -288,11 +315,14 @@ class StageIEvolver:
         parent_ids: tuple = (),
         metadata: Optional[dict] = None,
     ) -> RewardCandidate:
+        md = dict(metadata or {})
+        if raw_completion:
+            md["llm_diagnosis"] = extract_llm_diagnosis(raw_completion)
         cand = self._make_candidate(
             code,
             origin=origin,
             parent_ids=parent_ids,
-            metadata=metadata,
+            metadata=md,
         )
         self._log_rejection(
             phase=phase,
@@ -583,19 +613,20 @@ class StageIEvolver:
     ) -> RewardCandidate:
         proxy_fb = (parent.metadata or {}).get("proxy_feedback")
         if proxy_fb:
-            hack = (parent.metadata or {}).get("proxy_hack_mode")
-            hack_bit = (
-                f" Parent exhibited reward-hacking mode={hack}; "
-                f"edit the reward to eliminate that exploit.\n"
-                if hack
-                else ""
-            )
-            weakness = (
-                f"Parent {parent.candidate_id} Score1={parent.score}. "
-                f"Use the Stage-II proxy feedback below (higher priority than "
-                f"Score1 alone).{hack_bit}\n{proxy_fb}\n"
-                f"Global reflection: {reflection}"
-            )
+            if self._is_highway_pack():
+                # Evidence only - LLM diagnoses from numbers in the block.
+                weakness = (
+                    f"Parent {parent.candidate_id} Score1={parent.score}.\n"
+                    f"{proxy_fb}\n"
+                    f"Global reflection: {reflection}"
+                )
+            else:
+                weakness = (
+                    f"Parent {parent.candidate_id} Score1={parent.score}. "
+                    f"Use the Stage-II proxy feedback below (higher priority than "
+                    f"Score1 alone).\n{proxy_fb}\n"
+                    f"Global reflection: {reflection}"
+                )
         else:
             weakness = (
                 f"Parent {parent.candidate_id} score={parent.score}. "
@@ -603,7 +634,7 @@ class StageIEvolver:
                 f"Global reflection: {reflection}"
             )
         parent_hints = (parent.metadata or {}).get("score1_failure_hints")
-        if parent_hints and not proxy_fb:
+        if parent_hints and not proxy_fb and not self._is_highway_pack():
             weakness = (
                 f"{weakness}\nScore1 diagnostics for this parent: {parent_hints}"
             )
@@ -770,56 +801,146 @@ class StageIEvolver:
         assert len(next_pop) == cfg.population_size
         return next_pop
 
+    def _build_highway_evidence_reflection(
+        self, ranked: Sequence[RewardCandidate], *, generation: int
+    ) -> str:
+        """Numbers-only trends + population table + prior LLM diagnoses."""
+        from raise_core.selection import candidate_fitness
+
+        rows: List[Dict[str, Any]] = []
+        for c in ranked:
+            md = c.metadata or {}
+            m = md.get("last_metrics")
+            if not isinstance(m, dict):
+                m = {}
+            holdout = m.get("holdout") if isinstance(m.get("holdout"), dict) else m
+            try:
+                fit = float(candidate_fitness(c))
+            except Exception:  # noqa: BLE001
+                fit = float("-inf")
+            try:
+                speed = float(
+                    holdout.get("mean_speed", holdout.get("ITR", float("nan")))
+                )
+            except (TypeError, ValueError):
+                speed = float("nan")
+            rows.append(
+                {
+                    "id": str(c.candidate_id),
+                    "SR": float(
+                        (holdout or {}).get("SR", (holdout or {}).get("sr", float("nan")))
+                    ),
+                    "CR": float(
+                        (holdout or {}).get("CR", (holdout or {}).get("cr", float("nan")))
+                    ),
+                    "mean_speed": speed,
+                    "fitness": fit,
+                    "diagnosis": md.get("llm_diagnosis"),
+                }
+            )
+        rows_sorted = sorted(
+            rows, key=lambda r: r["fitness"], reverse=True
+        )
+        if rows_sorted:
+            best_fit = rows_sorted[0]["fitness"]
+            best_speed = rows_sorted[0]["mean_speed"]
+            if best_fit == best_fit:  # not NaN
+                self._highway_best_fitness_trend.append(float(best_fit))
+            if best_speed == best_speed:
+                self._highway_best_mean_speed_trend.append(float(best_speed))
+            self._highway_best_fitness_trend = self._highway_best_fitness_trend[-5:]
+            self._highway_best_mean_speed_trend = (
+                self._highway_best_mean_speed_trend[-5:]
+            )
+
+        def _fmt_trend(vals: Sequence[float]) -> str:
+            return "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"
+
+        trend = (
+            f"Gen{generation} best_fitness_trend="
+            f"{_fmt_trend(self._highway_best_fitness_trend)} "
+            f"best_mean_speed_trend="
+            f"{_fmt_trend(self._highway_best_mean_speed_trend)}"
+        )
+        table_bits = []
+        for r in rows_sorted:
+            table_bits.append(
+                f"{r['id']}: SR={r['SR']:.2f} CR={r['CR']:.2f} "
+                f"mean_speed={r['mean_speed']:.1f} fitness={r['fitness']:.3f}"
+            )
+        table = "Population(fitness desc): " + (
+            "; ".join(table_bits) if table_bits else "(empty)"
+        )
+        diag_bits: List[str] = []
+        for r in rows_sorted:
+            d = r.get("diagnosis")
+            if d:
+                diag_bits.append(f"{r['id']}: {str(d).strip()}")
+            if len(diag_bits) >= 3:
+                break
+        diag = (
+            "PriorLLMDiagnoses: " + " | ".join(diag_bits)
+            if diag_bits
+            else "PriorLLMDiagnoses: (none)"
+        )
+        return f"{trend}. {table}. {diag}"
+
     def _build_reflection(
         self, ranked: Sequence[RewardCandidate], *, generation: int
     ) -> str:
         """
         Reflective note for the next generation (§4.2).
 
-        Accumulates concise notes across generations (bounded), including
-        Score1 for all candidates - not only best/worst.
+        Highway: factual trends + population metrics + prior LLM diagnoses.
+        CrowdNav: Score1 summary + selective proxy line (unchanged advice path).
         """
-        score_bits = []
-        for c in ranked:
-            sc = c.score if c.score is not None else float("nan")
-            score_bits.append(f"{c.candidate_id}={sc}")
-        best = ranked[0]
-        worst = ranked[-1]
-        lower = ranked[len(ranked) // 2 :]
-        lower_ids = ", ".join(c.candidate_id for c in lower[:4]) if lower else "(none)"
-        note = (
-            f"Gen{generation} scores[{', '.join(score_bits)}]. "
-            f"Best={best.candidate_id} score={best.score}; "
-            f"Worst={worst.candidate_id} score={worst.score}. "
-            f"Underperformers ({lower_ids}): strengthen goal progress and "
-            f"collision/discomfort penalties while keeping dense shaping. "
-            f"Prefer combining elite safety terms with efficient progress; "
-            f"avoid near-constant rewards."
-        )
-        diag_bits = []
-        for c in (worst, *(lower[:2])):
-            hints = (c.metadata or {}).get("score1_failure_hints")
-            if hints:
-                diag_bits.append(f"{c.candidate_id}: {hints}")
-        if diag_bits:
-            # Deduplicate while preserving order.
-            seen = set()
-            uniq = []
-            for b in diag_bits:
-                if b not in seen:
-                    seen.add(b)
-                    uniq.append(b)
-            note = note + " Diagnostics: " + " | ".join(uniq[:3])
-        try:
-            from raise_core.raise_loop.proxy_feedback import (
-                proxy_summary_for_reflection,
+        if self._is_highway_pack():
+            note = self._build_highway_evidence_reflection(
+                ranked, generation=generation
             )
+        else:
+            score_bits = []
+            for c in ranked:
+                sc = c.score if c.score is not None else float("nan")
+                score_bits.append(f"{c.candidate_id}={sc}")
+            best = ranked[0]
+            worst = ranked[-1]
+            lower = ranked[len(ranked) // 2 :]
+            lower_ids = (
+                ", ".join(c.candidate_id for c in lower[:4]) if lower else "(none)"
+            )
+            note = (
+                f"Gen{generation} scores[{', '.join(score_bits)}]. "
+                f"Best={best.candidate_id} score={best.score}; "
+                f"Worst={worst.candidate_id} score={worst.score}. "
+                f"Underperformers ({lower_ids}): strengthen goal progress and "
+                f"collision/discomfort penalties while keeping dense shaping. "
+                f"Prefer combining elite safety terms with efficient progress; "
+                f"avoid near-constant rewards."
+            )
+            diag_bits = []
+            for c in (worst, *(lower[:2])):
+                hints = (c.metadata or {}).get("score1_failure_hints")
+                if hints:
+                    diag_bits.append(f"{c.candidate_id}: {hints}")
+            if diag_bits:
+                seen = set()
+                uniq = []
+                for b in diag_bits:
+                    if b not in seen:
+                        seen.add(b)
+                        uniq.append(b)
+                note = note + " Diagnostics: " + " | ".join(uniq[:3])
+            try:
+                from raise_core.raise_loop.proxy_feedback import (
+                    proxy_summary_for_reflection,
+                )
 
-            proxy_line = proxy_summary_for_reflection(ranked)
-            if proxy_line:
-                note = note + " " + proxy_line
-        except Exception:  # noqa: BLE001
-            pass
+                proxy_line = proxy_summary_for_reflection(ranked)
+                if proxy_line:
+                    note = note + " " + proxy_line
+            except Exception:  # noqa: BLE001
+                pass
         if not self.reflection.strip():
             return note
         parts = [p.strip() for p in self.reflection.split(" || ") if p.strip()]
```

### 1.6 `domains/highway/prompts.py`

Matches spec: fitness formula from constants in D1/D4; diagnose-before-code in D2/D3; removed lagging / soft@20 prescription language.

```diff
diff --git a/raise_env/domains/highway/prompts.py b/raise_env/domains/highway/prompts.py
index 0e24fa9..ca87863 100644
--- a/raise_env/domains/highway/prompts.py
+++ b/raise_env/domains/highway/prompts.py
@@ -4,20 +4,49 @@ from __future__ import annotations
 
 from typing import Optional
 
+from domains.highway.objective_constants import K_GATE, V_MIN, V_TARGET
+
+DOMAIN_NAME = "highway"
+
+_FITNESS_OBJECTIVE = (
+    "The reward function you write is NOT the selection score. After PPO "
+    "training, policies are scored on held-out rollouts using this fitness "
+    "function (higher is better):\n"
+    "  v_eff = 10th-percentile speed over the episode\n"
+    f"  gate  = sigmoid({K_GATE} * (v_eff - ({V_MIN}+{V_TARGET})/2) / "
+    f"({V_TARGET}-{V_MIN}))\n"
+    "  fitness = gate * (SR - CR - 0.5*TR)\n"
+    f"          + 0.35*tanh(progress/800) + 0.25*tanh(mean_speed/{V_TARGET})\n"
+    "          + 0.15*soft_success*gate - penalties\n"
+    f"  where V_MIN={V_MIN}, V_TARGET={V_TARGET} m/s.\n"
+    "Use this to reason about trade-offs yourself; do not assume any specific "
+    "failure mode is or isn't present."
+)
+
+_DIAGNOSIS_BEFORE_CODE = (
+    "Before writing code:\n"
+    "1) In 2-4 sentences, based ONLY on the evidence given above (not on any "
+    "assumption about what 'usually' goes wrong), diagnose what is currently "
+    "limiting fitness from improving further - e.g. a saturated objective term, "
+    "an unexploited safety/speed trade-off, a degenerate strategy, or something "
+    "else you notice in the numbers.\n"
+    "2) Then revise the reward function to address your own diagnosis.\n"
+    "Output your diagnosis as plain text BEFORE the code fence. The code fence "
+    "must contain ONLY the function, no diagnosis text inside it."
+)
+
 D1_SYSTEM_PROMPT = (
     "You are an expert in reinforcement learning and autonomous highway driving. "
-    "Your goal is to design reward functions that keep the ego vehicle safe, "
-    "on-road, and matching traffic flow: surrounding vehicles cruise at about "
-    "20 m/s or faster, so ego should typically stay near ~20-30 m/s while "
-    "making forward progress. "
-    "Return **only** valid Python code enclosed within a fenced code block. "
-    "The code must be fully executable and should not include comments or "
-    "explanations outside the block."
+    "Your goal is to design reward functions for highway-fast-v0. "
+    + _FITNESS_OBJECTIVE
+    + " Return **only** valid Python code enclosed within a fenced code block "
+    "for this initial generation (no commentary outside the block). "
+    "The code must be fully executable."
 )
 
 D1_USER_PROMPT = """Please write a Python function named {func_name} for highway driving (highway-fast-v0).
 Task Description:
-- Output a scalar reward from the ego vehicle's current state so a policy learns to drive forward safely without collisions or leaving the road.
+- Output a scalar reward from the ego vehicle's current state so a policy can learn safe forward driving under the fitness objective above.
 Function Interface (EXACT FIELD STRUCTURE):
 - Inputs:
   - state: A HighwayRewardState snapshot with ONLY these fields:
@@ -34,13 +63,9 @@ Function Interface (EXACT FIELD STRUCTURE):
 - Output:
   - A single finite float reward for the current frame.
 - Design Principles:
-- Progress / speed: reward forward motion and traffic-matching cruise (~20-30 m/s).
-  Nearby traffic already moves at ≥~20 m/s; ego slower than that lags the flow
-  and never closes on surrounding vehicles.
-- Safety: heavily penalize collision and off-road; shape clearance to nearby vehicles.
-- Do NOT reward lagging behind traffic: “survive by going much slower than ~20 m/s”
-  is a failure mode (not “near-zero crawl” - that is unrealistic here).
+- Prefer dense, finite shaping from documented state fields.
 - Interpretability: clear local variables; no extra signature args.
+- Reason about fitness trade-offs yourself from the objective statement; do not assume a named failure mode.
 Episode memory example:
 ```python
 def compute_reward(state, memory):
@@ -100,7 +125,7 @@ Sandbox rules (CRITICAL - invalid code is discarded):
 - Use ``memory`` (dict) for cross-timestep shaping; never invent state.history.
 {reward_state_access}
 - Hyperparameters as locals only; no extra args beyond (state, memory).
-- Return ONLY one Python fenced code block with no text outside it.
+- Put diagnosis as plain text BEFORE the fence; the fenced block must contain ONLY the function.
 """
 
 D2_CROSSOVER_PROMPT = """You are a reward function architect for highway driving. Synthesize a new function combining complementary strengths of two parents while addressing the reflection.
@@ -108,88 +133,79 @@ Parent A:
 - {code_A}
 Parent B:
 - {code_B}
-Reflection:
+Reflection / evidence:
 - {reflection}.
 Synthesis Task:
-- Write an improved `{func_name}` that merges safety and progress terms.
+- Write an improved `{func_name}` that merges complementary terms from the parents.
 - Strip any getattr/hasattr patterns; use direct state.* / v.* access.
 - Define exactly one function: def {func_name}(state, memory): ... returning a finite float.
 {sandbox_rules}
-- Return only a single Python fenced code block.
+""" + _DIAGNOSIS_BEFORE_CODE + """
 """
 
-D2_MUTATION_PROMPT = """You are a reward function optimizer for highway driving. Mutate the underperforming parent to address the weakness below with minimal edits.
-Prior Reflection:
+D2_MUTATION_PROMPT = """You are a reward function optimizer for highway driving. Mutate the underperforming parent using the evidence below with minimal edits.
+Prior Reflection / evidence:
 - {reflection}
 Underperforming Parent Code to Mutate:
 - {func_signature}
 - {parent_code}
 Mutation Task:
-- Create a mutated `{func_name}` with a small precise change.
+- Create a mutated `{func_name}` with a small precise change that addresses your diagnosis.
 - Keep direct dot access; no getattr/hasattr.
 - Define exactly one function: def {func_name}(state, memory): ... returning a finite float.
 {sandbox_rules}
-- Return only a single Python fenced code block.
+""" + _DIAGNOSIS_BEFORE_CODE + """
 """
 
 D3_SYSTEM_PROMPT = (
     "You are a senior researcher in autonomous driving and RL. "
-    "Rewrite the reward to produce a smooth, dense, numerically stable per-frame "
-    "signal that differentiates safe traffic-speed driving (~20-30 m/s) from "
-    "collisions, off-road, and lagging behind the ≥~20 m/s traffic stream. "
-    "IMPORTANT SCHEMA: def compute_reward(state, memory): - memory is a plain dict. "
+    + _FITNESS_OBJECTIVE
+    + " IMPORTANT SCHEMA: def compute_reward(state, memory): - memory is a plain dict. "
     "state.ego has .x .y .vx .vy .heading .speed .lane_index .on_road. "
     "state.others is a tuple; iterate with 'for v in state.others:'. "
     "state.collision / state.off_road / state.timeout (bool). "
     "state.progress and state.speed are available. "
     "NO state.robot, NO state.humans, NO state.gx, NO state.lane_position, NO state.distance. "
     "SANDBOX: never getattr/hasattr/__import__/eval; use ** 0.5; always return a finite float. "
-    "Output only valid Python code (no markdown fences or commentary)."
+    "Write a short diagnosis as plain text, then a single Python fenced code block "
+    "containing ONLY the revised function."
 )
 
 D3_USER_PROMPT = """Current score (best so far): {last_score:.4f} (higher is better)
-Core components:
-- Forward progress + traffic-matching speed shaping (~25 m/s target, ~20-30 band)
-- Collision and off-road penalties
-- Clearance to nearby vehicles
-- Penalty for lagging below traffic speed (~25 m/s)
-- Stability (bounded magnitudes); avoid flat constant ~20 m/s cruise
-Focus note: {feedback}
+Evidence / focus note:
+{feedback}
 {extra_context_if_any}
+""" + _DIAGNOSIS_BEFORE_CODE + """
 Revise the function below.
 Maintain signature def compute_reward(state, memory): and return a finite float.
 {current_code}
 """
 
-D4_EXTERNAL_KNOWLEDGE = """# External Knowledge - Highway Fast
+D4_EXTERNAL_KNOWLEDGE = f"""# External Knowledge - Highway Fast
 ## Task
 - Domain: multi-lane highway driving (highway-fast-v0)
-- Surrounding vehicles typically cruise near ~25 m/s (often ~20-30 m/s)
-- Ego must survive without collision/off-road while matching that flow (~25 m/s)
-  and making forward progress - do NOT lock to a flat ~20 m/s lane-keep cruise
+## Selection objective (not the reward itself)
+{_FITNESS_OBJECTIVE}
 ## Metrics (mapped to RAISE ProxyMetrics)
 - SR: fraction of episodes survived without collision/off-road
 - CR: collision rate; TR: off-road rate
-- Also optimize mean speed + forward progress (do NOT survive by lagging << traffic)
-- soft_success: survive AND traffic-speed cruise (~≥25 m/s) AND meaningful progress
-- Primary scalar mixes safety with throughput (progress/speed), with lag-behind-traffic
-  and constant-cruise penalties
+- mean_speed / mean_progress / soft_success / lane_change_rate as logged
+- soft_success: survive AND speed ≥ V_TARGET ({V_TARGET} m/s) AND meaningful progress
 """
 
-D5_SEED_FUNCTION = '''def compute_reward(state, memory):
-    """Highway seed: progress + speed; punish crash/off-road and lagging traffic."""
+D5_SEED_FUNCTION = f'''def compute_reward(state, memory):
+    """Highway seed: dense progress/speed shaping with collision/off-road costs."""
     collision_penalty = -20.0
     off_road_penalty = -10.0
     speed_coef = 0.08
     progress_coef = 1.0
-    traffic_speed = 25.0
+    traffic_speed = {V_TARGET}
     lag_penalty = 0.15
     if state.collision:
         return float(collision_penalty)
     if state.off_road:
         return float(off_road_penalty)
     reward = progress_coef * state.progress + speed_coef * state.speed
-    # Match traffic near ~25 m/s; slower than that lags the flow.
     if (not state.timeout) and state.ego.on_road and state.speed < traffic_speed:
         reward = reward - lag_penalty * (traffic_speed - state.speed)
     return float(reward)
@@ -214,7 +230,7 @@ def format_d1_initial(
     reflection_block = ""
     if reflection.strip():
         reflection_block = (
-            f"Reflective guidance from prior generations:\n{reflection.strip()}\n"
+            f"Reflective guidance / evidence from prior generations:\n{reflection.strip()}\n"
         )
     external_block = ""
     if include_external_knowledge:
@@ -257,7 +273,7 @@ def format_d1_initial_batch(
         )
     reflection_block = ""
     if reflection.strip():
-        reflection_block = f"Reflection:\n{reflection.strip()}\n"
+        reflection_block = f"Reflection / evidence:\n{reflection.strip()}\n"
     external_block = ""
     if include_external_knowledge:
         external_block = f"External knowledge:\n{D4_EXTERNAL_KNOWLEDGE}\n"
```

### 1.7 Related: `domains/highway/adapter.py`

```diff
diff --git a/raise_env/domains/highway/adapter.py b/raise_env/domains/highway/adapter.py
index 31aa961..27e9f8a 100644
--- a/raise_env/domains/highway/adapter.py
+++ b/raise_env/domains/highway/adapter.py
@@ -125,7 +125,7 @@ def _eval_metrics(
     n_episodes: int,
     seed: int = 0,
     seed_offset: int = 10_003,
-    soft_speed_mps: float = 25.0,
+    soft_speed_mps: float | None = None,
     soft_progress_m: float = 400.0,
 ) -> ProxyMetrics:
     """
@@ -156,8 +156,12 @@ def _eval_metrics(
     high_speed_steps = 0
     outcomes: list[str] = []
 
-    # Soft-success: survive AND match traffic (~≥V_TARGET=25 m/s) + progress.
-    min_speed_for_soft = float(soft_speed_mps)
+    # Soft-success: survive AND match traffic (~≥V_TARGET) + progress.
+    from domains.highway.objective_constants import V_TARGET
+
+    min_speed_for_soft = float(
+        V_TARGET if soft_speed_mps is None else soft_speed_mps
+    )
     min_progress_for_soft = float(soft_progress_m)
     n_eps = max(1, int(n_episodes))
     base = int(seed)
```

### 1.8 Grep audits

#### Command A

```text
rg -n "focus_note_from_metrics|should_attach_proxy_feedback" raise_env/
```

Live hits:

- `raise_core/raise_loop/proxy_feedback.py` - **definitions + CrowdNav-only call sites** (intentional; highway branch does not use the gate).
- `raise_core/tests/test_proxy_feedback.py`, `test_evidence_feedback.py` - tests.

#### Command B

```text
rg -n "lags traffic|strengthen goal progress|is a failure mode" raise_env/domains/highway/ raise_env/raise_core/
```

Live hits:

- `raise_core/explore.py` (~line 916) `strengthen goal progress` - **CrowdNav-only** branch of `_build_reflection`.
- `raise_core/tests/test_evidence_feedback.py` - asserts highway reflection does **not** contain that string.
- **Zero matches under `domains/highway/`.**
- **Zero matches** for `lags traffic` / `is a failure mode` in live highway / raise_core non-test code.

---

## SECTION 2 - Math verification of the recentered gate

Script: `raise_env/scripts/_verify_gate.py` (imports the real `_speed_gate`).

### Full printed table

```text
V_MIN=10.0 V_TARGET=25.0 K_GATE(new)=6.0
     v    gate_NEW    gate_OLD   delta_new-old
------------------------------------------------
   0.0    0.000911    0.039166       -0.038255
   5.0    0.006693    0.167982       -0.161289
  10.0    0.047426    0.500000       -0.452574
  12.5    0.119203    0.689974       -0.570772
  15.0    0.268941    0.832018       -0.563077
  17.5    0.500000    0.916827       -0.416827
  20.0    0.731059    0.960834       -0.229776
  22.5    0.880797    0.982014       -0.101217
  25.0    0.952574    0.991837       -0.039263
  27.5    0.982014    0.996316       -0.014302
  30.0    0.993307    0.998341       -0.005034
  35.0    0.999089    0.999665       -0.000576

gate(20)           = 0.731059
gate(25)           = 0.952574
gate(25)-gate(20)  = 0.221516
gate(V_MIN=10.0) = 0.047426
gate(V_TARGET=25.0) = 0.952574
old gate(20)       = 0.960834
old gate(25)       = 0.991837
old (25)-(20)      = 0.031003
```

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

## SECTION 3 - Prompt content audit (real render)

Representative holdout metrics (~20 m/s plateau failure mode):

```json
{
  "domain": "highway",
  "SR": 0.95, "CR": 0.05, "TR": 0.0,
  "mean_speed": 20.06, "speed_p10": 20.06,
  "mean_progress": 803.0, "soft_success": 0.0,
  "lane_change_rate": 0.0, "n_eval_episodes": 20
}
```

### Literal full render (`scripts/_verify_prompt_render.py`)

```text
===== EVIDENCE_BLOCK =====
Holdout eval (20 episodes): SR=0.95 CR=0.05 TR=0.00 mean_speed=20.1m/s speed_p10=20.1 progress=803m soft_success=0.00 lane_change_rate=0.000
Fitness=0.346 (speed_gate=0.736 at v_eff=20.1, v_min=10.0, v_target=25.0, low_speed_penalty=0.000)
Score1=0.612

===== D1_SYSTEM_PROMPT (fitness formula block) =====
You are an expert in reinforcement learning and autonomous highway driving. Your goal is to design reward functions for highway-fast-v0. The reward function you write is NOT the selection score. After PPO training, policies are scored on held-out rollouts using this fitness function (higher is better):
  v_eff = 10th-percentile speed over the episode
  gate  = sigmoid(6.0 * (v_eff - (10.0+25.0)/2) / (25.0-10.0))
  fitness = gate * (SR - CR - 0.5*TR)
          + 0.35*tanh(progress/800) + 0.25*tanh(mean_speed/25.0)
          + 0.15*soft_success*gate - penalties
  where V_MIN=10.0, V_TARGET=25.0 m/s.
Use this to reason about trade-offs yourself; do not assume any specific failure mode is or isn't present. Return **only** valid Python code enclosed within a fenced code block for this initial generation (no commentary outside the block). The code must be fully executable.

===== D2_MUTATION FULL RENDER =====
You are a reward function optimizer for highway driving. Mutate the underperforming parent using the evidence below with minimal edits.
Prior Reflection / evidence:
- Gen1 best_fitness_trend=[0.412, 0.455] best_mean_speed_trend=[20.060, 20.100]. Population(fitness desc): a: SR=0.95 CR=0.05 mean_speed=20.1 fitness=0.455; b: SR=0.80 CR=0.10 mean_speed=18.5 fitness=0.310. PriorLLMDiagnoses: (none)
Parent metrics evidence:
Holdout eval (20 episodes): SR=0.95 CR=0.05 TR=0.00 mean_speed=20.1m/s speed_p10=20.1 progress=803m soft_success=0.00 lane_change_rate=0.000
Fitness=0.346 (speed_gate=0.736 at v_eff=20.1, v_min=10.0, v_target=25.0, low_speed_penalty=0.000)
Score1=0.612
Underperforming Parent Code to Mutate:
- def compute_reward(state, memory):
- def compute_reward(state, memory):
    """Highway seed: dense progress/speed shaping with collision/off-road costs."""
    collision_penalty = -20.0
    off_road_penalty = -10.0
    speed_coef = 0.08
    progress_coef = 1.0
    traffic_speed = 25.0
    lag_penalty = 0.15
    if state.collision:
        return float(collision_penalty)
    if state.off_road:
        return float(off_road_penalty)
    reward = progress_coef * state.progress + speed_coef * state.speed
    if (not state.timeout) and state.ego.on_road and state.speed < traffic_speed:
        reward = reward - lag_penalty * (traffic_speed - state.speed)
    return float(reward)
Mutation Task:
- Create a mutated `compute_reward` with a small precise change that addresses your diagnosis.
- Keep direct dot access; no getattr/hasattr.
- Define exactly one function: def compute_reward(state, memory): ... returning a finite float.

Sandbox rules (CRITICAL - invalid code is discarded):
- Define exactly ONE top-level function `compute_reward(state, memory)` returning a finite float.
- Do NOT use import/from, classes, while loops, print, lambda, or reflection builtins.
- Forbidden: getattr, hasattr, __import__, eval, exec, type, setattr, delattr, globals, locals, vars, open.
- Access HighwayRewardState only via dot notation (see below).
- Use ``memory`` (dict) for cross-timestep shaping; never invent state.history.
HighwayRewardState access (dot notation only - never getattr/hasattr/__import__):
- state.ego.x, state.ego.y, state.ego.vx, state.ego.vy, state.ego.heading, state.ego.speed, state.ego.lane_index, state.ego.on_road
- state.others - loop `for v in state.others:` then v.x, v.y, v.vx, v.vy, v.heading
- state.collision, state.off_road, state.timeout
- state.action, state.time_step, state.global_time, state.time_limit, state.progress, state.speed
- memory: plain dict for episode-local state; cleared on reset
- FORBIDDEN (rejected by AST): lane_position, distance, robot, humans, gx/gy, history, px/py
- Math: no import math; use ** 0.5. No getattr/hasattr/__import__.

- Hyperparameters as locals only; no extra args beyond (state, memory).
- Put diagnosis as plain text BEFORE the fence; the fenced block must contain ONLY the function.

Before writing code:
1) In 2-4 sentences, based ONLY on the evidence given above (not on any assumption about what 'usually' goes wrong), diagnose what is currently limiting fitness from improving further - e.g. a saturated objective term, an unexploited safety/speed trade-off, a degenerate strategy, or something else you notice in the numbers.
2) Then revise the reward function to address your own diagnosis.
Output your diagnosis as plain text BEFORE the code fence. The code fence must contain ONLY the function, no diagnosis text inside it.


===== D3_USER FULL RENDER =====
Current score (best so far): 0.4550 (higher is better)
Evidence / focus note:
Holdout eval (20 episodes): SR=0.95 CR=0.05 TR=0.00 mean_speed=20.1m/s speed_p10=20.1 progress=803m soft_success=0.00 lane_change_rate=0.000
Fitness=0.346 (speed_gate=0.736 at v_eff=20.1, v_min=10.0, v_target=25.0, low_speed_penalty=0.000)
Score1=0.612
In-loop proxy refine (short Stage II metrics).
Before writing code:
1) In 2-4 sentences, based ONLY on the evidence given above (not on any assumption about what 'usually' goes wrong), diagnose what is currently limiting fitness from improving further - e.g. a saturated objective term, an unexploited safety/speed trade-off, a degenerate strategy, or something else you notice in the numbers.
2) Then revise the reward function to address your own diagnosis.
Output your diagnosis as plain text BEFORE the code fence. The code fence must contain ONLY the function, no diagnosis text inside it.
Revise the function below.
Maintain signature def compute_reward(state, memory): and return a finite float.
def compute_reward(state, memory):
    """Highway seed: dense progress/speed shaping with collision/off-road costs."""
    collision_penalty = -20.0
    off_road_penalty = -10.0
    speed_coef = 0.08
    progress_coef = 1.0
    traffic_speed = 25.0
    lag_penalty = 0.15
    if state.collision:
        return float(collision_penalty)
    if state.off_road:
        return float(off_road_penalty)
    reward = progress_coef * state.progress + speed_coef * state.speed
    if (not state.timeout) and state.ego.on_road and state.speed < traffic_speed:
        reward = reward - lag_penalty * (traffic_speed - state.speed)
    return float(reward)


===== BANNED-WORD SCAN (evidence_block only) =====
  evidence contains 'lags': False
  evidence contains 'hack': False
  evidence contains 'degenerate': False
  evidence contains 'improve': False
  evidence contains 'strengthen': False
  evidence contains 'weakness focus': False

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

## SECTION 4 - Live run evidence (real PPO)

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

- **No stubs / no `--fast`** - real SB3 PPO on `highway-fast-v0`.
- `--score1 smoke` because local Stage I dataset was missing; **PPO labels are real**.
- **No Groq/Ollama/OpenAI keys** → seed LLM only → live `llm_diagnosis` is null.
- Wall-clock: **21m 08.1s**. Artifacts: `raise_env/results/_verify_live_highway/`.

### Per-generation Stage II holdout metrics (parsed from run log)

```json
[
  {
    "epoch": 0,
    "n": 6,
    "best_fitness": 0.319,
    "best_id": "ini_0001",
    "best_mean_speed": 20.1,
    "mean_speed_mean": 20.917,
    "mean_speed_min": 20.1,
    "mean_speed_max": 25.0,
    "mean_speed_spread": 4.9,
    "candidates": [
      {
        "id": "ini_0001",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "ini_0002",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "ini_0000",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "ini_0003",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "ini_0004",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "ini_0005",
        "SR": "0.00",
        "speed": 25.0,
        "fitness": -0.635
      }
    ]
  },
  {
    "epoch": 1,
    "n": 6,
    "best_fitness": 0.8,
    "best_id": "mut_0009",
    "best_mean_speed": 21.8,
    "mean_speed_mean": 25.533,
    "mean_speed_min": 21.8,
    "mean_speed_max": 29.5,
    "mean_speed_spread": 7.7,
    "candidates": [
      {
        "id": "cro_0006",
        "SR": "0.00",
        "speed": 25.0,
        "fitness": -0.575
      },
      {
        "id": "mut_0009",
        "SR": "0.75",
        "speed": 21.8,
        "fitness": 0.8
      },
      {
        "id": "cro_0007",
        "SR": "0.62",
        "speed": 22.6,
        "fitness": 0.595
      },
      {
        "id": "mut_0010",
        "SR": "0.00",
        "speed": 29.4,
        "fitness": -0.667
      },
      {
        "id": "mut_0011",
        "SR": "0.00",
        "speed": 29.5,
        "fitness": -0.693
      },
      {
        "id": "mut_0014",
        "SR": "0.00",
        "speed": 24.9,
        "fitness": -0.652
      }
    ]
  },
  {
    "epoch": 2,
    "n": 6,
    "best_fitness": 0.319,
    "best_id": "mut_0016",
    "best_mean_speed": 20.1,
    "mean_speed_mean": 23.3,
    "mean_speed_min": 20.1,
    "mean_speed_max": 29.5,
    "mean_speed_spread": 9.4,
    "candidates": [
      {
        "id": "cro_0012",
        "SR": "0.00",
        "speed": 29.5,
        "fitness": -0.682
      },
      {
        "id": "mut_0015",
        "SR": "0.00",
        "speed": 25.0,
        "fitness": -0.654
      },
      {
        "id": "mut_0016",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "mut_0017",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "cro_0019",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      },
      {
        "id": "mut_0021",
        "SR": "0.00",
        "speed": 25.0,
        "fitness": -0.609
      }
    ]
  },
  {
    "epoch": 3,
    "n": 2,
    "best_fitness": 0.319,
    "best_id": "mut_0022",
    "best_mean_speed": 20.1,
    "mean_speed_mean": 22.55,
    "mean_speed_min": 20.1,
    "mean_speed_max": 25.0,
    "mean_speed_spread": 4.9,
    "candidates": [
      {
        "id": "mut_0020",
        "SR": "0.25",
        "speed": 25.0,
        "fitness": -0.067
      },
      {
        "id": "mut_0022",
        "SR": "1.00",
        "speed": 20.1,
        "fitness": 0.319
      }
    ]
  }
]
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
{
  "parent_proxy_feedback": "Holdout eval (10 episodes): SR=0.90 CR=0.08 TR=0.02 mean_speed=20.1m/s speed_p10=19.8 progress=780m soft_success=0.00 lane_change_rate=0.010\nFitness=0.859 (speed_gate=0.715 at v_eff=19.8, v_min=10.0, v_target=25.0, low_speed_penalty=0.000)\nScore1=0.550",
  "child_valid": true,
  "child_validation_error": null,
  "child_llm_diagnosis": "Holdout mean_speed sits near 20 m/s while V_TARGET is 25; the speed_gate is only ~0.73 so fitness still has headroom on the speed band, but CR=0.08 suggests pushing speed without more collision shaping may trade off badly. I will raise traffic-matching shaping gently and keep a stronger collision cost.",
  "extract_direct": "Holdout mean_speed sits near 20 m/s while V_TARGET is 25; the speed_gate is only ~0.73 so fitness still has headroom on the speed band, but CR=0.08 suggests pushing speed without more collision shaping may trade off badly. I will raise traffic-matching shaping gently and keep a stronger collision cost.",
  "evidence_has_verdict_lags": false
}
```

Verbatim stored diagnosis:

> Holdout mean_speed sits near 20 m/s while V_TARGET is 25; the speed_gate is only ~0.73 so fitness still has headroom on the speed band, but CR=0.08 suggests pushing speed without more collision shaping may trade off badly. I will raise traffic-matching shaping gently and keep a stronger collision cost.

This mentions gate / speed headroom / collision trade-off - but it is from a **compliance ScriptedLLM**, not from the 21-minute seed-LLM PPO run.

### Speed vs ~20 m/s plateau narrative

- Epoch 0 elites: **20.1 m/s**, fitness **0.319**, gate ≈ 0.736 (not saturated).
- Epoch 1 best: **mut_0009 @ 21.8 m/s, fitness 0.800** (best_ever).
- Epochs 2-3 fall back to **20.1 m/s** elites under seed mutations.
- **No sustained climb to 25 m/s.** No paired old-gate baseline was re-run here; epoch-0 matches the prior plateau narrative, epoch-1 shows the new objective can prefer slightly higher speed when achieved.

---

## SECTION 5 - Test results

### Targeted new tests (`-vv`)

```text
--= = = = = = = = = = = = = = = = = = = = = = = = = = = = =   t e s t   s e s s i o n   s t a r t s   = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
 
 p l a t f o r m   w i n 3 2   - -   P y t h o n   3 . 1 2 . 2 ,   p y t e s t - 8 . 4 . 2 ,   p l u g g y - 1 . 6 . 0   - -   C : \ U s e r s \ n e m a t k h a h \ A p p D a t a \ L o c a l \ P r o g r a m s \ P y t h o n \ P y t h o n 3 1 2 \ p y t h o n . e x e 
 
 c a c h e d i r :   . p y t e s t _ c a c h e 
 
 r o o t d i r :   F : \ C o d e \ r a i s e - p a p e r - b a s e l i n e \ r a i s e _ e n v 
 
 c o n f i g f i l e :   p y t e s t . i n i 
 
 p l u g i n s :   a n y i o - 4 . 1 3 . 0 ,   c o v - 5 . 0 . 0 
 
 c o l l e c t i n g   . . .   c o l l e c t e d   6   i t e m s 
 
 
 
 d o m a i n s / h i g h w a y / t e s t s / t e s t _ f i t n e s s _ g a t e _ g r a d i e n t . p y : : t e s t _ o b j e c t i v e _ c o n s t a n t s _ s h a r e d   P A S S E D   [   1 6 % ] 
 
 d o m a i n s / h i g h w a y / t e s t s / t e s t _ f i t n e s s _ g a t e _ g r a d i e n t . p y : : t e s t _ g a t e _ g r a d i e n t _ 2 0 _ v s _ 2 5   P A S S E D   [   3 3 % ] 
 
 d o m a i n s / h i g h w a y / t e s t s / t e s t _ f i t n e s s _ g a t e _ g r a d i e n t . p y : : t e s t _ g a t e _ n o t _ s a t u r a t e d _ a t _ b o u n d s   P A S S E D   [   5 0 % ] 
 
 d o m a i n s / h i g h w a y / t e s t s / t e s t _ f i t n e s s _ g a t e _ g r a d i e n t . p y : : t e s t _ f l o o r _ d i s q u a l i f y _ u n c h a n g e d   P A S S E D   [   6 6 % ] 
 
 r a i s e _ c o r e / t e s t s / t e s t _ e v i d e n c e _ f e e d b a c k . p y : : t e s t _ e v i d e n c e _ b l o c k _ i s _ n u m b e r s _ o n l y   P A S S E D   [   8 3 % ] 
 
 r a i s e _ c o r e / t e s t s / t e s t _ e v i d e n c e _ f e e d b a c k . p y : : t e s t _ h i g h w a y _ b u i l d _ r e f l e c t i o n _ n o _ h a r d c o d e d _ a d v i c e   P A S S E D   [ 1 0 0 % ] 
 
 
 
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =   6   p a s s e d   i n   1 . 3 9 s   = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
 
 
```

### Full highway + explore + crowdnav suites

```text
--. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   [   4 8 % ] 
 
 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   [   9 7 % ] 
 
 . . . .                                                                                                                                           [ 1 0 0 % ] 
 
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =   w a r n i n g s   s u m m a r y   = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
 
 d o m a i n s / h i g h w a y / t e s t s / t e s t _ s u r r o g a t e _ m u l t i t a r g e t . p y : : t e s t _ h i g h w a y _ m u l t i _ t a r g e t _ f i t _ p r e d i c t 
 
     F : \ C o d e \ r a i s e - p a p e r - b a s e l i n e \ r a i s e _ e n v \ r a i s e _ c o r e \ s u r r o g a t e \ m o d e l . p y : 1 4 5 :   C o n s t a n t I n p u t W a r n i n g :   A n   i n p u t   a r r a y   i s   c o n s t a n t ;   t h e   c o r r e l a t i o n   c o e f f i c i e n t   i s   n o t   d e f i n e d . 
 
         c o e f ,   _   =   s p e a r m a n r ( a ,   b ) 
 
 
 
 d o m a i n s / c r o w d n a v / t e s t s / t e s t _ s t a g e 1 _ r e a l _ d a t a s e t . p y : : t e s t _ c o l l e c t _ s t a g e 1 _ d a t a s e t _ s m o k e 
 
     C : \ U s e r s \ n e m a t k h a h \ A p p D a t a \ L o c a l \ P r o g r a m s \ P y t h o n \ P y t h o n 3 1 2 \ L i b \ s i t e - p a c k a g e s \ g y m \ l o g g e r . p y : 3 0 :   U s e r W a r n i n g :    [ 3 3 m W A R N :   B o x   b o u n d   p r e c i s i o n   l o w e r e d   b y   c a s t i n g   t o   f l o a t 3 2  [ 0 m 
 
         w a r n i n g s . w a r n ( c o l o r i z e ( ' % s :   % s ' % ( ' W A R N ' ,   m s g   %   a r g s ) ,   ' y e l l o w ' ) ) 
 
 
 
 - -   D o c s :   h t t p s : / / d o c s . p y t e s t . o r g / e n / s t a b l e / h o w - t o / c a p t u r e - w a r n i n g s . h t m l 
 
 1 4 8   p a s s e d ,   2   w a r n i n g s   i n   2 4 . 1 8 s 
 
 
```

**148 passed**, 2 warnings (surrogate ConstantInputWarning; gym Box float32). CrowdNav suite green.

---

## SECTION 6 - Open risks / caveats

1. **`K_GATE=6` not re-tuned on a long LLM-backed run.** Section 2 shape looks reasonable (midpoint 17.5, Δ20→25 = 0.22, target gate ≤ 0.97), but selection dynamics under real mutations remain empirically open.

2. **Live `llm_diagnosis` was 100% null** in the PPO run (seed LLM / no API keys). Plumbing works; frontier self-diagnosis quality is **unverified** here.

3. **Checkpoint `reflection` can show `fitness=-inf` / `nan` speeds** when built on the next generation *before* Stage II re-attaches `last_metrics`. Proxy evidence on labeled parents still attaches; global reflection channel is weaker until rebuilt post-label.

4. **CrowdNav advice strings remain in `explore.py`** behind `_is_highway_pack()` - intentional parity; regression risk if the guard is removed.

5. **Remaining hand-coded thresholds (out of prompt-feedback scope):** `LAG_SPEED_MPS=18` in fitness lag penalty; degeneracy penalties in `metrics.py`; soft-success progress floor 400 m in `adapter.py`.

6. **Stage I used smoke Score1** (no local highway dataset) - fine for gate/PPO verification, not for Score1↔fitness claims.

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

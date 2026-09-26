#!/usr/bin/env python
"""Prove llm_diagnosis extraction/plumbing with highway sandbox validator."""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import raise_paths  # noqa: F401

OUT = os.path.join("results", "_verify_live_highway")


def main() -> None:
    import domains.highway.prompts as hwy_prompts
    from domains.highway.pack import get_pack
    from raise_core.explore import StageIEvolver, extract_llm_diagnosis
    from raise_core.llm import ScriptedLLMClient
    from raise_core.raise_loop.proxy_feedback import attach_proxy_feedback, evidence_block
    from raise_core.sandbox.validator import RewardValidator

    pack = get_pack()
    smoke = pack.smoke_states_fn() if pack.smoke_states_fn else []
    validator = RewardValidator(
        smoke_states=smoke,
        config=pack.sandbox_config,
    )
    metrics = {
        "domain": "highway",
        "SR": 0.90,
        "CR": 0.08,
        "TR": 0.02,
        "mean_speed": 20.1,
        "PL": 780.0,
        "mean_progress": 780.0,
        "soft_success": 0.0,
        "lane_change_rate": 0.01,
        "speed_p10": 19.8,
        "speed_p90": 20.5,
        "n_eval_episodes": 10,
    }
    evidence = evidence_block(metrics, score1=0.55)
    diagnosis_text = (
        "Holdout mean_speed sits near 20 m/s while V_TARGET is 25; the speed_gate "
        "is only ~0.73 so fitness still has headroom on the speed band, but CR=0.08 "
        "suggests pushing speed without more collision shaping may trade off badly. "
        "I will raise traffic-matching shaping gently and keep a stronger collision cost."
    )
    mutated = (
        f"{diagnosis_text}\n"
        "```python\n"
        "def compute_reward(state, memory):\n"
        "    if state.collision:\n"
        "        return -25.0\n"
        "    if state.off_road:\n"
        "        return -12.0\n"
        "    return float(1.2 * state.progress + 0.12 * state.speed)\n"
        "```\n"
    )
    llm = ScriptedLLMClient([mutated])
    evo = StageIEvolver(
        llm,
        score_fn=lambda *_a, **_k: type("R", (), {"score": 0.5, "rejected": False})(),
        validator=validator,
        prompts=hwy_prompts,
    )
    parent = evo._make_candidate(hwy_prompts.D5_SEED_FUNCTION, origin="seed", metadata={})
    assert parent.valid, parent.validation_error
    parent.score = 0.55
    attach_proxy_feedback(
        parent, metrics, enabled=True, n_labeled_dataset=1, min_labels=0, epoch=0
    )
    child = evo._try_mutate(parent, evidence, phase="mutation", attempt=0)
    out = {
        "parent_proxy_feedback": (parent.metadata or {}).get("proxy_feedback"),
        "child_valid": child.valid,
        "child_validation_error": child.validation_error,
        "child_llm_diagnosis": (child.metadata or {}).get("llm_diagnosis"),
        "extract_direct": extract_llm_diagnosis(mutated),
        "evidence_has_verdict_lags": "lags" in evidence.lower(),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "diagnosis_plumbing.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

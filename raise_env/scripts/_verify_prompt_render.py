#!/usr/bin/env python
"""Render real D1 fitness block + D2 mutation prompt for verification report."""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from domains.highway import prompts as hwy_prompts
from raise_core.raise_loop.proxy_feedback import evidence_block


# Representative holdout metrics from a prior ~20 m/s plateau regime
# (numbers match the failure mode the gate fix targets).
REAL_METRICS = {
    "domain": "highway",
    "SR": 0.95,
    "CR": 0.05,
    "TR": 0.00,
    "mean_speed": 20.06,
    "mean_progress": 803.0,
    "PL": 803.0,
    "soft_success": 0.0,
    "lane_change_rate": 0.0,
    "speed_p10": 20.06,
    "speed_p90": 20.06,
    "progress_std": 0.0,
    "n_eval_episodes": 20,
    "fitness": None,  # filled by fitness_components inside evidence_block
}


def main() -> None:
    evidence = evidence_block(REAL_METRICS, score1=0.612)
    print("===== EVIDENCE_BLOCK =====")
    print(evidence)
    print()
    print("===== D1_SYSTEM_PROMPT (fitness formula block) =====")
    print(hwy_prompts.D1_SYSTEM_PROMPT)
    print()
    parent_code = hwy_prompts.D5_SEED_FUNCTION
    reflection = (
        "Gen1 best_fitness_trend=[0.412, 0.455] "
        "best_mean_speed_trend=[20.060, 20.100]. "
        "Population(fitness desc): a: SR=0.95 CR=0.05 mean_speed=20.1 "
        "fitness=0.455; b: SR=0.80 CR=0.10 mean_speed=18.5 fitness=0.310. "
        "PriorLLMDiagnoses: (none)\n"
        f"Parent metrics evidence:\n{evidence}"
    )
    d2 = hwy_prompts.format_d2_mutation(parent_code, reflection)
    print("===== D2_MUTATION FULL RENDER =====")
    print(d2)
    print()
    d3 = hwy_prompts.format_d3_refinement(
        parent_code,
        last_score=0.455,
        feedback=evidence,
        extra_context_if_any="In-loop proxy refine (short Stage II metrics).",
    )
    print("===== D3_USER FULL RENDER =====")
    print(d3)
    banned = ("lags", "hack", "degenerate", "improve", "strengthen", "weakness focus")
    blob = (evidence + "\n" + d2 + "\n" + d3).lower()
    # Note: instruction text in prompts may contain 'degenerate' as an example
    # in the diagnose-before-code instruction — report that separately.
    print()
    print("===== BANNED-WORD SCAN (evidence_block only) =====")
    elow = evidence.lower()
    for w in banned:
        print(f"  evidence contains {w!r}: {w in elow}")


if __name__ == "__main__":
    main()

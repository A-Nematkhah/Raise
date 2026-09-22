"""
Pre-flight checks shared by the real (non-stub) RAISE run scripts.

A long overnight run must not die hours in on a missing GST checkpoint or an
absent Score1 dataset, so every check returns copy-pasteable remediation lines
instead of letting Stage II raise deep in the stack.
"""

from __future__ import annotations

import os
from typing import List, Sequence

# Score1 dataset shipped with the sibling baseline checkout.
BASELINE_STAGE1_DATASET = (
    r"D:\Thesis\Implementation\Evonav_baseline\amfrs_env\data\stage1_dataset"
)


def check_stage1_dataset(root: str, path: str = "data/stage1_dataset") -> List[str]:
    """Score1 needs the collected scenario dataset (npz archive or jsonl dir)."""
    target = os.path.normpath(path if os.path.isabs(path) else os.path.join(root, path))
    npz = os.path.join(target, "stage1_dataset.npz")
    if os.path.isfile(npz):
        return []
    if os.path.isdir(target) and any(
        name.startswith("scenario_") and name.endswith(".jsonl")
        for name in os.listdir(target)
    ):
        return []
    return [
        f"Missing Score1 dataset: {npz}",
        "  Copy it from the baseline checkout:",
        f'    Copy-Item -Recurse -Force "{BASELINE_STAGE1_DATASET}" "{target}"',
        "  or collect a fresh one (runs the simulator, slow):",
        f"    python scripts/collect_stage1_dataset.py --out {path} "
        "--n-scenarios 100 --n-traj 10 --human-num 20",
    ]


def check_gst_model(root: str, regime: str) -> List[str]:
    """Stage II/III with predict_method=inferred loads a GST checkpoint tree."""
    from crowd_nav.reward_search.regime import gst_model_dir_for_regime

    rel = gst_model_dir_for_regime(regime)
    target = os.path.normpath(os.path.join(root, rel))
    if os.path.isdir(target) and os.path.isfile(
        os.path.join(target, "checkpoint", "args.pickle")
    ):
        return []
    return [
        f"Missing GST predictor for regime={regime}: {rel}",
        f"  Expected directory: {target}",
        "  It must contain checkpoint/args.pickle (+ the trained weights).",
        "  Fetch it via gst_updated/run/download_datasets_models.sh, or copy the",
        "  tree from another checkout (see gst_updated/results/README.md).",
    ]


def check_groq_keys(root: str, llm: str) -> List[str]:
    """The Groq provider needs either a key file or GROQ_API_KEY."""
    if str(llm).strip().lower() != "groq":
        return []
    keys = os.path.join(root, "groq_keys.json")
    if os.path.isfile(keys) or os.environ.get("GROQ_API_KEY"):
        return []
    return [
        "Groq LLM selected but no credentials found.",
        f"  Create {keys} (list of API keys) or set GROQ_API_KEY,",
        "  or pass --llm ollama / --llm vllm for a local provider.",
    ]


def report_problems(problems: Sequence[str], *, stream) -> bool:
    """Print a problem block; return True when there was nothing to report."""
    if not problems:
        return True
    print("Prerequisites missing - refusing to start:", file=stream)
    for line in problems:
        print(f"  {line}" if not line.startswith(" ") else line, file=stream)
    return False

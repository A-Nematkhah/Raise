#!/usr/bin/env python
"""Evaluate a trained RAISE checkpoint on the fixed 17-human split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import raise_paths  # noqa: F401

from domains.crowdnav.adapter import (  # noqa: E402
    CrowdNavPPOTrainer,
    metrics_to_crowdnav_dict,
)
from domains.crowdnav.reward_checks import make_crowdnav_validator  # noqa: E402
from raise_core.candidate import RewardCandidate  # noqa: E402
from raise_config import CONFIG, build_upstream_config  # noqa: E402
from raise_core.generalization_eval import resolve_stage3_checkpoint
from raise_core.validate import Stage3Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed RAISE run directory")
    parser.add_argument(
        "--evaluation-variant",
        default="generalization_17",
        help="raise_config.evaluation_variants key (default: fixed 17 humans)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", default=None, help="Metrics JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    run_dir = Path(args.run_dir).resolve()
    candidate_path = run_dir / "final_candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Final candidate file not found: {candidate_path}")
    variant = str(args.evaluation_variant).strip()
    upstream = build_upstream_config(CONFIG.env, variant=variant, config=CONFIG)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    code = str(payload.get("code") or "")
    candidate_id = str(payload.get("candidate_id") or "final_candidate")
    metadata = dict(payload.get("metadata") or {})
    metadata["checkpoint_path"] = str(
        resolve_stage3_checkpoint(metadata.get("checkpoint_path"), run_dir)
    )
    reward_fn, error = make_crowdnav_validator().try_validate(code)
    if reward_fn is None:
        raise ValueError(f"Final candidate reward failed validation: {error}")
    candidate = RewardCandidate(
        candidate_id=candidate_id,
        code=code,
        reward_fn=reward_fn,
        score=payload.get("score"),
        valid=True,
        origin=str(payload.get("origin") or "final"),
        parent_ids=tuple(payload.get("parent_ids") or ()),
        metadata=metadata,
    )
    config = Stage3Config(
        eval_episodes=args.episodes,
        seed=args.seed,
        output_root=str(run_dir / "generalization_eval"),
        device=args.device,
        robot_kinematics="unicycle",
        action_mode="continuous",
        env_config_obj=upstream,
        eval_mode="holdout_only",
        eval_split="test",
        eval_seed_base=args.seed,
        eval_deterministic=False,
    )
    metrics = CrowdNavPPOTrainer(stage="stage3").reevaluate(
        candidate, round_index=0, config=config
    )
    results = metrics_to_crowdnav_dict(metrics)
    results["holdout_episodes"] = list(
        getattr(metrics, "_crowdnav_episodes", [])
    )
    results["generalization"] = {
        "candidate_id": candidate_id,
        "checkpoint_path": metadata["checkpoint_path"],
        "evaluation_variant": variant,
        "human_num": int(upstream.sim.human_num),
        "population": "fixed 17 humans",
        "split": "test",
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "robot_visible_to_humans": True,
    }
    output_path = Path(args.output).resolve() if args.output else run_dir / "generalization_17_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{candidate_id} on fixed 17-human generalization set: "
        f"SR={results['SR']:.3f} CR={results['CR']:.3f} "
        f"TR={results['TR']:.3f} fitness={results['fitness']:.3f}"
    )
    print(f"Metrics: {output_path}")
    return 0


if __name__ == "__main__":
    import os as _os
    import sys as _sys

    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from raise_core.script_utils import bootstrap_raise_env

    bootstrap_raise_env()
    raise SystemExit(main())

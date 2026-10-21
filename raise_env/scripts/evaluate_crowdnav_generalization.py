#!/usr/bin/env python
"""Evaluate a trained RAISE checkpoint on density / random generalization variants.

Variants come from ``raise_config.evaluation_variants`` (default: humans_10 /
humans_15 / humans_20, plus optional ``*_random`` for the EvoNav w/ random
protocol). Reports SR/CR/TR/NT/PL/ITR/SD.
"""

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

_REPORT_KEYS = ("SR", "CR", "TR", "NT", "PL", "ITR", "SD", "fitness")


def build_parser() -> argparse.ArgumentParser:
    known = sorted((CONFIG.evaluation_variants or {}).keys())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Completed RAISE run directory")
    parser.add_argument(
        "--evaluation-variant",
        default=None,
        help=(
            "Single raise_config.evaluation_variants key "
            f"(known: {known}). Omit with --all-variants."
        ),
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Evaluate every evaluation_variants key (H=10/15/20 ± random)",
    )
    parser.add_argument(
        "--densities",
        action="store_true",
        help="Evaluate humans_10, humans_15, humans_20 (no random)",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", default=None, help="Metrics JSON output path")
    return parser


def _resolve_variants(args: argparse.Namespace) -> list[str]:
    known = sorted((CONFIG.evaluation_variants or {}).keys())
    if args.all_variants:
        if not known:
            raise ValueError("CONFIG.evaluation_variants is empty")
        return known
    if args.densities:
        dens = [k for k in ("humans_10", "humans_15", "humans_20") if k in (CONFIG.evaluation_variants or {})]
        if not dens:
            raise ValueError("density variants humans_10/15/20 not in evaluation_variants")
        return dens
    variant = str(args.evaluation_variant or "humans_20").strip()
    if variant not in (CONFIG.evaluation_variants or {}):
        raise KeyError(
            f"unknown evaluation variant {variant!r}; known={known}"
        )
    return [variant]


def _load_candidate(run_dir: Path) -> RewardCandidate:
    candidate_path = run_dir / "final_candidate.json"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Final candidate file not found: {candidate_path}")
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
    return RewardCandidate(
        candidate_id=candidate_id,
        code=code,
        reward_fn=reward_fn,
        score=payload.get("score"),
        valid=True,
        origin=str(payload.get("origin") or "final"),
        parent_ids=tuple(payload.get("parent_ids") or ()),
        metadata=metadata,
    )


def _eval_variant(
    candidate: RewardCandidate,
    *,
    variant: str,
    run_dir: Path,
    episodes: int,
    seed: int,
    device: str,
) -> dict:
    upstream = build_upstream_config(CONFIG.env, variant=variant, config=CONFIG)
    config = Stage3Config(
        eval_episodes=episodes,
        seed=seed,
        output_root=str(run_dir / "generalization_eval" / variant),
        device=device,
        robot_kinematics=str(CONFIG.env.kinematics),
        action_mode=str(CONFIG.env.action_mode),
        env_config_obj=upstream,
        eval_mode="holdout_only",
        eval_split="test",
        eval_seed_base=seed,
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
        "candidate_id": candidate.candidate_id,
        "checkpoint_path": candidate.metadata.get("checkpoint_path"),
        "evaluation_variant": variant,
        "human_num": int(upstream.sim.human_num),
        "randomize_attributes": bool(getattr(upstream.env, "randomize_attributes", False)),
        "random_goal_changing": bool(
            getattr(upstream.humans, "random_goal_changing", False)
        ),
        "split": "test",
        "episodes": int(episodes),
        "seed": int(seed),
        "robot_kinematics": str(CONFIG.env.kinematics),
        "robot_visible_to_humans": bool(upstream.robot.visible),
    }
    return results


def _format_line(variant: str, results: dict) -> str:
    parts = " ".join(f"{k}={float(results.get(k, 0.0)):.3f}" for k in _REPORT_KEYS)
    g = results.get("generalization") or {}
    return (
        f"{variant} (H={g.get('human_num')}, "
        f"visible={g.get('robot_visible_to_humans')}, "
        f"random={g.get('randomize_attributes')}): {parts}"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if sum(bool(x) for x in (args.all_variants, args.densities, args.evaluation_variant)) > 1:
        raise ValueError("use only one of --evaluation-variant / --all-variants / --densities")
    run_dir = Path(args.run_dir).resolve()
    variants = _resolve_variants(args)
    candidate = _load_candidate(run_dir)
    all_results: dict[str, dict] = {}
    for variant in variants:
        results = _eval_variant(
            candidate,
            variant=variant,
            run_dir=run_dir,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
        )
        all_results[variant] = results
        print(_format_line(variant, results))
        out = (
            Path(args.output).resolve()
            if args.output and len(variants) == 1
            else run_dir / f"generalization_{variant}_eval.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Metrics: {out}")
    if len(variants) > 1:
        summary_path = run_dir / "generalization_summary.json"
        summary = {
            variant: {k: all_results[variant].get(k) for k in _REPORT_KEYS}
            | {"generalization": all_results[variant].get("generalization")}
            for variant in variants
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Summary: {summary_path}")
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

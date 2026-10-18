#!/usr/bin/env python
"""P1-1 baselines for a finished RAISE CrowdNav run.

Baseline A: PPO on the native seed reward (D5_SEED_FUNCTION), K3 steps, S seeds.
Baseline B: best non-collapsed Stage-II genome from pop_epoch_* (+ optional --extra-ids).
Baseline C: ORCA + Linear reference controllers on the same final-eval test cases.

Everything is scored with the final-eval protocol (split=test, seed+FINAL_EVAL_SEED_OFFSET,
stochastic policy mode) and compared against the run's final candidate via paired
per-case differences.

Examples (from ``raise_env/``)::

    python scripts/run_baselines.py --run-dir results/crowdnav_16h
    python scripts/run_baselines.py --run-dir results/crowdnav_16h --seeds 3 --extra-ids ini_0000
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse  # noqa: E402
import glob  # noqa: E402
import importlib.util  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from dataclasses import replace  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

from domains.crowdnav.adapter import (  # noqa: E402
    CrowdNavPPOTrainer,
    metrics_to_crowdnav_dict,
)
from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv, eval_cases  # noqa: E402
from domains.crowdnav.metrics import paired_difference  # noqa: E402
from domains.crowdnav.prompts import D5_SEED_FUNCTION  # noqa: E402
from domains.crowdnav.reward_checks import make_crowdnav_validator  # noqa: E402
from domains.crowdnav.settings import settings_from_config, set_settings  # noqa: E402
from raise_core.candidate import RewardCandidate  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.json_io import atomic_write_json, read_json_dict  # noqa: E402
from raise_core.population_io import load_candidate_dict  # noqa: E402
from raise_core.validate import (  # noqa: E402
    FINAL_EVAL_SEED_OFFSET,
    FINAL_EVAL_SPLIT,
    STAGE3_STEPS,
    Stage3Config,
    stage2_is_collapsed,
)

os.chdir(_ROOT)

logger = logging.getLogger(__name__)

# Match the 16h-run blind eval budget (RaiseRunConfig default is 100).
DEFAULT_EVAL_EPISODES = 500


def _load_collect_module() -> Any:
    path = os.path.join(_ROOT, "scripts", "collect_crowdnav_stage1_dataset.py")
    spec = importlib.util.spec_from_file_location("collect_crowdnav_stage1_dataset", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _space_knobs(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_mode": str(run_cfg.get("action_mode") or "continuous"),
        "robot_kinematics": str(run_cfg.get("robot_kinematics") or "holonomic"),
        "action_speed_samples": int(run_cfg.get("action_speed_samples") or 5),
        "action_rotation_samples": int(run_cfg.get("action_rotation_samples") or 16),
        "env_config_path": run_cfg.get("env_config_path"),
    }


def _stage3_train_config(
    run_dir: str,
    run_cfg: Dict[str, Any],
    *,
    train_steps: int,
    seed: int,
    eval_episodes: int = 2,
    ppo_gamma: Optional[float] = None,
    output_subdir: str = "baselines_train",
) -> Stage3Config:
    if ppo_gamma is not None and output_subdir == "baselines_train":
        output_subdir = f"baselines_train_gamma{float(ppo_gamma):g}"
    return Stage3Config(
        train_env_steps=int(train_steps),
        eval_episodes=int(eval_episodes),
        seed=int(seed),
        output_root=os.path.join(run_dir, output_subdir),
        ppo_gamma=ppo_gamma,
        device=str(run_cfg.get("device") or "cpu"),
        resume=False,
        eval_mode=str(run_cfg.get("eval_mode") or "holdout_only"),
        eval_split="val",
        warm_start=False,
        n_envs=max(1, int(run_cfg.get("n_envs") or 1)),
        eval_deterministic=False,
        eval_seed_base=None,
        skip_collapsed=False,
        **_space_knobs(run_cfg),
    )


def _final_eval_config(
    train_cfg: Stage3Config,
    *,
    episodes: int,
    seed_base: int,
) -> Stage3Config:
    return replace(
        train_cfg,
        eval_episodes=int(episodes),
        eval_seed_base=int(seed_base),
        eval_split=FINAL_EVAL_SPLIT,
        eval_deterministic=False,
        eval_mode="holdout_only",
    )


def _episodes_from_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(metrics, dict):
        return None
    eps = metrics.get("holdout_episodes")
    if isinstance(eps, list):
        return eps
    holdout = metrics.get("holdout")
    if isinstance(holdout, dict):
        eps = holdout.get("holdout_episodes")
        if isinstance(eps, list):
            return eps
    return None


def _vs_final(
    baseline_eps: Optional[Sequence[Dict[str, Any]]],
    final_eps: Optional[Sequence[Dict[str, Any]]],
    *,
    baseline_fitness: float,
    final_fitness: float,
) -> Dict[str, Any]:
    """Paired comparison: delta = final - baseline (positive => RAISE better)."""
    row: Dict[str, Any] = {
        "delta_fitness": float(final_fitness) - float(baseline_fitness),
        "paired_se": None,
        "z": None,
        "n": None,
        "method": "fitness_gap_only",
    }
    if baseline_eps is None or final_eps is None:
        return row
    paired = paired_difference(baseline_eps, final_eps, key="task_return")
    if paired is None:
        paired = paired_difference(baseline_eps, final_eps, key="disc_return")
    if paired is None:
        return row
    mean = float(paired["mean"])
    se = float(paired["se"])
    row.update(
        delta_fitness=mean,
        paired_se=se,
        z=(mean / se) if se > 0.0 and math.isfinite(se) else None,
        n=int(paired["n"]),
        method="paired_task_return",
    )
    return row


def _candidate_row_to_reward(
    row: Dict[str, Any], *, validator: Any
) -> RewardCandidate:
    """Rebuild a RewardCandidate from a flattened pop_epoch / JSON row.

    ``validator=None`` skips the sandbox (``reward_fn`` stays None): each
    validation spawns a child that re-imports this script (~6 s on Windows).
    """
    md = dict(row.get("metadata") or {})
    for key in ("fitness", "last_metrics", "checkpoint_path", "train_steps"):
        if key in row and key not in md:
            md[key] = row[key]
    if validator is None:
        return RewardCandidate(
            candidate_id=str(row.get("candidate_id", "loaded")),
            code=str(row["code"]),
            score=row.get("score"),
            valid=True,
            origin=str(row.get("origin", "loaded")),
            parent_ids=tuple(row.get("parent_ids") or ()),
            validation_error=row.get("validation_error"),
            metadata=md,
        )
    payload = {
        "candidate_id": row.get("candidate_id", "loaded"),
        "code": row["code"],
        "score": row.get("score"),
        "valid": True,
        "origin": row.get("origin", "loaded"),
        "parent_ids": list(row.get("parent_ids") or []),
        "validation_error": row.get("validation_error"),
        "metadata": md,
    }
    return load_candidate_dict(payload, validator=validator)


def load_stage2_candidates(
    run_dir: str, *, validator: Any = None
) -> Dict[str, RewardCandidate]:
    """Latest Stage-II snapshot per id across ``closed_loop/pop_epoch_*.json``."""
    cl = os.path.join(run_dir, "closed_loop")
    by_id: Dict[str, Tuple[int, RewardCandidate]] = {}
    paths = []
    for path in glob.glob(os.path.join(cl, "pop_epoch_*.json")):
        base = os.path.basename(path)
        try:
            epoch = int(base.replace("pop_epoch_", "").replace(".json", ""))
        except ValueError:
            continue
        paths.append((epoch, path))
    for epoch, path in sorted(paths):
        data = read_json_dict(path) or {}
        for row in data.get("candidates") or []:
            if not isinstance(row, dict) or not row.get("code"):
                continue
            cid = str(row.get("candidate_id") or "")
            if not cid:
                continue
            cand = _candidate_row_to_reward(row, validator=validator)
            by_id[cid] = (epoch, cand)
    return {cid: cand for cid, (_ep, cand) in by_id.items()}


def select_baseline_b_ids(
    candidates: Dict[str, RewardCandidate],
    *,
    collapse_cr: float,
    extra_ids: Sequence[str],
) -> List[str]:
    """Best non-collapsed Stage-II id, plus any ``extra_ids`` that exist."""
    chosen: List[str] = []
    eligible = [
        c
        for c in candidates.values()
        if c.metadata.get("fitness") is not None
        and not stage2_is_collapsed(c, collapse_cr=float(collapse_cr))
    ]
    if eligible:
        best = max(eligible, key=lambda c: float(c.metadata.get("fitness")))
        chosen.append(str(best.candidate_id))
    for cid in extra_ids:
        cid = str(cid).strip()
        if not cid or cid in chosen:
            continue
        if cid in candidates:
            chosen.append(cid)
        else:
            logger.warning("extra-id %s not found in pop_epoch_*", cid)
    return chosen


def _summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    holdout = metrics.get("holdout") if isinstance(metrics.get("holdout"), dict) else metrics
    return {
        "fitness": float(metrics.get("fitness", float("nan"))),
        "fitness_se": metrics.get("fitness_se"),
        "SR": holdout.get("SR") if isinstance(holdout, dict) else metrics.get("SR"),
        "CR": holdout.get("CR") if isinstance(holdout, dict) else metrics.get("CR"),
        "TR": holdout.get("TR") if isinstance(holdout, dict) else metrics.get("TR"),
        "task_return": (
            holdout.get("task_return") if isinstance(holdout, dict) else metrics.get("task_return")
        ),
        "n_eval_episodes": metrics.get("n_eval_episodes"),
        "eval_context": metrics.get("eval_context"),
        "holdout_episodes": list(metrics.get("holdout_episodes") or []),
    }


def _train_and_final_eval(
    candidate: RewardCandidate,
    *,
    trainer: CrowdNavPPOTrainer,
    train_cfg: Stage3Config,
    final_cfg: Stage3Config,
    round_index: int,
) -> Dict[str, Any]:
    bundle = trainer.train_and_eval(candidate, round_index=round_index, config=train_cfg)
    ckpt = None
    if hasattr(bundle, "checkpoint_path"):
        ckpt = bundle.checkpoint_path
    ckpt = ckpt or (candidate.metadata or {}).get("checkpoint_path")
    holdout = trainer.reevaluate(candidate, round_index=0, config=final_cfg)
    metrics = dict((candidate.metadata or {}).get("last_metrics") or {})
    if not metrics and hasattr(holdout, "as_dict"):
        metrics = metrics_to_crowdnav_dict(holdout)
    summary = _summarize_metrics(metrics)
    summary["checkpoint_path"] = ckpt
    summary["candidate_id"] = str(candidate.candidate_id)
    summary["train_steps"] = int(train_cfg.train_env_steps)
    summary["train_seed"] = int(train_cfg.seed) + int(round_index)
    summary["ppo_gamma"] = (candidate.metadata or {}).get("ppo_gamma")
    return summary


def _eval_controller(
    behaviour: str,
    *,
    cases: Sequence[int],
    run_cfg: Dict[str, Any],
    collect_mod: Any,
) -> Dict[str, Any]:
    """Evaluate ORCA/Linear via the same metrics path as ``_eval_metrics``."""
    from domains.crowdnav.adapter import _eval_with_step
    from domains.crowdnav.metrics import attach_fitness

    settings = set_settings(settings_from_config(SimpleNamespace(**run_cfg)))
    env = RewardInjectedCrowdNavEnv(None, phase=FINAL_EVAL_SPLIT, settings=settings)
    rng = np.random.default_rng(0)
    try:
        # ORCA keeps an internal sim; build the controller lazily after each
        # ``_eval_with_step`` reset so it sees the post-reset joint state.
        ctrl_holder: Dict[str, Any] = {"fn": None}

        def _step(_obs: Any) -> Any:
            if ctrl_holder["fn"] is None:
                ctrl_holder["fn"] = collect_mod._make_controller(behaviour, env, rng)
            return env.step_upstream(ctrl_holder["fn"]())

        parts = []
        for case in cases:
            ctrl_holder["fn"] = None
            parts.append(_eval_with_step(env, _step, cases=[int(case)]))
        holdout = _merge_proxy_metrics(parts, time_limit=float(env.time_limit))
        metrics = metrics_to_crowdnav_dict(holdout)
        metrics["holdout_episodes"] = list(getattr(holdout, "_crowdnav_episodes", []))
        metrics["holdout"] = {
            k: v
            for k, v in metrics.items()
            if k not in ("holdout", "train_dist", "holdout_episodes", "eval_context")
        }
        metrics["eval_context"] = {
            "split": FINAL_EVAL_SPLIT,
            "case_set_size": int(env.sim.case_size[FINAL_EVAL_SPLIT]),
            "cases": [int(c) for c in cases],
            "human_num": int(env.human_num),
            "deterministic": True,
            "controller": behaviour,
        }
        metrics = attach_fitness(metrics)
        summary = _summarize_metrics(metrics)
        summary["controller"] = behaviour
        return summary
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def _merge_proxy_metrics(parts: Sequence[Any], *, time_limit: float) -> Any:
    """Concatenate per-case ProxyMetrics into one (same schema as ``_eval_metrics``)."""
    from domains.crowdnav.metrics import episode_fields, fitness_standard_error
    from raise_core.refine import ProxyMetrics

    records: List[Dict[str, Any]] = []
    for m in parts:
        records.extend(list(getattr(m, "_crowdnav_episodes", []) or []))
    outcomes = [str(r.get("outcome") or "timeout") for r in records]
    nav_times = [float(r.get("time") or 0.0) for r in records]
    task_returns = [float(r.get("task_return") or 0.0) for r in records]
    danger_counts = [int(r.get("danger_steps") or 0) for r in records]
    step_counts = [int(r.get("steps") or 0) for r in records]
    returns = [float(r.get("return") or 0.0) for r in records]
    min_seps = [float(r.get("min_separation") or 0.0) for r in records]
    success_paths = [
        float(r.get("path_length") or 0.0)
        for r in records
        if r.get("outcome") == "success"
    ]
    fields = episode_fields(
        outcomes,
        nav_times,
        task_returns,
        danger_counts,
        step_counts,
        [],
        time_limit=float(time_limit),
    )
    metrics = ProxyMetrics(
        sr=fields["SR"],
        cr=fields["CR"],
        tr=fields["TR"],
        nt=fields["NT"],
        pl=float(np.mean(success_paths)) if success_paths else 0.0,
        itr=fields["ITR"],
        sd=fields["SD"],
    )
    metrics._crowdnav_extras = {  # type: ignore[attr-defined]
        "task_return": fields["task_return"],
        "fitness_se": fitness_standard_error(task_returns),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_min_separation": float(np.mean(min_seps)) if min_seps else 0.0,
        "n_success": sum(1 for o in outcomes if o == "success"),
        "n_collision": sum(1 for o in outcomes if o == "collision"),
        "n_timeout": sum(1 for o in outcomes if o == "timeout"),
        "n_eval_episodes": len(outcomes),
    }
    metrics._crowdnav_episodes = records  # type: ignore[attr-defined]
    return metrics


def _load_final_reference(run_dir: str) -> Tuple[str, float, Optional[List[Dict[str, Any]]]]:
    final = read_json_dict(os.path.join(run_dir, "final_candidate.json")) or {}
    cid = str(final.get("candidate_id") or "unknown")
    md = dict(final.get("metadata") or {})
    fe = md.get("final_eval") if isinstance(md.get("final_eval"), dict) else None
    metrics = None
    fitness = float("-inf")
    if fe and fe.get("status") == "ok":
        fitness = float(fe.get("fitness", float("-inf")))
        metrics = fe.get("metrics") if isinstance(fe.get("metrics"), dict) else None
    if metrics is None and isinstance(md.get("last_metrics"), dict):
        metrics = md["last_metrics"]
        if fitness == float("-inf") and metrics.get("fitness") is not None:
            fitness = float(metrics["fitness"])
    return cid, fitness, _episodes_from_metrics(metrics)


def run_baselines(
    run_dir: str,
    *,
    seeds: int = 1,
    extra_ids: Optional[Sequence[str]] = None,
    eval_episodes: Optional[int] = None,
    train_steps: Optional[int] = None,
    collapse_cr: Optional[float] = None,
    skip_a: bool = False,
    skip_b: bool = False,
    skip_c: bool = False,
    ppo_gamma: Optional[float] = None,
) -> Dict[str, Any]:
    """Train/eval baselines for ``run_dir`` and write ``baselines.json``.

    A non-default ``ppo_gamma`` writes ``baselines_gamma<g>.json`` instead so
    the reference results are not overwritten.
    """
    run_dir = os.path.abspath(run_dir)
    run_cfg = read_json_dict(os.path.join(run_dir, "config.json")) or {}
    seed = int(run_cfg.get("seed", 425))
    k3 = int(train_steps if train_steps is not None else run_cfg.get("stage3_train_steps") or STAGE3_STEPS)
    if eval_episodes is not None:
        n_eval = max(1, int(eval_episodes))
    else:
        # Honour the run's blind-eval budget when present; else the P1-1 default (500).
        n_eval = int(run_cfg.get("final_eval_episodes") or DEFAULT_EVAL_EPISODES)
        if n_eval <= 0:
            n_eval = DEFAULT_EVAL_EPISODES
    cr_thresh = float(
        collapse_cr if collapse_cr is not None else run_cfg.get("collapse_cr") or 0.8
    )
    seed_base = int(seed) + FINAL_EVAL_SEED_OFFSET
    extra = [str(x) for x in (extra_ids or []) if str(x).strip()]

    final_id, final_fit, final_eps = _load_final_reference(run_dir)
    validator = make_crowdnav_validator()
    trainer = CrowdNavPPOTrainer(stage="stage3")
    results: Dict[str, Any] = {
        "run_dir": run_dir,
        "seed": seed,
        "seed_base": seed_base,
        "eval_episodes": n_eval,
        "eval_split": FINAL_EVAL_SPLIT,
        "eval_deterministic": False,
        "train_steps_k3": k3,
        "ppo_gamma": ppo_gamma,
        "final_candidate_id": final_id,
        "final_fitness": final_fit if math.isfinite(final_fit) else None,
        "baselines": {},
        "comparisons": {},
    }

    # ----- Baseline A: native reward PPO -----
    if not skip_a:
        fn = validator.validate_code(D5_SEED_FUNCTION)
        runs: List[Dict[str, Any]] = []
        for i in range(max(1, int(seeds))):
            cand = RewardCandidate(
                candidate_id=f"baseline_A_seed{i}",
                code=D5_SEED_FUNCTION,
                reward_fn=fn,
                valid=True,
                origin="baseline_A",
            )
            train_cfg = _stage3_train_config(
                run_dir, run_cfg, train_steps=k3, seed=seed + i, eval_episodes=2,
                ppo_gamma=ppo_gamma,
            )
            final_cfg = _final_eval_config(train_cfg, episodes=n_eval, seed_base=seed_base)
            summary = _train_and_final_eval(
                cand, trainer=trainer, train_cfg=train_cfg, final_cfg=final_cfg, round_index=0
            )
            runs.append(summary)
        best_a = max(runs, key=lambda r: float(r.get("fitness") or float("-inf")))
        entry = {
            "kind": "A_native_ppo",
            "n_seeds": len(runs),
            "runs": runs,
            "fitness": best_a.get("fitness"),
            "fitness_se": best_a.get("fitness_se"),
            "holdout_episodes": best_a.get("holdout_episodes"),
            "SR": best_a.get("SR"),
            "CR": best_a.get("CR"),
            "TR": best_a.get("TR"),
        }
        results["baselines"]["A_native_ppo"] = entry
        results["comparisons"]["A_native_ppo"] = _vs_final(
            entry.get("holdout_episodes"),
            final_eps,
            baseline_fitness=float(entry.get("fitness") or float("-inf")),
            final_fitness=final_fit,
        )

    # ----- Baseline B: Stage-II genomes -----
    if not skip_b:
        stage2 = load_stage2_candidates(run_dir)
        b_ids = select_baseline_b_ids(stage2, collapse_cr=cr_thresh, extra_ids=extra)
        for cid in b_ids:
            src = stage2[cid]
            key = f"B_{cid}"
            if cid == final_id and cid not in extra:
                # Same genome, seed and K3 as Stage III: retraining repeats it.
                logger.info("Baseline B pick %s is the final candidate; not retrained", cid)
                results["baselines"][key] = {
                    "kind": "B_stage2",
                    "source_id": cid,
                    "status": "same_as_final",
                    "stage2_fitness": (src.metadata or {}).get("fitness"),
                }
                continue
            cand = RewardCandidate(
                candidate_id=f"baseline_B_{cid}",
                code=src.code,
                reward_fn=validator.validate_code(src.code),
                valid=True,
                origin="baseline_B",
                parent_ids=(cid,),
            )
            train_cfg = _stage3_train_config(
                run_dir, run_cfg, train_steps=k3, seed=seed, eval_episodes=2,
                ppo_gamma=ppo_gamma,
            )
            final_cfg = _final_eval_config(train_cfg, episodes=n_eval, seed_base=seed_base)
            summary = _train_and_final_eval(
                cand, trainer=trainer, train_cfg=train_cfg, final_cfg=final_cfg, round_index=0
            )
            entry = {
                "kind": "B_stage2",
                "source_id": cid,
                "stage2_fitness": (src.metadata or {}).get("fitness"),
                **{k: summary.get(k) for k in (
                    "fitness", "fitness_se", "SR", "CR", "TR",
                    "holdout_episodes", "checkpoint_path", "train_steps", "train_seed",
                )},
            }
            results["baselines"][key] = entry
            results["comparisons"][key] = _vs_final(
                entry.get("holdout_episodes"),
                final_eps,
                baseline_fitness=float(entry.get("fitness") or float("-inf")),
                final_fitness=final_fit,
            )

    # ----- Baseline C: ORCA + Linear -----
    if not skip_c:
        collect_mod = _load_collect_module()
        # Probe case-set size from a throwaway env.
        settings = set_settings(settings_from_config(SimpleNamespace(**run_cfg)))
        probe = RewardInjectedCrowdNavEnv(None, phase=FINAL_EVAL_SPLIT, settings=settings)
        try:
            size = int(probe.sim.case_size[FINAL_EVAL_SPLIT])
        finally:
            probe.close()
        cases = eval_cases(seed_base, n_eval, size)
        for behaviour in ("orca", "linear"):
            summary = _eval_controller(
                behaviour, cases=cases, run_cfg=run_cfg, collect_mod=collect_mod
            )
            key = f"C_{behaviour}"
            entry = {
                "kind": "C_controller",
                "controller": behaviour,
                **{k: summary.get(k) for k in (
                    "fitness", "fitness_se", "SR", "CR", "TR", "holdout_episodes",
                )},
                "cases": [int(c) for c in cases],
            }
            results["baselines"][key] = entry
            results["comparisons"][key] = _vs_final(
                entry.get("holdout_episodes"),
                final_eps,
                baseline_fitness=float(entry.get("fitness") or float("-inf")),
                final_fitness=final_fit,
            )

    out_name = "baselines.json" if ppo_gamma is None else f"baselines_gamma{float(ppo_gamma):g}.json"
    out_path = os.path.join(run_dir, out_name)
    atomic_write_json(out_path, results)
    logger.info("Wrote %s", out_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAISE P1-1 baselines (CrowdNav)")
    p.add_argument("--run-dir", type=str, required=True, help="Finished RAISE output directory")
    p.add_argument("--seeds", type=int, default=1, help="Baseline A PPO seeds (default: 1)")
    p.add_argument(
        "--extra-ids",
        type=str,
        default="",
        help="Comma-separated Stage-II candidate ids to include in Baseline B",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=f"Final-eval episodes (default: run config or {DEFAULT_EVAL_EPISODES})",
    )
    p.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="K3 PPO steps for A/B (default: run config stage3_train_steps)",
    )
    p.add_argument("--collapse-cr", type=float, default=None)
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-b", action="store_true")
    p.add_argument("--skip-c", action="store_true")
    p.add_argument(
        "--ppo-gamma",
        type=float,
        default=None,
        help="PPO per-step discount for A/B (default: 0.99; writes baselines_gamma<g>.json)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    extra = [x.strip() for x in str(args.extra_ids or "").split(",") if x.strip()]
    run_baselines(
        args.run_dir,
        seeds=max(1, int(args.seeds)),
        extra_ids=extra,
        eval_episodes=args.eval_episodes,
        train_steps=args.train_steps,
        collapse_cr=args.collapse_cr,
        skip_a=bool(args.skip_a),
        skip_b=bool(args.skip_b),
        skip_c=bool(args.skip_c),
        ppo_gamma=args.ppo_gamma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

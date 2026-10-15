#!/usr/bin/env python
"""PPO training-discount sweep for a finished RAISE CrowdNav run.

The reward function and the blind evaluation are held fixed; only the PPO
discount changes. PPO discounts per policy step (dt = time_step), while the
CrowdNav fitness weighs step t by CROWDNAV_EVAL_GAMMA ** (t * time_step * v_pref)
(0.9 per second at v_pref = 1). The per-step gamma that matches the fitness is
therefore CROWDNAV_EVAL_GAMMA ** (time_step * v_pref) (~0.974 for the upstream
config); the pipeline default 0.99 corresponds to a ~2.6x longer horizon.

Every (reward, gamma, seed) run trains for K3 steps and is scored with the
final-eval protocol (split=test, same cases, stochastic policy). Training seeds
are shared across gammas so results can be compared per seed and per case.
Results are written to ``<run-dir>/gamma_sweep.json`` after every run and
finished runs are skipped on restart.

Examples (from ``raise_env/``)::

    python scripts/run_gamma_sweep.py --run-dir results/crowdnav_16h
    python scripts/run_gamma_sweep.py --run-dir results/crowdnav_16h \\
        --gammas matched,0.99,0.995 --rewards native,final --seeds 3
"""

from __future__ import annotations

import os as _os
import sys as _sys

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: E402

from domains.crowdnav.adapter import CrowdNavPPOTrainer  # noqa: E402
from domains.crowdnav.env_wrapper import load_env_config  # noqa: E402
from domains.crowdnav.metrics import paired_difference  # noqa: E402
from domains.crowdnav.objective_constants import CROWDNAV_EVAL_GAMMA, PPO_GAMMA  # noqa: E402
from domains.crowdnav.prompts import D5_SEED_FUNCTION  # noqa: E402
from domains.crowdnav.reward_checks import make_crowdnav_validator  # noqa: E402
from domains.crowdnav.settings import DEFAULT_ENV_CONFIG_PATH  # noqa: E402
from raise_core.candidate import RewardCandidate  # noqa: E402
from raise_core.console import ensure_utf8_stdio  # noqa: E402
from raise_core.json_io import atomic_write_json, read_json_dict  # noqa: E402
from raise_core.validate import FINAL_EVAL_SEED_OFFSET, FINAL_EVAL_SPLIT, STAGE3_STEPS  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_GAMMAS = "matched,0.99,0.995"
DEFAULT_REWARDS = "native,final"
OUT_NAME = "gamma_sweep.json"
TRAIN_SUBDIR = "gamma_sweep_train"


def _load_baselines_module() -> Any:
    path = os.path.join(_ROOT, "scripts", "run_baselines.py")
    spec = importlib.util.spec_from_file_location("run_baselines", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def matched_gamma(env_config_path: Optional[str]) -> float:
    """Per-step PPO gamma equal to the fitness discount for this env config."""
    cfg = load_env_config(env_config_path or DEFAULT_ENV_CONFIG_PATH)
    time_step = cfg.getfloat("env", "time_step")
    v_pref = cfg.getfloat("robot", "v_pref")
    return float(CROWDNAV_EVAL_GAMMA ** (time_step * v_pref))


def parse_gammas(text: str, *, matched: float) -> List[float]:
    out: List[float] = []
    for tok in str(text).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        g = round(matched, 4) if tok == "matched" else float(tok)
        if not (0.0 < g <= 1.0):
            raise ValueError(f"gamma must be in (0, 1], got {tok!r}")
        if g not in out:
            out.append(g)
    if not out:
        raise ValueError("no gammas given")
    return out


def resolve_rewards(run_dir: str, labels: Sequence[str], rb: Any) -> Dict[str, str]:
    """Reward label -> code. ``native`` = seed reward, ``final`` = run winner, else a Stage-II id."""
    out: Dict[str, str] = {}
    stage2: Optional[Dict[str, RewardCandidate]] = None
    for label in labels:
        if label == "native":
            out[label] = D5_SEED_FUNCTION
        elif label == "final":
            final = read_json_dict(os.path.join(run_dir, "final_candidate.json")) or {}
            if not final.get("code"):
                raise ValueError(f"{run_dir}/final_candidate.json has no code")
            out[label] = str(final["code"])
        else:
            if stage2 is None:
                stage2 = rb.load_stage2_candidates(run_dir)
            if label not in stage2:
                raise ValueError(f"reward {label!r} not found in closed_loop/pop_epoch_*")
            out[label] = stage2[label].code
    return out


def run_key(reward: str, gamma: float, seed_index: int) -> str:
    return f"{reward}|g{gamma:g}|s{seed_index}"


def summarize(
    runs: Dict[str, Dict[str, Any]],
    *,
    rewards: Sequence[str],
    gammas: Sequence[float],
    n_seeds: int,
    reference_gamma: float,
) -> Dict[str, Any]:
    """Per (reward, gamma) seed means, plus per-seed paired deltas vs ``reference_gamma``."""
    table: Dict[str, Dict[str, Any]] = {}
    for reward in rewards:
        rows: Dict[str, Any] = {}
        for g in gammas:
            done = [runs[k] for k in (run_key(reward, g, i) for i in range(n_seeds)) if k in runs]
            fits = [float(r["fitness"]) for r in done if r.get("fitness") is not None]
            row: Dict[str, Any] = {"n_seeds": len(fits)}
            if fits:
                row["fitness_mean"] = statistics.fmean(fits)
                row["fitness_se_seeds"] = (
                    statistics.stdev(fits) / math.sqrt(len(fits)) if len(fits) > 1 else None
                )
                for m in ("SR", "CR", "TR"):
                    vals = [float(r[m]) for r in done if r.get(m) is not None]
                    row[m] = statistics.fmean(vals) if vals else None
            if g != reference_gamma:
                paired = []
                for i in range(n_seeds):
                    a = runs.get(run_key(reward, reference_gamma, i))
                    b = runs.get(run_key(reward, g, i))
                    if not a or not b:
                        continue
                    d = paired_difference(
                        a.get("holdout_episodes") or [], b.get("holdout_episodes") or []
                    )
                    if d is None:
                        continue
                    se = float(d["se"])
                    paired.append({
                        "seed_index": i,
                        "delta": float(d["mean"]),
                        "se": se,
                        "z": float(d["mean"]) / se if se > 0.0 else None,
                        "n": int(d["n"]),
                    })
                row["vs_reference"] = paired
            rows[f"{g:g}"] = row
        table[reward] = rows
    return table


def run_gamma_sweep(
    run_dir: str,
    *,
    gammas: str = DEFAULT_GAMMAS,
    rewards: str = DEFAULT_REWARDS,
    seeds: int = 2,
    train_steps: Optional[int] = None,
    eval_episodes: Optional[int] = None,
    reference_gamma: float = PPO_GAMMA,
    out_name: str = OUT_NAME,
) -> Dict[str, Any]:
    rb = _load_baselines_module()
    run_dir = os.path.abspath(run_dir)
    run_cfg = read_json_dict(os.path.join(run_dir, "config.json")) or {}
    seed = int(run_cfg.get("seed", 425))
    k3 = int(train_steps if train_steps is not None else run_cfg.get("stage3_train_steps") or STAGE3_STEPS)
    if eval_episodes is not None:
        n_eval = max(1, int(eval_episodes))
    else:
        n_eval = int(run_cfg.get("final_eval_episodes") or rb.DEFAULT_EVAL_EPISODES)
        if n_eval <= 0:
            n_eval = rb.DEFAULT_EVAL_EPISODES
    seed_base = int(seed) + FINAL_EVAL_SEED_OFFSET
    n_seeds = max(1, int(seeds))

    g_match = matched_gamma(run_cfg.get("env_config_path"))
    gamma_list = parse_gammas(gammas, matched=g_match)
    ref = float(reference_gamma)
    if ref not in gamma_list:
        gamma_list.append(ref)
    labels = [x.strip() for x in str(rewards).split(",") if x.strip()]
    codes = resolve_rewards(run_dir, labels, rb)

    out_path = os.path.join(run_dir, out_name)
    previous = read_json_dict(out_path) or {}
    runs: Dict[str, Dict[str, Any]] = dict(previous.get("runs") or {})
    # Resumed runs only count when train budget and eval protocol match.
    if previous and (
        previous.get("train_steps_k3") != k3
        or previous.get("eval_episodes") != n_eval
        or previous.get("seed_base") != seed_base
    ):
        logger.warning("%s was produced with a different budget/protocol; starting fresh", out_path)
        runs = {}

    validator = make_crowdnav_validator()
    trainer = CrowdNavPPOTrainer(stage="stage3")
    fns = {label: validator.validate_code(code) for label, code in codes.items()}

    results: Dict[str, Any] = {
        "run_dir": run_dir,
        "seed": seed,
        "seed_base": seed_base,
        "eval_split": FINAL_EVAL_SPLIT,
        "eval_episodes": n_eval,
        "eval_deterministic": False,
        "train_steps_k3": k3,
        "eval_gamma_per_second": CROWDNAV_EVAL_GAMMA,
        "matched_gamma": g_match,
        "reference_gamma": ref,
        "gammas": gamma_list,
        "rewards": labels,
        "n_seeds": n_seeds,
        "runs": runs,
    }

    plan: List[Tuple[str, float, int]] = [
        (label, g, i) for i in range(n_seeds) for label in labels for g in gamma_list
    ]
    todo = [p for p in plan if run_key(*p) not in runs]
    logger.info(
        "gamma sweep: %d runs planned, %d already done, %d to train (K3=%d, eval=%d)",
        len(plan), len(plan) - len(todo), len(todo), k3, n_eval,
    )
    for label, g, i in todo:
        key = run_key(label, g, i)
        cand = RewardCandidate(
            candidate_id=f"gamma_{label}_g{g:g}_s{i}",
            code=codes[label],
            reward_fn=fns[label],
            valid=True,
            origin="gamma_sweep",
        )
        train_cfg = rb._stage3_train_config(
            run_dir, run_cfg, train_steps=k3, seed=seed + i, eval_episodes=2,
            ppo_gamma=g, output_subdir=TRAIN_SUBDIR,
        )
        final_cfg = rb._final_eval_config(train_cfg, episodes=n_eval, seed_base=seed_base)
        summary = rb._train_and_final_eval(
            cand, trainer=trainer, train_cfg=train_cfg, final_cfg=final_cfg, round_index=0
        )
        summary.update(reward=label, gamma=g, seed_index=i)
        runs[key] = summary
        results["summary"] = summarize(
            runs, rewards=labels, gammas=gamma_list, n_seeds=n_seeds, reference_gamma=ref
        )
        atomic_write_json(out_path, results)
        logger.info("gamma sweep %s: fitness=%.4f SR=%s", key, float(summary["fitness"]), summary.get("SR"))

    results["summary"] = summarize(
        runs, rewards=labels, gammas=gamma_list, n_seeds=n_seeds, reference_gamma=ref
    )
    atomic_write_json(out_path, results)
    logger.info("Wrote %s", out_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAISE CrowdNav PPO discount sweep")
    p.add_argument("--run-dir", type=str, required=True, help="Finished RAISE output directory")
    p.add_argument(
        "--gammas",
        type=str,
        default=DEFAULT_GAMMAS,
        help="Comma list of per-step PPO gammas; 'matched' = fitness-equivalent gamma",
    )
    p.add_argument(
        "--rewards",
        type=str,
        default=DEFAULT_REWARDS,
        help="Comma list: native (seed reward), final (run winner) or Stage-II candidate ids",
    )
    p.add_argument("--seeds", type=int, default=2, help="PPO train seeds per (reward, gamma)")
    p.add_argument("--train-steps", type=int, default=None, help="Default: run stage3_train_steps")
    p.add_argument("--eval-episodes", type=int, default=None, help="Default: run final_eval_episodes")
    p.add_argument(
        "--reference-gamma",
        type=float,
        default=PPO_GAMMA,
        help=f"Gamma the paired deltas are computed against (default {PPO_GAMMA})",
    )
    p.add_argument("--out-name", type=str, default=OUT_NAME)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdio()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_gamma_sweep(
        args.run_dir,
        gammas=args.gammas,
        rewards=args.rewards,
        seeds=args.seeds,
        train_steps=args.train_steps,
        eval_episodes=args.eval_episodes,
        reference_gamma=args.reference_gamma,
        out_name=args.out_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

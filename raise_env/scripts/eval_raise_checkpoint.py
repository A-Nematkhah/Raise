#!/usr/bin/env python
"""
Eval-only re-evaluation of an RAISE Stage III checkpoint (no training).

Use this to lock in best-ever policies discovered mid-run (e.g. R0 peak)
without trusting last-round selection or fixed test_case GIFs.

Examples::

    # Re-eval the known R0 peak from run_scaled_h5_gst (SR was ~0.68 @ E=150)
    python scripts/eval_raise_checkpoint.py \\
        --run-dir results/run_scaled_h5_gst \\
        --candidate-id mut_0060_v2 \\
        --round 0 \\
        --episodes 150 \\
        --device cuda

    # Auto-pick best-ever from stage3 history, then re-eval
    python scripts/eval_raise_checkpoint.py \\
        --run-dir results/run_scaled_h5_gst \\
        --best-ever \\
        --episodes 150 \\
        --device cuda
"""

from __future__ import annotations

import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import raise_paths  # noqa: E402,F401 — arms domains/crowdnav/runtime on sys.path
import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_BASELINES_ROOT = os.path.abspath(os.path.join(_ROOT, "..", "baselines_openai"))
if _BASELINES_ROOT not in sys.path:
    sys.path.insert(0, _BASELINES_ROOT)
os.chdir(_ROOT)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scalar(m: Dict[str, Any]) -> float:
    return float(m.get("SR", 0) - m.get("CR", 0) - 0.5 * m.get("TR", 0))


def _find_candidate_payload(
    run_dir: str, candidate_id: str
) -> Tuple[Dict[str, Any], str]:
    """Locate reward code JSON for ``candidate_id`` (prefer Stage II pop)."""
    search_order = [
        os.path.join(run_dir, "stage2_population.json"),
        os.path.join(run_dir, "stage3_population.json"),
        os.path.join(run_dir, "best_stage2.json"),
        os.path.join(run_dir, "best_stage3.json"),
        os.path.join(run_dir, "final_candidate.json"),
    ]
    for path in search_order:
        if not os.path.isfile(path):
            continue
        data = _load_json(path)
        items: List[Dict[str, Any]]
        if "population" in data:
            items = list(data["population"])
        else:
            items = [data]
        for cand in items:
            if str(cand.get("candidate_id")) == candidate_id:
                return cand, path
    raise FileNotFoundError(
        f"No candidate JSON with id={candidate_id!r} under {run_dir} "
        f"(checked stage2/3 population and best_*.json)"
    )


def _best_ever_from_history(run_dir: str) -> Tuple[str, int, Dict[str, Any]]:
    pop_path = os.path.join(run_dir, "stage3_population.json")
    if not os.path.isfile(pop_path):
        raise FileNotFoundError(f"Missing {pop_path}")
    hist = _load_json(pop_path).get("history") or []
    if not hist:
        raise ValueError(f"No Stage III history in {pop_path}")
    best = max(hist, key=lambda h: _scalar(h.get("metrics") or {}))
    return (
        str(best["candidate_id"]),
        int(best["round_index"]),
        dict(best.get("metrics") or {}),
    )


def _resolve_train_dir(run_dir: str, candidate_id: str, round_index: int) -> str:
    folder = os.path.join(
        run_dir, "stage3_train", f"r{round_index:02d}_{candidate_id}"
    )
    if os.path.isdir(folder):
        return folder
    # Fallback: any folder matching candidate id for that round prefix.
    pattern = os.path.join(run_dir, "stage3_train", f"r{round_index:02d}_*{candidate_id}*")
    hits = sorted(glob.glob(pattern))
    if hits:
        return hits[0]
    raise FileNotFoundError(
        f"No Stage III train dir for round={round_index} id={candidate_id} "
        f"(expected {folder})"
    )


def _latest_checkpoint(train_dir: str) -> str:
    pts = sorted(glob.glob(os.path.join(train_dir, "checkpoints", "*.pt")))
    if not pts:
        raise FileNotFoundError(f"No .pt under {train_dir}/checkpoints")
    return pts[-1]


def run_eval(
    *,
    run_dir: str,
    candidate_id: str,
    round_index: int,
    episodes: int,
    device_pref: str,
    human_num: Optional[int],
    seed: Optional[int],
    output_json: Optional[str],
) -> Dict[str, Any]:
    import torch
    from rl.networks.envs import make_vec_envs
    from rl.networks.model import Policy

    from domains.crowdnav.regime import env_name_for_predict_method
    from domains.crowdnav.reporting import load_candidate_dict, write_json
    from raise_core.refine import evaluate_proxy_policy
    from raise_core.validate import (
        Stage3Config,
        _make_full_env_config,
        _parse_stage3_algo_args,
        _resolve_horizon,
    )

    run_dir = os.path.abspath(run_dir)
    run_cfg = (
        _load_json(os.path.join(run_dir, "config.json"))
        if os.path.isfile(os.path.join(run_dir, "config.json"))
        else {}
    )
    cand_payload, cand_src = _find_candidate_payload(run_dir, candidate_id)
    candidate = load_candidate_dict(cand_payload)
    if candidate.reward_fn is None:
        raise RuntimeError(
            f"Candidate reward failed sandbox: {candidate.validation_error}"
        )

    train_dir = _resolve_train_dir(run_dir, candidate_id, round_index)
    ckpt_path = _latest_checkpoint(train_dir)

    predict_method = str(run_cfg.get("predict_method") or "inferred")
    regime = str(run_cfg.get("randomization_regime") or "without_random")
    h_num = int(human_num if human_num is not None else run_cfg.get("human_num", 20))
    eval_seed = int(seed if seed is not None else run_cfg.get("seed", 425))
    env_name = env_name_for_predict_method(predict_method)

    s3_cfg = Stage3Config(
        train_env_steps=1,
        eval_episodes=int(episodes),
        seed=eval_seed,
        env_name=env_name,
        predict_method=predict_method,
        randomization_regime=regime,
        output_root=os.path.join(run_dir, "stage3_eval_only"),
        device=device_pref,
        train_human_num=h_num,
        num_processes=1,
    )
    algo_args = _parse_stage3_algo_args(s3_cfg, candidate.candidate_id, int(round_index))
    if device_pref != "cuda" or not torch.cuda.is_available():
        algo_args.cuda = False
    algo_args.seed = eval_seed

    env_config = _make_full_env_config(s3_cfg)
    env_config.env.test_size = int(episodes)
    horizon = _resolve_horizon(s3_cfg, env_config)

    device = torch.device(
        "cuda" if algo_args.cuda and torch.cuda.is_available() else "cpu"
    )

    # Build Policy with matching obs spaces, then load Stage III weights.
    probe = make_vec_envs(
        algo_args.env_name,
        eval_seed,
        1,
        algo_args.gamma,
        None,
        device,
        True,
        config=env_config,
        pretext_wrapper=env_config.env.use_wrapper,
        reward_fn=candidate.reward_fn,
    )
    try:
        actor_critic = Policy(
            probe.observation_space.spaces,
            probe.action_space,
            base_kwargs=algo_args,
            base=env_config.robot.policy,
        )
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        actor_critic.load_state_dict(state)
        actor_critic.to(device)
        actor_critic.base.nenv = 1
    finally:
        probe.close()

    print(
        f"[eval] candidate={candidate_id} round={round_index} "
        f"ckpt={ckpt_path}"
    )
    print(
        f"[eval] reward_src={cand_src} env={env_name} predict={predict_method} "
        f"H={h_num} episodes={episodes} seed={eval_seed} device={device}"
    )

    metrics = evaluate_proxy_policy(
        actor_critic,
        candidate.reward_fn,
        algo_args,
        env_config,
        device,
        n_episodes=int(episodes),
        horizon_steps=horizon,
        human_num=h_num,
    )
    md = metrics.as_dict()
    result = {
        "created_utc": _utc_now(),
        "run_dir": run_dir,
        "candidate_id": candidate_id,
        "round_index": int(round_index),
        "checkpoint_path": os.path.abspath(ckpt_path),
        "reward_json_source": cand_src,
        "eval_episodes": int(episodes),
        "seed": eval_seed,
        "human_num": h_num,
        "predict_method": predict_method,
        "randomization_regime": regime,
        "device": str(device),
        "metrics": md,
        "scalar_SR_minus_CR_minus_0.5TR": _scalar(md),
        "logged_full_metrics_txt": None,
    }
    logged = os.path.join(train_dir, "full_metrics.txt")
    if os.path.isfile(logged):
        with open(logged, encoding="utf-8") as f:
            result["logged_full_metrics_txt"] = f.read().strip()

    out_path = output_json or os.path.join(
        run_dir,
        "evals",
        f"eval_r{round_index:02d}_{candidate_id}_E{episodes}_s{eval_seed}.json",
    )
    write_json(out_path, result)
    print(f"[eval] {metrics.feedback_text()}")
    print(f"[eval] scalar={result['scalar_SR_minus_CR_minus_0.5TR']:.4f}")
    print(f"[eval] wrote {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval-only re-evaluation of an RAISE Stage III checkpoint"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="RAISE result directory (contains config.json, stage3_train/...)",
    )
    parser.add_argument(
        "--candidate-id",
        default=None,
        help="e.g. mut_0060_v2 (required unless --best-ever)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Stage III round index of the train folder (e.g. 0 for r00_*)",
    )
    parser.add_argument(
        "--best-ever",
        action="store_true",
        help="Pick candidate_id + round from best Stage III history scalar",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=150,
        help="Evaluation episodes (match Stage III E3; default 150)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Default: run config seed")
    parser.add_argument("--human-num", type=int, default=None, help="Default: run config")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--output-json",
        default=None,
        help="Where to write metrics JSON (default: <run-dir>/evals/...)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if args.best_ever:
        cid, rnd, hist_m = _best_ever_from_history(run_dir)
        print(
            f"[eval] --best-ever -> {cid} round={rnd} "
            f"logged_scalar={_scalar(hist_m):.4f} "
            f"(SR={hist_m.get('SR')}, CR={hist_m.get('CR')}, TR={hist_m.get('TR')})"
        )
    else:
        if not args.candidate_id or args.round is None:
            parser.error("Provide --candidate-id and --round, or use --best-ever")
        cid, rnd = str(args.candidate_id), int(args.round)

    run_eval(
        run_dir=run_dir,
        candidate_id=cid,
        round_index=rnd,
        episodes=int(args.episodes),
        device_pref=str(args.device),
        human_num=args.human_num,
        seed=args.seed,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()

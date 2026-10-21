"""Record CrowdNav rollouts of a trained SB3 checkpoint: GIF, frame strip, trajectory plot."""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PIL import Image
from stable_baselines3 import PPO

from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv, eval_cases
from raise_core.domains import get_pack, make_validator

logger = logging.getLogger(__name__)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_checkpoint(run_dir: str, cand: Dict[str, Any]) -> Optional[str]:
    md = cand.get("metadata") or {}
    ckpt = md.get("checkpoint_path")
    if ckpt:
        candidates = [
            str(ckpt),
            os.path.join(run_dir, str(ckpt)),
            os.path.normpath(os.path.join(run_dir, str(ckpt).replace("\\", os.sep))),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return os.path.abspath(c)
            # SB3 save may be path without .zip
            if os.path.isfile(c + ".zip"):
                return os.path.abspath(c + ".zip")
    cid = str(cand.get("candidate_id", ""))
    for stage in ("stage3_train", "stage2_train", "closed_loop/stage2_train"):
        pattern = os.path.join(run_dir, stage, f"*_{cid}", "model.zip")
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
        loose = sorted(glob.glob(os.path.join(run_dir, stage, f"*{cid}*", "model.zip")))
        if loose:
            return loose[-1]
    return None


def _pick_best_candidate(run_dir: str) -> Optional[Dict[str, Any]]:
    for name in ("best_stage3.json", "final_candidate.json", "best_stage2.json"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return _load_json(path)
    return None


def _frames_to_gif(frames: Sequence[np.ndarray], gif_path: str, *, fps: int = 4) -> None:
    if not frames:
        raise ValueError("No frames to encode")
    parent = os.path.dirname(gif_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    duration_ms = max(20, int(1000 / max(1, fps)))
    images = [Image.fromarray(np.asarray(f, dtype=np.uint8)).convert("RGB") for f in frames]
    try:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    finally:
        for im in images:
            im.close()


def record_policy_episode(
    *,
    reward_fn: Any,
    model: Any,
    case: int,
    split: str = "test",
    render: bool = True,
    deterministic: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Roll out one episode on upstream ``split`` case ``case``; return frames + summary."""
    env = RewardInjectedCrowdNavEnv(
        reward_fn, phase=split, render_mode="rgb_array" if render else None
    )
    obs, _info = env.reset(options={"case": int(case)})
    frames: List[np.ndarray] = []
    robot_path = [(float(env.sim.robot.px), float(env.sim.robot.py))]
    human_paths = [[(float(h.px), float(h.py))] for h in env.sim.humans]
    outcome = "timeout"
    done = False
    steps = 0
    min_sep = float("inf")
    while not done:
        if render:
            frames.append(env.render())
        action, _ = model.predict(obs, deterministic=bool(deterministic))
        obs, _r, terminated, truncated, info = env.step(action)
        steps += 1
        min_sep = min(min_sep, float(info["raise_min_separation"]))
        robot_path.append((float(env.sim.robot.px), float(env.sim.robot.py)))
        for path, h in zip(human_paths, env.sim.humans):
            path.append((float(h.px), float(h.py)))
        done = bool(terminated or truncated)
        if done:
            outcome = str(info.get("raise_outcome") or "timeout")
    if render:
        frames.append(env.render())
    summary = {
        "outcome": outcome,
        "n_frames": len(frames),
        "steps": steps,
        "nav_time_s": float(env.sim.global_time),
        "min_separation_m": float(min_sep),
        "split": str(split),
        "case": int(case),
        "goal": [float(env.sim.robot.gx), float(env.sim.robot.gy)],
        "robot_path": robot_path,
        "human_paths": human_paths,
        "human_radius": [float(h.radius) for h in env.sim.humans],
        "robot_radius": float(env.sim.robot.radius),
    }
    env.close()
    return frames, summary


def save_trajectory_plot(summary: Dict[str, Any], out_path: str) -> None:
    """Static robot / human paths (like upstream ``render(mode='traj')``)."""
    fig = Figure(figsize=(6, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cmap = matplotlib.colormaps["tab10"]
    for i, path in enumerate(summary["human_paths"]):
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=cmap(i % 10), lw=1.0)
        ax.add_artist(Circle(path[-1], summary["human_radius"][i], fill=False, color=cmap(i % 10)))
    xs, ys = zip(*summary["robot_path"])
    ax.plot(xs, ys, color="gold", lw=2.0, label="robot")
    ax.add_artist(Circle(summary["robot_path"][-1], summary["robot_radius"], color="gold"))
    gx, gy = summary["goal"]
    ax.plot([gx], [gy], marker="*", color="red", markersize=15, linestyle="None", label="goal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"{summary['split']} case {summary['case']}: {summary['outcome']} "
        f"({summary['nav_time_s']:.2f} s)",
        fontsize=10,
    )
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=110)


def save_frame_strip(frames: Sequence[np.ndarray], out_path: str, *, max_panels: int = 8) -> None:
    """Static montage of evenly spaced frames."""
    if not frames:
        return
    n = min(int(max_panels), len(frames))
    idxs = [int(round(i * (len(frames) - 1) / max(1, n - 1))) for i in range(n)]
    fig = Figure(figsize=(2.4 * n, 2.6))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, n, squeeze=False)[0]
    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i])
        ax.set_title(f"step {i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("CrowdNav episode frames", fontsize=10)
    fig.tight_layout()
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=120)


def _select_viz_cases(
    run_dir: str,
    *,
    seed: int,
    episodes: int,
    case_set_size: int,
) -> List[int]:
    """Prefer a mix that includes a non-success when final_eval recorded one."""
    default = eval_cases(int(seed), int(episodes), int(case_set_size))
    final_path = os.path.join(run_dir, "final_candidate.json")
    if not os.path.isfile(final_path) or episodes < 1:
        return default
    try:
        final = json.loads(Path(final_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
    md = (final.get("metadata") or {}).get("final_eval") or {}
    metrics = md.get("metrics") or {}
    eps = metrics.get("holdout_episodes") or []
    if not isinstance(eps, list) or not eps:
        return default
    fail = next(
        (
            int(e["case"])
            for e in eps
            if isinstance(e, dict)
            and e.get("case") is not None
            and str(e.get("outcome") or "") not in ("success", "")
        ),
        None,
    )
    if fail is None:
        return default
    cases = [fail] + [c for c in default if c != fail]
    return cases[: int(episodes)]


def visualize_best_policy(
    run_dir: str,
    *,
    output_dir: Optional[str] = None,
    episodes: int = 2,
    seed: int = 425,
    fps: int = 4,
    viz_mode: str = "stochastic",
) -> Dict[str, Any]:
    """
    Load the best Stage III (else II) SB3 zip + reward code; write GIF(s),
    frame strips and trajectory plots under ``plots/viz``.

    Episodes run on upstream ``test`` cases chosen from ``seed`` exactly as
    the final blind evaluation chooses them, so pass that evaluation's seed
    base to show scenarios the policy was not selected on.

    ``viz_mode`` is ``stochastic`` (default; fixed torch seed) or
    ``deterministic``. When final-eval episodes are available, at least one
    non-success case is preferred if any exist.
    """
    run_dir = os.path.abspath(run_dir)
    out_root = os.path.abspath(output_dir or os.path.join(run_dir, "plots", "viz"))
    os.makedirs(out_root, exist_ok=True)
    mode = str(viz_mode or "stochastic").strip().lower()
    deterministic = mode == "deterministic"

    cand = _pick_best_candidate(run_dir)
    if cand is None:
        return {"ok": False, "reason": "no best_stage*.json"}
    ckpt = _resolve_checkpoint(run_dir, cand)
    if not ckpt:
        return {
            "ok": False,
            "reason": "no model.zip",
            "candidate_id": cand.get("candidate_id"),
        }
    code = str(cand.get("code") or "")
    if not code.strip():
        return {"ok": False, "reason": "empty reward code"}

    pack = get_pack()
    validator = make_validator(pack)
    reward_fn, err = validator.try_validate(code)
    if reward_fn is None:
        return {"ok": False, "reason": f"reward invalid: {err}"}

    model = PPO.load(ckpt, device="cpu")
    if not deterministic:
        model.set_random_seed(int(seed))
    cid = str(cand.get("candidate_id", "best"))
    written: List[str] = []
    summaries: List[Dict[str, Any]] = []
    probe = RewardInjectedCrowdNavEnv(None, phase="test")
    case_set_size = int(probe.sim.case_size["test"])
    probe.close()
    cases = _select_viz_cases(
        run_dir, seed=int(seed), episodes=int(episodes), case_set_size=case_set_size
    )

    for ep, case in enumerate(cases):
        frames, summary = record_policy_episode(
            reward_fn=reward_fn,
            model=model,
            case=case,
            split="test",
            deterministic=deterministic,
        )
        traj = os.path.join(out_root, f"{cid}_ep{ep}_traj.png")
        save_trajectory_plot(summary, traj)
        written.append(traj)
        strip = os.path.join(out_root, f"{cid}_ep{ep}_strip.png")
        save_frame_strip(frames, strip)
        written.append(strip)
        gif_path = os.path.join(out_root, f"{cid}_ep{ep}.gif")
        try:
            _frames_to_gif(frames, gif_path, fps=fps)
            written.append(gif_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GIF encode failed for ep %s: %s", ep, exc)
        summaries.append(
            {k: v for k, v in summary.items() if k not in ("robot_path", "human_paths")}
        )

    meta = {
        "ok": True,
        "candidate_id": cid,
        "checkpoint": ckpt,
        "split": "test",
        "seed_base": int(seed),
        "viz_mode": mode,
        "deterministic": bool(deterministic),
        "cases": [int(c) for c in cases],
        "episodes": summaries,
        "outcomes": [s.get("outcome") for s in summaries],
        "written": written,
        "output_dir": out_root,
    }
    with open(os.path.join(out_root, "viz_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta

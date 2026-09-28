"""Record highway-env rgb frames + GIF for a trained SB3 checkpoint."""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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


def _frames_to_gif(frames: Sequence[np.ndarray], gif_path: str, *, fps: int = 10) -> None:
    if not frames:
        raise ValueError("No frames to encode")
    parent = os.path.dirname(gif_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    duration_ms = max(20, int(1000 / max(1, fps)))
    try:
        from PIL import Image

        images = [Image.fromarray(np.asarray(f, dtype=np.uint8)).convert("RGB") for f in frames]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        for im in images:
            im.close()
        return
    except ImportError:
        pass
    # Fallback: matplotlib PillowWriter / imageio not required for PNGs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def _update(i: int):
        im.set_data(frames[i])
        return (im,)

    anim = animation.FuncAnimation(fig, _update, frames=len(frames), interval=duration_ms, blit=True)
    try:
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
    finally:
        plt.close(fig)


def record_policy_episode(
    *,
    reward_fn: Any,
    model: Any,
    seed: int,
    max_steps: int = 400,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Roll out one episode; return rgb frames + summary."""
    from domains.highway.env_wrapper import RewardInjectedHighwayEnv, training_env_config

    env = RewardInjectedHighwayEnv(
        reward_fn,
        seed=int(seed),
        config=training_env_config(),
        render_mode="rgb_array",
    )
    obs, _info = env.reset(seed=int(seed))
    frames: List[np.ndarray] = []
    progress = 0.0
    speeds: List[float] = []
    outcome = "truncation"
    for _ in range(int(max_steps)):
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        progress += float(info.get("raise_progress", 0.0) or 0.0)
        speeds.append(float(info.get("raise_speed", 0.0) or 0.0))
        if info.get("raise_collision"):
            outcome = "collision"
            break
        if info.get("raise_off_road"):
            outcome = "off_road"
            break
        if terminated or truncated:
            outcome = "timeout" if info.get("raise_timeout") else "done"
            break
    # Final frame
    frame = env.render()
    if frame is not None:
        frames.append(np.asarray(frame, dtype=np.uint8))
    env.close()
    summary = {
        "outcome": outcome,
        "n_frames": len(frames),
        "progress_m": float(progress),
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "seed": int(seed),
    }
    return frames, summary


def save_frame_strip(frames: Sequence[np.ndarray], out_path: str, *, max_panels: int = 8) -> None:
    """Static montage of evenly spaced frames (always available without Pillow GIF)."""
    if not frames:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(int(max_panels), len(frames))
    idxs = [int(round(i * (len(frames) - 1) / max(1, n - 1))) for i in range(n)]
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, idxs):
        ax.imshow(frames[i])
        ax.set_title(f"t={i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Highway episode frames", fontsize=10)
    fig.tight_layout()
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def visualize_best_policy(
    run_dir: str,
    *,
    output_dir: Optional[str] = None,
    episodes: int = 2,
    seed: int = 425,
    fps: int = 10,
) -> Dict[str, Any]:
    """
    Load best Stage III (else II) SB3 zip + reward code; write GIF(s) under plots/viz.
    """
    run_dir = os.path.abspath(run_dir)
    out_root = os.path.abspath(output_dir or os.path.join(run_dir, "plots", "viz"))
    os.makedirs(out_root, exist_ok=True)

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

    from raise_core.domains import load_domain, make_validator_for_domain
    from stable_baselines3 import PPO

    pack = load_domain("highway")
    validator = make_validator_for_domain(pack)
    reward_fn, err = validator.try_validate(code)
    if reward_fn is None:
        return {"ok": False, "reason": f"reward invalid: {err}"}

    model = PPO.load(ckpt, device="cpu")
    cid = str(cand.get("candidate_id", "best"))
    written: List[str] = []
    summaries: List[Dict[str, Any]] = []

    for ep in range(int(episodes)):
        ep_seed = int(seed) + 10003 + ep * 97
        frames, summary = record_policy_episode(
            reward_fn=reward_fn, model=model, seed=ep_seed
        )
        summaries.append(summary)
        strip = os.path.join(out_root, f"{cid}_ep{ep}_strip.png")
        save_frame_strip(frames, strip)
        written.append(strip)
        if frames:
            gif_path = os.path.join(out_root, f"{cid}_ep{ep}.gif")
            try:
                _frames_to_gif(frames, gif_path, fps=fps)
                written.append(gif_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GIF encode failed for ep %s: %s", ep, exc)
                # Persist raw frames as PNGs so the run still has animation-ish artifacts.
                frame_dir = os.path.join(out_root, f"{cid}_ep{ep}_frames")
                os.makedirs(frame_dir, exist_ok=True)
                for i, fr in enumerate(frames[:: max(1, len(frames) // 40)]):
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(fr)
                    ax.axis("off")
                    p = os.path.join(frame_dir, f"{i:04d}.png")
                    fig.savefig(p, dpi=80, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                    written.append(p)

    meta = {
        "ok": True,
        "candidate_id": cid,
        "checkpoint": ckpt,
        "episodes": summaries,
        "written": written,
        "output_dir": out_root,
    }
    with open(os.path.join(out_root, "viz_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta

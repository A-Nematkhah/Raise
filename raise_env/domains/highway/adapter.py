"""Highway Stage II/III trainers (Stable-Baselines3 PPO) + stubs."""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, Dict, Optional

from raise_core.refine import ProxyMetrics, StubPolicyTrainer

logger = logging.getLogger(__name__)


class HighwayAdapter:
    """Domain train/eval surface for highway-fast-v0."""

    def __init__(
        self,
        *,
        stage2_delegate: Any = None,
        stage3_delegate: Any = None,
    ) -> None:
        self._stage2_delegate = stage2_delegate
        self._stage3_delegate = stage3_delegate

    def _stage2_trainer(self) -> Any:
        if self._stage2_delegate is not None:
            return self._stage2_delegate
        return HighwayPPOTrainer(stage="stage2")

    def _stage3_trainer(self) -> Any:
        if self._stage3_delegate is not None:
            return self._stage3_delegate
        return HighwayPPOTrainer(stage="stage3")

    def train_and_eval(
        self,
        candidate: Any,
        *,
        round_index: int,
        config: Any,
        stage: str = "stage2",
        **train_kwargs: Any,
    ) -> Any:
        stage_key = str(stage).strip().lower()
        if stage_key in ("2", "stage2", "ii"):
            return self._stage2_trainer().train_and_eval(
                candidate, round_index=round_index, config=config
            )
        if stage_key in ("3", "stage3", "iii"):
            return self._stage3_trainer().train_and_eval(
                candidate,
                round_index=round_index,
                config=config,
                **train_kwargs,
            )
        raise ValueError(f"Unknown stage for HighwayAdapter: {stage!r}")

    def evaluate_at_human_counts(
        self,
        candidate: Any,
        bundle: Any,
        *,
        config: Any,
        human_counts: Optional[Any] = None,
    ) -> Any:
        """No-op H-sweep for highway (CrowdNav-only concept)."""
        from raise_core.validate import HumanSweepReport

        return HumanSweepReport(candidate_id=str(candidate.candidate_id), by_human_count={})


class HighwayStage2Trainer:
    def __init__(self, adapter: HighwayAdapter) -> None:
        self.adapter = adapter

    def train_and_eval(self, candidate: Any, *, round_index: int, config: Any) -> Any:
        return self.adapter.train_and_eval(
            candidate, round_index=round_index, config=config, stage="stage2"
        )


class HighwayStage3Trainer:
    def __init__(self, adapter: HighwayAdapter) -> None:
        self.adapter = adapter

    def train_and_eval(
        self, candidate: Any, *, round_index: int, config: Any, **kwargs: Any
    ) -> Any:
        return self.adapter.train_and_eval(
            candidate,
            round_index=round_index,
            config=config,
            stage="stage3",
            **kwargs,
        )

    def evaluate_at_human_counts(
        self,
        candidate: Any,
        bundle: Any,
        *,
        config: Any,
        human_counts: Optional[Any] = None,
    ) -> Any:
        return self.adapter.evaluate_at_human_counts(
            candidate, bundle, config=config, human_counts=human_counts
        )


def _eval_metrics(env: Any, model: Any, *, n_episodes: int) -> ProxyMetrics:
    """
    Evaluate a highway policy.

    Survival metrics (SR/CR/TR) stay episode-level rates for surrogate labels.
    Continuous driving quality is packed into NT/PL/ITR/SD plus extra keys on
    the returned dict via ``metrics_to_highway_dict`` (called by the trainer):

    - PL  = mean forward progress (m)
    - ITR = mean ego speed (m/s)   [highway remap; CrowdNav uses intrusion %]
    - SD  = mean nearest-vehicle gap proxy (m)
    - NT  = mean episode duration (s)
    """
    import numpy as np

    sr = cr = tr = 0
    times: list[float] = []
    dists: list[float] = []
    speeds: list[float] = []
    gaps: list[float] = []
    soft_ok = 0
    lane_changes = 0
    total_steps = 0
    high_speed_steps = 0

    # Soft-success thresholds: survive AND move like a real driver.
    min_speed_for_soft = 15.0  # m/s
    min_progress_for_soft = 400.0  # m over a typical 40s episode

    for _ in range(max(1, int(n_episodes))):
        obs, _info = env.reset()
        done = False
        ep_t = 0.0
        ep_dist = 0.0
        ep_speed_sum = 0.0
        ep_steps = 0
        ep_lane_chg = 0
        ep_gap_sum = 0.0
        ep_high = 0
        crashed = False
        off = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_i = int(np.asarray(action).reshape(-1)[0])
            # DiscreteMetaAction: 0=LANE_LEFT, 2=LANE_RIGHT — count each such step.
            if action_i in (0, 2):
                ep_lane_chg += 1

            obs, reward, terminated, truncated, info = env.step(action)
            dt = float(getattr(env, "_time_step", 0.2))
            ep_t += dt
            prog = float(info.get("raise_progress", 0.0))
            spd = float(info.get("raise_speed", 0.0))
            ep_dist += prog
            ep_speed_sum += spd
            ep_steps += 1
            if spd >= min_speed_for_soft:
                ep_high += 1

            # Gap proxy: min |x| among others in observation (ego-relative stored in state).
            gap = _nearest_gap_from_obs(obs)
            if gap is not None:
                ep_gap_sum += gap

            if info.get("raise_collision"):
                crashed = True
            if info.get("raise_off_road"):
                off = True
            done = bool(terminated or truncated)

        if crashed:
            cr += 1
        elif off:
            tr += 1
        else:
            sr += 1

        mean_spd = ep_speed_sum / float(max(1, ep_steps))
        if (not crashed) and (not off) and mean_spd >= min_speed_for_soft and ep_dist >= min_progress_for_soft:
            soft_ok += 1

        times.append(ep_t)
        dists.append(ep_dist)
        speeds.append(mean_spd)
        gaps.append(ep_gap_sum / float(max(1, ep_steps)))
        lane_changes += ep_lane_chg
        total_steps += ep_steps
        high_speed_steps += ep_high

    n = float(max(1, int(n_episodes)))
    metrics = ProxyMetrics(
        sr=sr / n,
        cr=cr / n,
        tr=tr / n,
        nt=float(np.mean(times)) if times else 0.0,
        pl=float(np.mean(dists)) if dists else 0.0,
        # Highway: ITR slot = mean speed (m/s). Documented in pack spec.
        itr=float(np.mean(speeds)) if speeds else 0.0,
        sd=float(np.mean(gaps)) if gaps else 0.0,
    )
    # Stash continuous extras on the instance for the trainer to merge into dicts.
    metrics._highway_extras = {  # type: ignore[attr-defined]
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_progress": float(np.mean(dists)) if dists else 0.0,
        "soft_success": soft_ok / n,
        "lane_change_rate": float(lane_changes) / float(max(1, total_steps)),
        "high_speed_frac": float(high_speed_steps) / float(max(1, total_steps)),
        "speed_p10": float(np.percentile(speeds, 10)) if speeds else 0.0,
        "speed_p90": float(np.percentile(speeds, 90)) if speeds else 0.0,
    }
    return metrics


def _nearest_gap_from_obs(obs: Any) -> Optional[float]:
    import numpy as np

    arr = np.asarray(obs, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return None
    best = None
    for row in arr[1:]:
        if float(row[0]) <= 0.5:
            continue
        # Absolute kinematics: distance in xy to ego row0.
        dx = float(row[1]) - float(arr[0][1])
        dy = float(row[2]) - float(arr[0][2])
        dist = float((dx * dx + dy * dy) ** 0.5)
        if best is None or dist < best:
            best = dist
    return best


def metrics_to_highway_dict(metrics: ProxyMetrics) -> Dict[str, float]:
    """ProxyMetrics.as_dict() + continuous highway fields + selection_scalar."""
    from domains.highway.metrics import attach_selection_scalar

    payload = metrics.as_dict()
    extras = getattr(metrics, "_highway_extras", None) or {}
    payload.update({k: float(v) for k, v in extras.items()})
    # Alias ITR as mean_speed for readability in logs / feedback.
    payload["mean_speed"] = float(payload.get("mean_speed", payload.get("ITR", 0.0)))
    payload["mean_progress"] = float(payload.get("mean_progress", payload.get("PL", 0.0)))
    return attach_selection_scalar(payload)


def _ppo_progress_callback(total_timesteps: int, desc: str):
    """SB3 callback that drives ``raise_core.console.progress``."""
    from stable_baselines3.common.callbacks import BaseCallback

    from raise_core import console

    bar = console.progress(
        total=max(1, int(total_timesteps)),
        desc=desc,
        unit="step",
        leave=True,
        mininterval=0.4,
    )

    class _CB(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._last = 0

        def _on_step(self) -> bool:
            n = int(self.num_timesteps)
            delta = n - self._last
            if delta > 0:
                bar.update(delta)
                self._last = n
            return True

        def _on_training_end(self) -> None:
            rem = max(0, int(total_timesteps) - self._last)
            if rem:
                bar.update(rem)
                self._last = int(total_timesteps)
            try:
                bar.close()
            except Exception:  # noqa: BLE001
                pass

    return _CB()


class HighwayPPOTrainer:
    """Train SB3 PPO under a candidate reward; return ProxyMetrics or Stage3 bundle."""

    def __init__(self, *, stage: str = "stage2") -> None:
        self.stage = stage

    def train_and_eval(
        self,
        candidate: Any,
        *,
        round_index: int,
        config: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Highway PPO trainer needs stable-baselines3. "
                "Install: pip install -r requirements_highway.txt"
            ) from exc

        from domains.highway.env_wrapper import RewardInjectedHighwayEnv
        from raise_core import console
        from raise_core.validate import TrainEvalBundle

        if candidate.reward_fn is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no reward_fn")

        train_steps = int(getattr(config, "train_env_steps", 20_000))
        eval_episodes = int(getattr(config, "eval_episodes", 20))
        seed = int(getattr(config, "seed", 425)) + int(round_index)
        device_raw = str(getattr(config, "device", "cpu"))
        # MLP policies train better on CPU by default (SB3 guidance).
        prefer_cpu = os.environ.get("RAISE_HIGHWAY_FORCE_CUDA", "").strip() not in (
            "1",
            "true",
            "yes",
        )
        if prefer_cpu and device_raw.startswith("cuda"):
            device = "cpu"
        else:
            device = device_raw if device_raw.startswith("cuda") else "cpu"

        out_root = str(getattr(config, "output_root", "trained_models/highway"))
        cid = str(candidate.candidate_id)
        out_dir = os.path.join(out_root, f"r{int(round_index):02d}_{cid}")
        os.makedirs(out_dir, exist_ok=True)

        stage_tag = "Stage II" if self.stage in ("stage2", "2", "ii") else "Stage III"
        console.status(
            f"PPO train {cid} | steps={train_steps:,} eval={eval_episodes} "
            f"device={device} round={round_index}",
            stage=stage_tag,
        )

        env = RewardInjectedHighwayEnv(candidate.reward_fn, seed=seed)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are trying to run PPO on the GPU",
                category=UserWarning,
            )
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                device=device,
                n_steps=min(256, max(16, train_steps)),
                batch_size=min(64, max(8, train_steps)),
                learning_rate=3e-4,
            )

        t0 = time.perf_counter()
        cb = _ppo_progress_callback(train_steps, desc=f"PPO {cid} ({stage_tag})")
        model.learn(
            total_timesteps=max(1, train_steps),
            progress_bar=False,
            callback=cb,
        )
        train_wall = time.perf_counter() - t0
        sps = float(train_steps) / max(train_wall, 1e-6)
        console.status(
            f"PPO train done {cid} in {console.format_seconds(train_wall)} "
            f"({sps:.0f} steps/s)",
            stage=stage_tag,
        )

        ckpt = os.path.join(out_dir, "model.zip")
        model.save(ckpt)

        console.status(
            f"PPO eval {cid} | episodes={eval_episodes}",
            stage=stage_tag,
        )
        t1 = time.perf_counter()
        metrics = _eval_metrics(env, model, n_episodes=eval_episodes)
        eval_wall = time.perf_counter() - t1
        env.close()

        metrics_dict = metrics_to_highway_dict(metrics)
        from domains.highway.metrics import format_highway_metrics_line

        console.status(
            f"PPO metrics {cid} | {format_highway_metrics_line(metrics_dict)} | "
            f"eval {console.format_seconds(eval_wall)}",
            stage=stage_tag,
        )

        md = dict(candidate.metadata or {})
        md["last_metrics"] = metrics_dict
        md["selection_scalar"] = float(metrics_dict["selection_scalar"])
        md["checkpoint_path"] = ckpt
        md["train_wall_seconds"] = float(train_wall)
        md["eval_wall_seconds"] = float(eval_wall)
        candidate.metadata = md

        if self.stage in ("stage3", "3", "iii"):
            return TrainEvalBundle(
                metrics=metrics,
                checkpoint_path=ckpt,
            )
        return metrics


def make_stage2_trainer(*, use_stub: bool = False) -> HighwayStage2Trainer:
    if use_stub:
        adapter = HighwayAdapter(stage2_delegate=StubPolicyTrainer())
    else:
        adapter = HighwayAdapter()
    return HighwayStage2Trainer(adapter)


def make_stage3_trainer(*, use_stub: bool = False) -> HighwayStage3Trainer:
    if use_stub:
        from raise_core.validate import StubPolicyTrainer as S3Stub

        adapter = HighwayAdapter(stage3_delegate=S3Stub())
    else:
        adapter = HighwayAdapter()
    return HighwayStage3Trainer(adapter)

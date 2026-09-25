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


def _eval_metrics(
    env: Any,
    model: Any,
    *,
    n_episodes: int,
    seed: int = 0,
    seed_offset: int = 10_003,
    soft_speed_mps: float = 20.0,
    soft_progress_m: float = 400.0,
) -> ProxyMetrics:
    """
    Evaluate a highway policy.

    Survival metrics (SR/CR/TR) stay episode-level rates for surrogate labels.
    Continuous driving quality is packed into NT/PL/ITR/SD plus extra keys on
    the returned dict via ``metrics_to_highway_dict`` (called by the trainer):

    - PL  = mean forward progress (m)
    - ITR = mean ego speed (m/s)   [highway remap; CrowdNav uses intrusion %]
    - SD  = mean nearest-vehicle gap proxy (m)
    - NT  = mean episode duration (s)

    Each episode uses a distinct ``reset(seed=…)`` so SR/CR are real rates
    over varied traffic, not 20 copies of one trajectory.
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
    outcomes: list[str] = []

    # Soft-success: survive AND traffic-speed cruise (~≥20 m/s) + progress.
    min_speed_for_soft = float(soft_speed_mps)
    min_progress_for_soft = float(soft_progress_m)
    n_eps = max(1, int(n_episodes))
    base = int(seed)
    offset = int(seed_offset)

    for ep in range(n_eps):
        # Offset well above train seeds so eval scenarios differ from train.
        obs, _info = env.reset(seed=base + offset + ep * 97)
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
            outcomes.append("collision")
        elif off:
            tr += 1
            outcomes.append("off_road")
        else:
            sr += 1
            outcomes.append("success")

        mean_spd = ep_speed_sum / float(max(1, ep_steps))
        if (
            (not crashed)
            and (not off)
            and mean_spd >= min_speed_for_soft
            and ep_dist >= min_progress_for_soft
        ):
            soft_ok += 1

        times.append(ep_t)
        dists.append(ep_dist)
        speeds.append(mean_spd)
        gaps.append(ep_gap_sum / float(max(1, ep_steps)))
        lane_changes += ep_lane_chg
        total_steps += ep_steps
        high_speed_steps += ep_high

    n = float(n_eps)
    metrics = ProxyMetrics(
        sr=sr / n,
        cr=cr / n,
        tr=tr / n,
        nt=float(np.mean(times)) if times else 0.0,
        pl=float(np.mean(dists)) if dists else 0.0,
        itr=float(np.mean(speeds)) if speeds else 0.0,
        sd=float(np.mean(gaps)) if gaps else 0.0,
    )
    metrics._highway_extras = {  # type: ignore[attr-defined]
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_progress": float(np.mean(dists)) if dists else 0.0,
        "soft_success": soft_ok / n,
        "lane_change_rate": float(lane_changes) / float(max(1, total_steps)),
        "high_speed_frac": float(high_speed_steps) / float(max(1, total_steps)),
        "speed_p10": float(np.percentile(speeds, 10)) if speeds else 0.0,
        "speed_p90": float(np.percentile(speeds, 90)) if speeds else 0.0,
        "progress_std": float(np.std(dists)) if len(dists) > 1 else 0.0,
        "n_success": int(sr),
        "n_collision": int(cr),
        "n_off_road": int(tr),
        "n_eval_episodes": int(n_eps),
        "outcome_unique": int(len(set(outcomes))),
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
    """ProxyMetrics.as_dict() + continuous highway fields + official fitness."""
    from domains.highway.metrics import attach_fitness

    payload = metrics.as_dict()
    extras = getattr(metrics, "_highway_extras", None) or {}
    payload.update({k: float(v) for k, v in extras.items()})
    # Alias ITR as mean_speed for readability in logs / feedback.
    payload["mean_speed"] = float(payload.get("mean_speed", payload.get("ITR", 0.0)))
    payload["mean_progress"] = float(payload.get("mean_progress", payload.get("PL", 0.0)))
    return attach_fitness(payload)


def _highway_n_envs(config: Any) -> int:
    raw = getattr(config, "highway_n_envs", None)
    if raw is None:
        raw = os.environ.get("RAISE_HIGHWAY_N_ENVS", "1")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _highway_eval_mode(config: Any) -> str:
    mode = str(getattr(config, "highway_eval_mode", None) or "both").strip().lower()
    if mode not in ("both", "holdout_only"):
        return "both"
    return mode


def _highway_warm_start_enabled(config: Any) -> bool:
    if hasattr(config, "highway_warm_start"):
        return bool(getattr(config, "highway_warm_start"))
    return os.environ.get("RAISE_HIGHWAY_WARM_START", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _make_train_vec_env(reward_fn: Any, *, seed: int, n_envs: int, env_config: Optional[Dict] = None):
    """DummyVecEnv of RewardInjectedHighwayEnv (Windows-safe; same-process)."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.highway.env_wrapper import RewardInjectedHighwayEnv

    n = max(1, int(n_envs))

    def _thunk(rank: int):
        def _init():
            return RewardInjectedHighwayEnv(
                reward_fn,
                seed=int(seed) + int(rank) * 997,
                config=env_config,
            )

        return _init

    return DummyVecEnv([_thunk(i) for i in range(n)])


def _resolve_warm_start_path(candidate: Any) -> Optional[str]:
    md = getattr(candidate, "metadata", None) or {}
    for key in ("warm_start_checkpoint", "parent_checkpoint_path"):
        path = md.get(key)
        if path and os.path.isfile(str(path)):
            return str(path)
    return None


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
        n_envs = _highway_n_envs(config)
        eval_mode = _highway_eval_mode(config)
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
        warm_path = (
            _resolve_warm_start_path(candidate)
            if _highway_warm_start_enabled(config)
            else None
        )
        warm_bit = f" warm={os.path.basename(warm_path)}" if warm_path else ""
        console.status(
            f"PPO train {cid} | steps={train_steps:,} n_envs={n_envs} "
            f"eval={eval_episodes}({eval_mode}) device={device} "
            f"round={round_index}{warm_bit}",
            stage=stage_tag,
        )

        train_env = _make_train_vec_env(
            candidate.reward_fn, seed=seed, n_envs=n_envs
        )
        # Per-env rollout length; keep ~256 total steps/env as before when n_envs=1.
        n_steps = min(256, max(16, train_steps // max(1, n_envs)))
        batch_size = min(64 * n_envs, max(8, n_steps * n_envs))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are trying to run PPO on the GPU",
                category=UserWarning,
            )
            # Always build a fresh PPO so n_steps/batch_size match this VecEnv.
            # PPO.load + mutating n_steps leaves a stale RolloutBuffer (IndexError).
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=0,
                seed=seed,
                device=device,
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=3e-4,
            )
            if warm_path:
                try:
                    donor = PPO.load(warm_path, device=device)
                    model.policy.load_state_dict(donor.policy.state_dict())
                    del donor
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Warm-start load failed for %s (%s); training from scratch",
                        cid,
                        exc,
                    )
                    warm_path = None

        t0 = time.perf_counter()
        cb = _ppo_progress_callback(train_steps, desc=f"PPO {cid} ({stage_tag})")
        model.learn(
            total_timesteps=max(1, train_steps),
            progress_bar=False,
            callback=cb,
            reset_num_timesteps=warm_path is None,
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
        try:
            train_env.close()
        except Exception:  # noqa: BLE001
            pass

        console.status(
            f"PPO eval {cid} | episodes={eval_episodes} ({eval_mode})",
            stage=stage_tag,
        )
        t1 = time.perf_counter()
        from domains.highway.env_wrapper import (
            HOLDOUT_SEED_OFFSET,
            holdout_env_config,
        )

        train_dict: Dict[str, float] = {}
        if eval_mode == "both":
            eval_env = RewardInjectedHighwayEnv(candidate.reward_fn, seed=seed)
            try:
                train_metrics = _eval_metrics(
                    eval_env,
                    model,
                    n_episodes=eval_episodes,
                    seed=seed,
                    seed_offset=10_003,
                )
                train_dict = metrics_to_highway_dict(train_metrics)
            finally:
                eval_env.close()

        holdout_env = RewardInjectedHighwayEnv(
            candidate.reward_fn,
            seed=seed + 7,
            config=holdout_env_config(),
        )
        try:
            holdout_metrics = _eval_metrics(
                holdout_env,
                model,
                n_episodes=eval_episodes,
                seed=seed,
                seed_offset=HOLDOUT_SEED_OFFSET,
            )
        finally:
            holdout_env.close()
        eval_wall = time.perf_counter() - t1

        holdout_dict = metrics_to_highway_dict(holdout_metrics)
        metrics_dict = dict(holdout_dict)
        metrics_dict["eval_profile"] = "holdout"
        if train_dict:
            metrics_dict["train_dist"] = {
                k: v for k, v in train_dict.items() if k not in ("holdout", "train_dist")
            }
        metrics_dict["holdout"] = {
            k: v
            for k, v in holdout_dict.items()
            if k not in ("holdout", "train_dist")
        }
        metrics_dict["n_envs"] = float(n_envs)
        metrics_dict["warm_start"] = 1.0 if warm_path else 0.0
        from domains.highway.metrics import attach_fitness, format_highway_metrics_line

        metrics_dict = attach_fitness(metrics_dict)

        console.status(
            f"PPO metrics {cid} | {format_highway_metrics_line(metrics_dict)} | "
            f"eval {console.format_seconds(eval_wall)}",
            stage=stage_tag,
        )

        md = dict(candidate.metadata or {})
        md["last_metrics"] = metrics_dict
        fit = float(metrics_dict["fitness"])
        md["fitness"] = fit
        md["selection_scalar"] = fit  # alias
        # Invalidate stale Pareto stamps from a prior epoch ranking pass.
        for key in (
            "pareto_rank",
            "pareto_n",
            "pareto_score",
            "pareto_feasible",
            "pareto_v_floor",
            "pareto_cr_ceiling",
            "pareto_tr_ceiling",
        ):
            md.pop(key, None)
        md["checkpoint_path"] = ckpt
        md["train_wall_seconds"] = float(train_wall)
        md["eval_wall_seconds"] = float(eval_wall)
        candidate.metadata = md

        if self.stage in ("stage3", "3", "iii"):
            return TrainEvalBundle(
                metrics=holdout_metrics,
                checkpoint_path=ckpt,
            )
        return holdout_metrics


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

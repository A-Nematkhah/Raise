"""Highway Stage II/III trainers (Stable-Baselines3 PPO) + stubs."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from crowd_nav.reward_search.refine import ProxyMetrics, StubPolicyTrainer

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
        from crowd_nav.reward_search.validate import HumanSweepReport

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
    import numpy as np

    sr = cr = tr = 0
    times = []
    dists = []
    min_gaps = []
    for _ in range(max(1, int(n_episodes))):
        obs, _info = env.reset()
        done = False
        ep_t = 0.0
        ep_dist = 0.0
        crashed = False
        off = False
        timed = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_t += float(getattr(env, "_time_step", 0.2))
            ep_dist += float(info.get("raise_progress", 0.0))
            if info.get("raise_collision"):
                crashed = True
            if info.get("raise_off_road"):
                off = True
            if info.get("raise_timeout"):
                timed = True
            # crude clearance: min |y| of others not available; use speed proxy
            min_gaps.append(float(info.get("raise_speed", 0.0)))
            done = bool(terminated or truncated)
        if crashed:
            cr += 1
        elif off:
            tr += 1
        else:
            # Survived without crash/off-road (incl. natural timeout horizon).
            sr += 1
        times.append(ep_t)
        dists.append(ep_dist)

    n = float(max(1, int(n_episodes)))
    return ProxyMetrics(
        sr=sr / n,
        cr=cr / n,
        tr=tr / n,
        nt=float(np.mean(times)) if times else 0.0,
        pl=float(np.mean(dists)) if dists else 0.0,
        itr=0.0,
        sd=float(np.mean(min_gaps)) if min_gaps else 0.0,
    )


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

        from crowd_nav.domains.highway.env_wrapper import RewardInjectedHighwayEnv
        from crowd_nav.reward_search.validate import TrainEvalBundle

        if candidate.reward_fn is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no reward_fn")

        train_steps = int(getattr(config, "train_env_steps", 20_000))
        eval_episodes = int(getattr(config, "eval_episodes", 20))
        seed = int(getattr(config, "seed", 425)) + int(round_index)
        device = str(getattr(config, "device", "cpu"))
        out_root = str(getattr(config, "output_root", "trained_models/highway"))
        out_dir = os.path.join(
            out_root, f"r{int(round_index):02d}_{candidate.candidate_id}"
        )
        os.makedirs(out_dir, exist_ok=True)

        env = RewardInjectedHighwayEnv(candidate.reward_fn, seed=seed)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device if device.startswith("cuda") else "cpu",
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
        )
        logger.info(
            "Highway PPO train candidate=%s steps=%d device=%s",
            candidate.candidate_id,
            train_steps,
            device,
        )
        model.learn(total_timesteps=max(1, train_steps), progress_bar=False)
        ckpt = os.path.join(out_dir, "model.zip")
        model.save(ckpt)

        metrics = _eval_metrics(env, model, n_episodes=eval_episodes)
        env.close()

        # Attach metrics on candidate for selection helpers.
        md = dict(candidate.metadata or {})
        md["last_metrics"] = metrics.as_dict()
        md["checkpoint_path"] = ckpt
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
        from crowd_nav.reward_search.validate import StubPolicyTrainer as S3Stub

        adapter = HighwayAdapter(stage3_delegate=S3Stub())
    else:
        adapter = HighwayAdapter()
    return HighwayStage3Trainer(adapter)

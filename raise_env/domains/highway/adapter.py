"""Highway Stage II/III trainers (Stable-Baselines3 PPO) + stubs."""

from __future__ import annotations

import copy
import logging
import os
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np

from domains.highway.action_config import (
    configure_action_mode_from_config,
    spaces_compatible,
)
from domains.highway.metrics import (
    attach_fitness,
    bootstrap_fitness_se,
    episode_fitness_fields,
    format_highway_metrics_line,
)
from domains.highway.objective_constants import PPO_GAMMA, SOFT_PROGRESS_M, V_TARGET
from domains.highway.overtake import OvertakeTracker
from domains.highway.training_dynamics import TrainingDynamicsRecorder
from domains.highway.watchdog import StallWatchdog, stall_seconds_from_config
from raise_core import console
from raise_core.refine import ProxyMetrics, StubPolicyTrainer
from raise_core.sandbox.runtime import MAX_REWARD_COMPONENT_KEYS
from raise_core.validate import StubPolicyTrainer as Stage3StubTrainer
from raise_core.validate import TrainEvalBundle

logger = logging.getLogger(__name__)

# Process-wide (thread-safe enough): do NOT use warnings.catch_warnings() inside
# parallel Stage II label workers — that context manager mutates global filters.
warnings.filterwarnings(
    "ignore",
    message="You are trying to run PPO on the GPU",
    category=UserWarning,
)


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


_cuda_fallback_logged = False


def _log_cuda_fallback_once(requested: str) -> None:
    global _cuda_fallback_logged
    if _cuda_fallback_logged:
        return
    _cuda_fallback_logged = True
    logger.info(
        "Highway PPO: device=%s requested, training on CPU (MLP policy is faster "
        "there); set RAISE_HIGHWAY_FORCE_CUDA=1 to use the GPU",
        requested,
    )


def _eval_metrics(
    env: Any,
    model: Any,
    *,
    n_episodes: int,
    seed: int = 0,
    seed_offset: int = 10_003,
    soft_speed_mps: float | None = None,
    soft_progress_m: float = SOFT_PROGRESS_M,
    deterministic: bool = True,
    record_components: bool = False,
) -> ProxyMetrics:
    """
    Evaluate a highway policy.

    Also keeps one compact record per episode on ``metrics._highway_episodes``
    (outcome, steps, speed, progress, lane changes, overtakes, undiscounted and
    PPO-discounted return under the candidate reward; with
    ``record_components`` also per-episode sums of the reward components).

    Survival metrics (SR/CR/TR) are episode-level rates.
    Continuous driving quality is packed into NT/PL/ITR/SD plus extra keys on
    the returned dict via ``metrics_to_highway_dict`` (called by the trainer):

    - PL  = mean forward progress (m)
    - ITR = mean ego speed (m/s)
    - SD  = mean nearest-vehicle gap proxy (m)
    - NT  = mean episode duration (s)

    Each episode uses a distinct ``reset(seed=…)`` so SR/CR are real rates
    over varied traffic, not 20 copies of one trajectory.
    """
    sr = cr = tr = 0
    times: list[float] = []
    dists: list[float] = []
    speeds: list[float] = []
    gaps: list[float] = []
    soft_ok = 0
    ep_lane_changes: list[int] = []
    ep_step_counts: list[int] = []
    outcomes: list[str] = []
    ep_overtakes: list[int] = []
    episode_records: list[Dict[str, Any]] = []

    # Soft-success: survive AND mean_speed ≥ V_TARGET AND progress ≥ threshold.
    #
    # Human-facing diagnostic only: NOT a selection / LLM objective. Threshold
    # stays FIXED at V_TARGET so the logged indicator is a stable aspirational
    # bar for humans reading reports.
    min_speed_for_soft = float(
        V_TARGET if soft_speed_mps is None else soft_speed_mps
    )
    min_progress_for_soft = float(soft_progress_m)
    n_eps = max(1, int(n_episodes))
    base = int(seed)
    offset = int(seed_offset)

    for ep in range(n_eps):
        # Offset well above train seeds so eval scenarios differ from train.
        obs, reset_info = env.reset(seed=base + offset + ep * 97)
        prev_lane = reset_info.get("raise_lane_index")
        done = False
        ep_t = 0.0
        ep_dist = 0.0
        ep_speed_sum = 0.0
        ep_steps = 0
        ep_lane_chg = 0
        ep_gap_sum = 0.0
        crashed = False
        off = False
        ep_return = 0.0
        ep_disc_return = 0.0
        discount = 1.0
        ep_components: Dict[str, float] = {}
        overtake_tracker = OvertakeTracker()
        overtake_tracker.observe(env)
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_disc_return += discount * float(reward)
            discount *= PPO_GAMMA
            if record_components:
                for key, value in (info.get("raise_reward_components") or {}).items():
                    if key not in ep_components and len(ep_components) >= MAX_REWARD_COMPONENT_KEYS:
                        key = "_other"
                    ep_components[key] = ep_components.get(key, 0.0) + float(value)
            # Lane-change from lane index Δ (works for Discrete + Continuous).
            # Continuous lateral=False never changes lane → rate stays 0.
            cur_lane = info.get("raise_lane_index")
            if prev_lane is not None and cur_lane is not None and prev_lane != cur_lane:
                ep_lane_chg += 1
            prev_lane = cur_lane

            dt = float(getattr(env, "_time_step", 0.2))
            ep_t += dt
            prog = float(info.get("raise_progress", 0.0))
            spd = float(info.get("raise_speed", 0.0))
            ep_dist += prog
            ep_speed_sum += spd
            ep_steps += 1

            gap = info.get("raise_nearest_gap")
            if gap is not None:
                ep_gap_sum += gap

            if info.get("raise_collision"):
                crashed = True
            if info.get("raise_off_road"):
                off = True
            if not (crashed or off):
                overtake_tracker.observe(env)
            done = bool(terminated or truncated)

        ep_overtakes.append(overtake_tracker.passes)
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
        ep_lane_changes.append(ep_lane_chg)
        ep_step_counts.append(ep_steps)
        record: Dict[str, Any] = {
            "outcome": outcomes[-1],
            "steps": int(ep_steps),
            "mean_speed": round(mean_spd, 3),
            "progress": round(ep_dist, 3),
            "lane_changes": int(ep_lane_chg),
            "overtakes": int(overtake_tracker.passes),
            "return": round(ep_return, 4),
            "disc_return": round(ep_disc_return, 4),
        }
        if record_components:
            record["components"] = {k: round(v, 4) for k, v in ep_components.items()}
        episode_records.append(record)

    n = float(n_eps)
    total_km = float(sum(dists)) / 1000.0
    total_overtakes = float(sum(ep_overtakes))
    episodes = (outcomes, dists, speeds, ep_lane_changes, ep_step_counts)
    fields = episode_fitness_fields(*episodes)
    metrics = ProxyMetrics(
        sr=fields["SR"],
        cr=fields["CR"],
        tr=fields["TR"],
        nt=float(np.mean(times)) if times else 0.0,
        pl=fields["PL"],
        itr=fields["mean_speed"],
        sd=float(np.mean(gaps)) if gaps else 0.0,
    )
    metrics._highway_extras = {  # type: ignore[attr-defined]
        "mean_speed": fields["mean_speed"],
        "mean_progress": fields["PL"],
        "soft_success": soft_ok / n,
        "lane_change_rate": fields["lane_change_rate"],
        "overtakes_per_km": total_overtakes / total_km if total_km > 0 else 0.0,
        "overtake_episode_frac": (
            float(sum(1 for k in ep_overtakes if k > 0)) / n
        ),
        "speed_p10": fields["speed_p10"],
        "speed_p90": fields["speed_p90"],
        "progress_std": fields["progress_std"],
        "n_success": int(sr),
        "n_collision": int(cr),
        "n_off_road": int(tr),
        "n_eval_episodes": int(n_eps),
        "outcome_unique": int(fields["outcome_unique"]),
        "fitness_se": bootstrap_fitness_se(*episodes, seed=base + offset),
    }
    metrics._highway_episodes = episode_records  # type: ignore[attr-defined]
    return metrics


def metrics_to_highway_dict(metrics: ProxyMetrics) -> Dict[str, float]:
    """ProxyMetrics.as_dict() + continuous highway fields + official fitness."""
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
    raw = getattr(config, "highway_warm_start", None)
    if raw is not None:
        return bool(raw)
    return os.environ.get("RAISE_HIGHWAY_WARM_START", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def clone_reward_fn(reward_fn: Any) -> Any:
    """Independent copy of ``reward_fn`` with its own per-episode ``memory``.

    Envs that step concurrently must not share one reward object: any env's
    ``reset()`` would clear the others' memory mid-episode.
    """
    try:
        return copy.deepcopy(reward_fn)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "reward_fn %s is not copyable (%s); envs will share episode memory",
            type(reward_fn).__name__,
            exc,
        )
        return reward_fn


def _make_train_vec_env(reward_fn: Any, *, seed: int, n_envs: int, env_config: Optional[Dict] = None):
    """DummyVecEnv of Monitor(RewardInjectedHighwayEnv) (Windows-safe; same-process).

    ``Monitor`` (no filename → no CSV) injects ``info["episode"]`` on episode
    end so SB3 can log ``rollout/ep_rew_mean`` / ``ep_len_mean``. Eval uses a
    separate unwrapped env in ``_eval_metrics`` and does not need Monitor.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from domains.highway.env_wrapper import RewardInjectedHighwayEnv

    n = max(1, int(n_envs))

    def _thunk(rank: int):
        def _init():
            # PPO(seed=s) re-seeds VecEnv slot i with s + i on the first reset.
            env = RewardInjectedHighwayEnv(
                reward_fn if rank == 0 else clone_reward_fn(reward_fn),
                seed=int(seed) + int(rank),
                config=env_config,
            )
            # filename omitted on purpose — avoid per-worker Monitor CSVs.
            return Monitor(env)

        return _init

    return DummyVecEnv([_thunk(i) for i in range(n)])


def _resolve_warm_start_path(candidate: Any) -> Optional[str]:
    md = getattr(candidate, "metadata", None) or {}
    for key in ("warm_start_checkpoint", "parent_checkpoint_path"):
        path = md.get(key)
        if path and os.path.isfile(str(path)):
            return str(path)
    return None


def _ppo_progress_callback(
    total_timesteps: int,
    desc: str,
    dynamics: Optional[TrainingDynamicsRecorder] = None,
):
    """SB3 callback that drives ``raise_core.console.progress`` (and ``dynamics``)."""
    from stable_baselines3.common.callbacks import BaseCallback

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
            if dynamics is not None:
                dynamics.observe(self.locals.get("infos") or (), self.locals.get("dones"), n)
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
        with StallWatchdog(
            stall_seconds_from_config(config),
            label=str(getattr(candidate, "candidate_id", "")),
        ):
            return self._train_and_eval(
                candidate, round_index=round_index, config=config, **kwargs
            )

    def _train_and_eval(
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
                "Install: pip install -r requirements.txt"
            ) from exc

        from domains.highway.env_wrapper import RewardInjectedHighwayEnv

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
            _log_cuda_fallback_once(device_raw)
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

        from domains.highway.env_wrapper import training_env_config

        configure_action_mode_from_config(config)
        env_cfg = training_env_config()

        train_env = _make_train_vec_env(
            candidate.reward_fn, seed=seed, n_envs=n_envs, env_config=env_cfg
        )
        # Per-env rollout length; keep ~256 total steps/env as before when n_envs=1.
        n_steps_cap = max(16, int(getattr(config, "highway_n_steps", 256) or 256))
        n_steps = min(n_steps_cap, max(16, train_steps // max(1, n_envs)))
        batch_size = min(64 * n_envs, max(8, n_steps * n_envs))
        n_epochs = max(1, int(getattr(config, "highway_n_epochs", 10) or 10))
        # Always build a fresh PPO so n_steps/batch_size match this VecEnv.
        # PPO.load + mutating n_steps leaves a stale RolloutBuffer (IndexError).
        # Warm-start copies policy weights only (new model → _last_obs is None,
        # so SB3 resets the env regardless of reset_num_timesteps).
        # MlpPolicy: Discrete → Categorical; Box → Gaussian (read from env).
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=seed,
            device=device,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=PPO_GAMMA,
            learning_rate=3e-4,
            ent_coef=float(getattr(config, "highway_ent_coef", 0.0) or 0.0),
        )
        if warm_path:
            try:
                donor = PPO.load(warm_path, device=device)
                if not spaces_compatible(donor.action_space, model.action_space):
                    logger.warning(
                        "Warm-start skipped for %s: parent action_space %s "
                        "!= current %s (action_mode mismatch)",
                        cid,
                        donor.action_space,
                        model.action_space,
                    )
                    warm_path = None
                else:
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
        dynamics = (
            TrainingDynamicsRecorder(train_steps)
            if bool(getattr(config, "highway_training_dynamics", False))
            else None
        )
        cb = _ppo_progress_callback(
            train_steps, desc=f"PPO {cid} ({stage_tag})", dynamics=dynamics
        )
        model.learn(
            total_timesteps=max(1, train_steps),
            progress_bar=False,
            callback=cb,
            # reset_num_timesteps omitted on purpose: this PPO is always freshly
            # constructed (_last_obs is None), so SB3 resets the env regardless
            # of that flag — passing it would be misleading documentation.
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

        eval_deterministic = bool(getattr(config, "highway_eval_deterministic", False))
        eval_seed_raw = getattr(config, "highway_eval_seed_base", None)
        eval_seed = seed if eval_seed_raw is None else int(eval_seed_raw)
        train_dict: Dict[str, float] = {}
        if eval_mode == "both":
            eval_env = RewardInjectedHighwayEnv(
                candidate.reward_fn, seed=eval_seed, config=env_cfg
            )
            if not eval_deterministic:
                model.set_random_seed(int(eval_seed) + 10_003)
            try:
                train_metrics = _eval_metrics(
                    eval_env,
                    model,
                    n_episodes=eval_episodes,
                    seed=eval_seed,
                    seed_offset=10_003,
                    deterministic=eval_deterministic,
                )
                train_dict = metrics_to_highway_dict(train_metrics)
            finally:
                eval_env.close()

        holdout_cfg = holdout_env_config()
        holdout_env = RewardInjectedHighwayEnv(
            candidate.reward_fn,
            seed=eval_seed + 7,
            config=holdout_cfg,
        )
        if not eval_deterministic:
            # Action sampling must not depend on RNG consumed during training.
            model.set_random_seed(int(eval_seed) + int(HOLDOUT_SEED_OFFSET))
        try:
            holdout_metrics = _eval_metrics(
                holdout_env,
                model,
                n_episodes=eval_episodes,
                seed=eval_seed,
                seed_offset=HOLDOUT_SEED_OFFSET,
                deterministic=eval_deterministic,
                record_components=bool(getattr(config, "highway_reward_components", False)),
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
        metrics_dict["holdout_episodes"] = list(getattr(holdout_metrics, "_highway_episodes", []))
        metrics_dict["eval_context"] = {
            "traffic": "holdout",
            "vehicles_count": holdout_cfg.get("vehicles_count"),
            "vehicles_density": holdout_cfg.get("vehicles_density", 1.0),
            "duration_s": holdout_cfg.get("duration"),
            "policy_frequency": holdout_cfg.get("policy_frequency"),
            "deterministic": bool(eval_deterministic),
        }
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
        md["checkpoint_path"] = ckpt
        md["train_steps"] = int(train_steps)
        md["train_seed"] = int(seed)
        md["eval_seed_base"] = int(eval_seed) + int(HOLDOUT_SEED_OFFSET)
        md["eval_episodes"] = int(eval_episodes)
        md["ppo_n_steps"] = int(n_steps)
        md["eval_deterministic"] = bool(eval_deterministic)
        md["train_wall_seconds"] = float(train_wall)
        md["eval_wall_seconds"] = float(eval_wall)
        if dynamics is not None:
            md["training_dynamics"] = dynamics.summary()
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
        adapter = HighwayAdapter(stage3_delegate=Stage3StubTrainer())
    else:
        adapter = HighwayAdapter()
    return HighwayStage3Trainer(adapter)

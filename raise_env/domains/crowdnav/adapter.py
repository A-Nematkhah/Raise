"""CrowdNav Stage II/III trainers (Stable-Baselines3 PPO) + stubs."""

from __future__ import annotations

import copy
import logging
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from domains.crowdnav.env_wrapper import RewardInjectedCrowdNavEnv, eval_cases
from domains.crowdnav.features import CrowdNavFeaturesExtractor
from domains.crowdnav.metrics import (
    attach_fitness,
    episode_task_return,
    episode_fields,
    episode_selection_score,
    fitness_standard_error,
    format_crowdnav_metrics_line,
)
from domains.crowdnav.objective_constants import PPO_GAMMA
from domains.crowdnav.settings import settings_from_config, set_settings, spaces_compatible
from domains.crowdnav.training_dynamics import TrainingDynamicsRecorder
from domains.crowdnav.watchdog import StallWatchdog, stall_seconds_from_config
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

EVAL_SPLITS = ("val", "test")
# Seed offsets of the training-scenario eval (``eval_mode=both``) and of the
# stochastic-policy action sampling, relative to the eval seed base.
TRAIN_EVAL_SEED_OFFSET = 10_003
HOLDOUT_SEED_OFFSET = 50_003


class CrowdNavAdapter:
    """Domain train/eval surface for the CrowdNav simulator."""

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
        return CrowdNavPPOTrainer(stage="stage2")

    def _stage3_trainer(self) -> Any:
        if self._stage3_delegate is not None:
            return self._stage3_delegate
        return CrowdNavPPOTrainer(stage="stage3")

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
        raise ValueError(f"Unknown stage for CrowdNavAdapter: {stage!r}")

    def reevaluate(
        self, candidate: Any, *, round_index: int, config: Any, stage: str = "stage2"
    ) -> Any:
        """Eval-only pass of an already trained candidate."""
        stage_key = str(stage).strip().lower()
        if stage_key in ("3", "stage3", "iii"):
            trainer = self._stage3_trainer()
        else:
            trainer = self._stage2_trainer()
        return trainer.reevaluate(candidate, round_index=round_index, config=config)


class CrowdNavStage2Trainer:
    def __init__(self, adapter: CrowdNavAdapter) -> None:
        self.adapter = adapter

    def train_and_eval(self, candidate: Any, *, round_index: int, config: Any) -> Any:
        return self.adapter.train_and_eval(
            candidate, round_index=round_index, config=config, stage="stage2"
        )

    def reevaluate(self, candidate: Any, *, round_index: int, config: Any) -> Any:
        return self.adapter.reevaluate(candidate, round_index=round_index, config=config)


class CrowdNavStage3Trainer:
    def __init__(self, adapter: CrowdNavAdapter) -> None:
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

    def reevaluate(self, candidate: Any, *, round_index: int, config: Any) -> Any:
        return self.adapter.reevaluate(
            candidate, round_index=round_index, config=config, stage="stage3"
        )


_cuda_fallback_logged = False


def _log_cuda_fallback_once(requested: str) -> None:
    global _cuda_fallback_logged
    if _cuda_fallback_logged:
        return
    _cuda_fallback_logged = True
    logger.info(
        "CrowdNav PPO: device=%s requested, training on CPU (MLP policy is faster "
        "there); set RAISE_FORCE_CUDA=1 to use the GPU",
        requested,
    )


def _resolve_device(config: Any) -> str:
    """CPU unless CUDA is requested and ``RAISE_FORCE_CUDA`` is set."""
    device_raw = str(getattr(config, "device", "cpu"))
    # MLP policies train better on CPU by default (SB3 guidance).
    prefer_cpu = os.environ.get("RAISE_FORCE_CUDA", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    )
    if prefer_cpu and device_raw.startswith("cuda"):
        _log_cuda_fallback_once(device_raw)
        return "cpu"
    return device_raw if device_raw.startswith("cuda") else "cpu"


def _eval_with_step(
    env: RewardInjectedCrowdNavEnv,
    step_fn: Any,
    *,
    cases: Sequence[int],
    record_components: bool = False,
) -> ProxyMetrics:
    """Shared holdout rollout stats for PPO models and upstream controllers."""
    outcomes: List[str] = []
    nav_times: List[float] = []
    task_returns: List[float] = []
    selection_scores: List[float] = []
    danger_counts: List[int] = []
    step_counts: List[int] = []
    danger_dists: List[float] = []
    success_paths: List[float] = []
    min_seps: List[float] = []
    returns: List[float] = []
    records: List[Dict[str, Any]] = []
    for case in cases:
        obs, _info = env.reset(options={"case": int(case)})
        prev_xy = (float(env.sim.robot.px), float(env.sim.robot.py))
        done = False
        native: List[float] = []
        ep_return = ep_disc = 0.0
        discount = 1.0
        path = 0.0
        ep_min_sep = float("inf")
        ep_danger = 0
        ep_components: Dict[str, float] = {}
        info: Dict[str, Any] = {}
        while not done:
            obs, reward, terminated, truncated, info = step_fn(obs)
            native.append(float(info["raise_native_reward"]))
            ep_return += float(reward)
            ep_disc += discount * float(reward)
            discount *= PPO_GAMMA
            xy = (env.last_state.robot.px, env.last_state.robot.py)
            path += math.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])
            prev_xy = xy
            ep_min_sep = min(ep_min_sep, float(info["raise_min_separation"]))
            if info["raise_danger"]:
                ep_danger += 1
                danger_dists.append(float(info["raise_danger_min_dist"]))
            if record_components:
                for key, value in (info.get("raise_reward_components") or {}).items():
                    if key not in ep_components and len(ep_components) >= MAX_REWARD_COMPONENT_KEYS:
                        key = "_other"
                    ep_components[key] = ep_components.get(key, 0.0) + float(value)
            done = bool(terminated or truncated)

        outcome = str(info.get("raise_outcome") or "timeout")
        ep_time = float(info.get("raise_global_time", 0.0))
        task_ret = episode_task_return(native)
        selection_score = episode_selection_score(outcome)
        outcomes.append(outcome)
        nav_times.append(ep_time)
        task_returns.append(task_ret)
        selection_scores.append(selection_score)
        danger_counts.append(ep_danger)
        step_counts.append(len(native))
        min_seps.append(ep_min_sep)
        returns.append(ep_return)
        if outcome == "success":
            success_paths.append(path)
        record: Dict[str, Any] = {
            "case": int(case),
            "outcome": outcome,
            "steps": len(native),
            "time": round(ep_time, 3),
            "path_length": round(path, 3),
            "min_separation": round(ep_min_sep, 4),
            "danger_steps": int(ep_danger),
            "return": round(ep_return, 4),
            "disc_return": round(ep_disc, 4),
            "task_return": round(task_ret, 4),
            "selection_score": selection_score,
        }
        if record_components:
            record["components"] = {k: round(v, 4) for k, v in ep_components.items()}
        records.append(record)

    fields = episode_fields(
        outcomes,
        nav_times,
        task_returns,
        selection_scores,
        danger_counts,
        step_counts,
        danger_dists,
        time_limit=float(env.time_limit),
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
        "selection_score": fields["selection_score"],
        "fitness_se": fitness_standard_error(selection_scores),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_min_separation": float(np.mean(min_seps)) if min_seps else 0.0,
        "n_success": sum(1 for o in outcomes if o == "success"),
        "n_collision": sum(1 for o in outcomes if o == "collision"),
        "n_timeout": sum(1 for o in outcomes if o == "timeout"),
        "n_eval_episodes": len(outcomes),
    }
    metrics._crowdnav_episodes = records  # type: ignore[attr-defined]
    return metrics


def _eval_metrics(
    env: RewardInjectedCrowdNavEnv,
    model: Any,
    *,
    cases: Sequence[int],
    deterministic: bool = True,
    record_components: bool = False,
) -> ProxyMetrics:
    """
    Evaluate a policy on fixed upstream scenarios (``cases`` of ``env.phase``).

    Statistics follow CrowdNav++ upstream ``rl/evaluation.py`` and
    ``domains.crowdnav.metrics``. One compact record per episode is kept on
    ``metrics._crowdnav_episodes`` (outcome, steps, time, path length, min
    separation, danger steps, candidate-reward return undiscounted and
    PPO-discounted, and the upstream native task-return sum; with
    ``record_components`` also per-episode sums of the reward components).
    """

    def _step(obs: Any) -> Any:
        action, _ = model.predict(obs, deterministic=deterministic)
        return env.step(action)

    return _eval_with_step(
        env, _step, cases=cases, record_components=record_components
    )


def eval_act_fn_metrics(
    env: RewardInjectedCrowdNavEnv,
    act_fn: Any,
    *,
    cases: Sequence[int],
    record_components: bool = False,
) -> ProxyMetrics:
    """Same metrics as ``_eval_metrics``, but ``act_fn()`` returns an upstream action."""

    def _step(_obs: Any) -> Any:
        return env.step_upstream(act_fn())

    return _eval_with_step(
        env, _step, cases=cases, record_components=record_components
    )


def metrics_to_crowdnav_dict(metrics: ProxyMetrics) -> Dict[str, Any]:
    """ProxyMetrics.as_dict() + CrowdNav extras + official fitness."""
    payload: Dict[str, Any] = dict(metrics.as_dict())
    extras = getattr(metrics, "_crowdnav_extras", None) or {}
    payload.update({k: float(v) for k, v in extras.items()})
    return attach_fitness(payload)


def _n_envs(config: Any) -> int:
    try:
        return max(1, int(getattr(config, "n_envs", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _ppo_gamma(config: Any) -> float:
    raw = getattr(config, "ppo_gamma", None)
    if raw is None:
        return float(PPO_GAMMA)
    gamma = float(raw)
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"ppo_gamma must be in (0, 1], got {raw!r}")
    return gamma


def _eval_mode(config: Any) -> str:
    mode = str(getattr(config, "eval_mode", None) or "both").strip().lower()
    return mode if mode in ("both", "holdout_only") else "both"


def _eval_split(config: Any) -> str:
    split = str(getattr(config, "eval_split", None) or "val").strip().lower()
    if split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be one of {EVAL_SPLITS}, got {split!r}")
    return split


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


def _make_train_vec_env(reward_fn: Any, *, seed: int, n_envs: int, settings: Any):
    """DummyVecEnv of Monitor(RewardInjectedCrowdNavEnv) on upstream ``train`` scenarios.

    ``Monitor`` (no filename → no CSV) injects ``info["episode"]`` on episode
    end so SB3 can log ``rollout/ep_rew_mean`` / ``ep_len_mean``. Rendering is
    off (``render_mode=None``): no frames are drawn during training.
    """
    n = max(1, int(n_envs))

    def _thunk(rank: int):
        def _init():
            # PPO(seed=s) re-seeds VecEnv slot i with s + i on the first reset.
            env = RewardInjectedCrowdNavEnv(
                reward_fn if rank == 0 else clone_reward_fn(reward_fn),
                phase="train",
                seed=int(seed) + int(rank),
                settings=settings,
            )
            return Monitor(env)

        return _init

    return DummyVecEnv([_thunk(i) for i in range(n)])


def _normalize_stored_path(path: Any) -> str:
    """POSIX-style path for persistence; readers still accept Windows separators."""
    return Path(str(path)).as_posix()


def _resolve_existing_path(path: Any) -> Optional[str]:
    """Return an existing filesystem path, accepting ``/`` or ``\\`` separators."""
    if not path:
        return None
    raw = str(path)
    candidates = [
        raw,
        raw.replace("\\", os.sep).replace("/", os.sep),
        Path(raw).as_posix(),
    ]
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            return cand
        if os.path.isfile(cand + ".zip"):
            return cand + ".zip"
    return None


def _resolve_warm_start_path(candidate: Any) -> Optional[str]:
    md = getattr(candidate, "metadata", None) or {}
    for key in ("warm_start_checkpoint", "parent_checkpoint_path"):
        resolved = _resolve_existing_path(md.get(key))
        if resolved is not None:
            return resolved
    return None


def _crowdnav_policy_kwargs() -> Dict[str, Any]:
    return {
        "features_extractor_class": CrowdNavFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    }


def _ppo_progress_callback(
    total_timesteps: int,
    desc: str,
    dynamics: Optional[TrainingDynamicsRecorder] = None,
):
    """SB3 callback that drives ``raise_core.console.progress`` (and ``dynamics``)."""
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


class CrowdNavPPOTrainer:
    """Train SB3 PPO under a candidate reward; return ProxyMetrics or Stage3 bundle."""

    def __init__(self, *, stage: str = "stage2") -> None:
        self.stage = stage

    @property
    def _is_stage3(self) -> bool:
        return self.stage in ("stage3", "3", "iii")

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
        if candidate.reward_fn is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no reward_fn")

        settings = set_settings(settings_from_config(config))
        from raise_config import CONFIG

        if not hasattr(config, "train_env_steps"):
            raise AttributeError(
                "trainer config is missing required field 'train_env_steps' "
                f"(expected Stage2Config/Stage3Config; default is "
                f"{CONFIG.budget.stage2_train_steps})"
            )
        train_steps = int(config.train_env_steps)
        eval_episodes = int(getattr(config, "eval_episodes", 20))
        train_seed_index = kwargs.get("train_seed_index", None)
        if train_seed_index is None:
            # Legacy: PPO train seed confounded with round_index / candidate index.
            seed = int(getattr(config, "seed", 425)) + int(round_index)
        else:
            seed = int(getattr(config, "seed", 425)) + int(train_seed_index)
        n_envs = _n_envs(config)
        eval_mode = _eval_mode(config)
        device = _resolve_device(config)

        out_root = str(getattr(config, "output_root", "trained_models/crowdnav"))
        cid = str(candidate.candidate_id)
        out_dir = os.path.join(out_root, f"r{int(round_index):02d}_{cid}")
        os.makedirs(out_dir, exist_ok=True)

        stage_tag = "Stage III" if self._is_stage3 else "Stage II"
        warm_path = (
            _resolve_warm_start_path(candidate)
            if bool(getattr(config, "warm_start", False))
            else None
        )
        warm_bit = f" warm={os.path.basename(warm_path)}" if warm_path else ""
        console.status(
            f"PPO train {cid} | steps={train_steps:,} n_envs={n_envs} "
            f"eval={eval_episodes}({eval_mode}, {_eval_split(config)}) device={device} "
            f"round={round_index}{warm_bit}",
            stage=stage_tag,
        )

        train_env = _make_train_vec_env(
            candidate.reward_fn, seed=seed, n_envs=n_envs, settings=settings
        )
        normalize_reward = bool(getattr(config, "ppo_normalize_reward", False))
        if normalize_reward:
            from stable_baselines3.common.vec_env import VecNormalize

            train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
        try:
            n_steps_cap = max(16, int(getattr(config, "ppo_n_steps", 256) or 256))
            n_steps = min(n_steps_cap, max(16, train_steps // max(1, n_envs)))
            batch_size = min(64 * n_envs, max(8, n_steps * n_envs))
            n_epochs = max(1, int(getattr(config, "ppo_n_epochs", 10) or 10))
            ppo_gamma = _ppo_gamma(config)
            # Always build a fresh PPO so n_steps/batch_size match this VecEnv.
            # PPO.load + mutating n_steps leaves a stale RolloutBuffer (IndexError).
            # Warm-start copies policy weights only (new model → _last_obs is None,
            # so SB3 resets the env regardless of reset_num_timesteps).
            # MultiInputPolicy uses the domain extractor for padded human observations.
            model = PPO(
                "MultiInputPolicy",
                train_env,
                verbose=0,
                seed=seed,
                device=device,
                policy_kwargs=_crowdnav_policy_kwargs(),
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=ppo_gamma,
                learning_rate=3e-4,
                ent_coef=float(getattr(config, "ppo_ent_coef", 0.0) or 0.0),
            )
            if warm_path:
                try:
                    donor = PPO.load(
                        warm_path,
                        device=device,
                        policy_kwargs=_crowdnav_policy_kwargs(),
                    )
                    if not spaces_compatible(donor.action_space, model.action_space) or not (
                        spaces_compatible(donor.observation_space, model.observation_space)
                    ):
                        logger.warning(
                            "Warm-start skipped for %s: parent spaces %s/%s != current %s/%s",
                            cid,
                            donor.observation_space,
                            donor.action_space,
                            model.observation_space,
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
                if bool(getattr(config, "record_training_dynamics", False))
                else None
            )
            cb = _ppo_progress_callback(
                train_steps, desc=f"PPO {cid} ({stage_tag})", dynamics=dynamics
            )
            model.learn(total_timesteps=max(1, train_steps), progress_bar=False, callback=cb)
            train_wall = time.perf_counter() - t0
            sps = float(train_steps) / max(train_wall, 1e-6)
            console.status(
                f"PPO train done {cid} in {console.format_seconds(train_wall)} "
                f"({sps:.0f} steps/s)",
                stage=stage_tag,
            )

            ckpt = Path(out_dir, "model.zip").as_posix()
            model.save(ckpt)
        finally:
            try:
                train_env.close()
            except Exception:  # noqa: BLE001
                pass

        holdout_metrics, metrics_dict = self._evaluate(
            model,
            candidate,
            config=config,
            seed=seed,
            settings=settings,
            stage_tag=stage_tag,
        )
        metrics_dict["n_envs"] = float(n_envs)
        metrics_dict["warm_start"] = 1.0 if warm_path else 0.0
        md = dict(candidate.metadata or {})
        md["checkpoint_path"] = ckpt
        md["train_steps"] = int(train_steps)
        # PPO rounds up to a multiple of n_envs * n_steps (e.g. 5_000_192).
        md["actual_train_steps"] = int(getattr(model, "num_timesteps", train_steps) or train_steps)
        md["train_seed"] = int(seed)
        md["train_seed_index"] = (
            None if train_seed_index is None else int(train_seed_index)
        )
        md["ppo_n_steps"] = int(n_steps)
        md["ppo_gamma"] = float(ppo_gamma)
        md["ppo_normalize_reward"] = bool(normalize_reward)
        md["train_wall_seconds"] = float(train_wall)
        md["train_steps_per_second"] = float(sps)
        if dynamics is not None:
            md["training_dynamics"] = dynamics.summary()
        candidate.metadata = md

        if self._is_stage3:
            return TrainEvalBundle(metrics=holdout_metrics, checkpoint_path=ckpt)
        return holdout_metrics

    def reevaluate(self, candidate: Any, *, round_index: int, config: Any) -> Any:
        """Evaluate the saved policy again (no training) on ``config``'s eval scenarios.

        A genome carried into a later generation is ranked against candidates
        evaluated on that generation's scenarios, so its fitness must come
        from the same scenarios rather than from the generation it won.
        """
        with StallWatchdog(
            stall_seconds_from_config(config),
            label=str(getattr(candidate, "candidate_id", "")),
        ):
            md = dict(candidate.metadata or {})
            ckpt = md.get("checkpoint_path")
            if not ckpt or not os.path.isfile(str(ckpt)):
                raise FileNotFoundError(
                    f"{candidate.candidate_id}: no saved policy to re-evaluate ({ckpt!r})"
                )
            settings = set_settings(settings_from_config(config))
            model = PPO.load(
                str(ckpt),
                device=_resolve_device(config),
                policy_kwargs=_crowdnav_policy_kwargs(),
            )
            holdout_metrics, _ = self._evaluate(
                model,
                candidate,
                config=config,
                seed=int(getattr(config, "seed", 425)) + int(round_index),
                settings=settings,
                stage_tag="Stage III re-eval" if self._is_stage3 else "Stage II re-eval",
            )
            return holdout_metrics

    def _evaluate(
        self,
        model: Any,
        candidate: Any,
        *,
        config: Any,
        seed: int,
        settings: Any,
        stage_tag: str,
    ) -> Any:
        """Holdout evaluation (plus training scenarios in ``both`` mode).

        Holdout scenarios are upstream ``val`` cases (``eval_split="val"``,
        selection) or ``test`` cases (``eval_split="test"``, the final blind
        evaluation); training uses the disjoint ``train`` cases. Writes
        ``last_metrics``, ``fitness`` and the eval fields onto the candidate
        metadata; returns ``(holdout ProxyMetrics, metrics dict)``.
        """
        cid = str(candidate.candidate_id)
        eval_episodes = int(getattr(config, "eval_episodes", 20))
        eval_mode = _eval_mode(config)
        split = _eval_split(config)
        console.status(
            f"PPO eval {cid} | episodes={eval_episodes} ({eval_mode}, {split})",
            stage=stage_tag,
        )
        t1 = time.perf_counter()
        eval_deterministic = bool(getattr(config, "eval_deterministic", False))
        eval_seed_raw = getattr(config, "eval_seed_base", None)
        eval_seed = seed if eval_seed_raw is None else int(eval_seed_raw)

        train_dict: Dict[str, Any] = {}
        if eval_mode == "both":
            train_env = RewardInjectedCrowdNavEnv(
                candidate.reward_fn, phase="train", settings=settings
            )
            if not eval_deterministic:
                model.set_random_seed(int(eval_seed) + TRAIN_EVAL_SEED_OFFSET)
            try:
                train_cases = eval_cases(
                    eval_seed + TRAIN_EVAL_SEED_OFFSET,
                    eval_episodes,
                    int(train_env.sim.case_size["train"]),
                )
                train_dict = metrics_to_crowdnav_dict(
                    _eval_metrics(
                        train_env,
                        model,
                        cases=train_cases,
                        deterministic=eval_deterministic,
                    )
                )
            finally:
                train_env.close()

        holdout_env = RewardInjectedCrowdNavEnv(candidate.reward_fn, phase=split, settings=settings)
        if not eval_deterministic:
            # Action sampling must not depend on RNG consumed during training.
            model.set_random_seed(int(eval_seed) + HOLDOUT_SEED_OFFSET)
        try:
            holdout_cases = eval_cases(
                eval_seed, eval_episodes, int(holdout_env.sim.case_size[split])
            )
            holdout_metrics = _eval_metrics(
                holdout_env,
                model,
                cases=holdout_cases,
                deterministic=eval_deterministic,
                record_components=bool(getattr(config, "record_reward_components", False)),
            )
            context = {
                "split": split,
                "selection_pool": split,
                "case_set_size": int(holdout_env.sim.case_size[split]),
                "cases": [int(c) for c in holdout_cases],
                "human_num": int(holdout_env.human_num),
                "scenario": split,
                "time_limit_s": float(holdout_env.time_limit),
                "time_step_s": float(holdout_env.time_step),
                "circle_radius_m": float(holdout_env.sim.circle_radius),
                "v_pref_mps": float(holdout_env.v_pref),
                "ideal_nav_time_s": 2.0 * float(holdout_env.sim.circle_radius) / float(holdout_env.v_pref),
                "kinematics": settings.kinematics,
                "action_mode": settings.action_mode,
                "deterministic": bool(eval_deterministic),
            }
        finally:
            holdout_env.close()
        eval_wall = time.perf_counter() - t1

        holdout_dict = metrics_to_crowdnav_dict(holdout_metrics)
        metrics_dict: Dict[str, Any] = dict(holdout_dict)
        metrics_dict["eval_profile"] = "holdout"
        if train_dict:
            metrics_dict["train_dist"] = {
                k: v for k, v in train_dict.items() if k not in ("holdout", "train_dist")
            }
        metrics_dict["holdout"] = {
            k: v for k, v in holdout_dict.items() if k not in ("holdout", "train_dist")
        }
        metrics_dict["holdout_episodes"] = list(getattr(holdout_metrics, "_crowdnav_episodes", []))
        metrics_dict["eval_context"] = context
        metrics_dict = attach_fitness(metrics_dict)

        console.status(
            f"PPO metrics {cid} | {format_crowdnav_metrics_line(metrics_dict)} | "
            f"eval {console.format_seconds(eval_wall)}",
            stage=stage_tag,
        )

        md = dict(candidate.metadata or {})
        md["last_metrics"] = metrics_dict
        fit = float(metrics_dict["fitness"])
        md["fitness"] = fit
        md["selection_scalar"] = fit  # alias
        md["eval_seed_base"] = int(eval_seed)
        md["eval_split"] = split
        md["eval_episodes"] = int(eval_episodes)
        md["eval_deterministic"] = bool(eval_deterministic)
        md["eval_wall_seconds"] = float(eval_wall)
        candidate.metadata = md
        return holdout_metrics, metrics_dict


def make_stage2_trainer(*, use_stub: bool = False) -> CrowdNavStage2Trainer:
    if use_stub:
        adapter = CrowdNavAdapter(stage2_delegate=StubPolicyTrainer())
    else:
        adapter = CrowdNavAdapter()
    return CrowdNavStage2Trainer(adapter)


def make_stage3_trainer(*, use_stub: bool = False) -> CrowdNavStage3Trainer:
    if use_stub:
        adapter = CrowdNavAdapter(stage3_delegate=Stage3StubTrainer())
    else:
        adapter = CrowdNavAdapter()
    return CrowdNavStage3Trainer(adapter)

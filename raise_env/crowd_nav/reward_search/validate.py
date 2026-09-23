"""
RAISE Stage III — full-scale PPO training + D.3 refinement + Table 6 H-sweep.

Structurally identical to Stage II, but:
  - ``--algo ppo`` with fixed K3 environment steps (no early stopping)
  - G3=3 rounds, E3=500 evaluation episodes (Table 6)
  - Full-episode evaluation (env time_limit), not Stage II's T_short
  - After final round, re-evaluate at H in {5, 10, 15, 20} and report SR/CR/TR

Paper K3 = 1e7 (Table 6). Default ``STAGE3_STEPS`` is smaller for local iteration;
scale up on a GPU cluster for paper-faithful runs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from crowd_nav.reward_search.explore import RewardCandidate
from crowd_nav.reward_search import console
from crowd_nav.reward_search.llm import (
    LLMClient,
    extract_python_code,
    normalize_to_compute_reward,
)
from crowd_nav.reward_search.parallelism import (
    default_num_mini_batch,
    resolve_num_processes,
)
from crowd_nav.reward_search.prompts import (
    D3_SYSTEM_PROMPT,
    format_d3_refinement,
    format_d3_repair,
)
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.selection import (
    candidate_nav_scalar,
    is_same_genome,
)
from crowd_nav.reward_search.refine import ProxyMetrics, evaluate_proxy_policy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# K3 training budget (Table 6)
# ---------------------------------------------------------------------------
#
# Paper (RAISE Table 6 / §C.3): K3 = 10^7 environment steps with PPO on
# NVIDIA A6000-class GPUs.
#
# Default below is deliberately smaller for practical iteration on a laptop /
# single workstation. For paper-faithful Stage III on a GPU cluster, set:
#
#     STAGE3_STEPS = int(1e7)
#     # or: Stage3Config(train_env_steps=int(1e7), num_processes=16, ...)
#
STAGE3_STEPS: int = int(5e5)  # practical default; paper uses 1e7

STAGE3_PAPER_STEPS: int = int(1e7)
STAGE3_HUMAN_COUNTS: Tuple[int, ...] = (5, 10, 15, 20)  # Table 6 H sweep


@dataclass
class Stage3Config:
    """RAISE Table 6 Stage III defaults (with practical K3 override)."""

    population_size: int = 8
    rounds: int = 3  # G3
    # K3 — see STAGE3_STEPS docstring / paper 1e7 note above.
    train_env_steps: int = STAGE3_STEPS
    eval_episodes: int = 500  # E3
    # Full episodes: None → use env time_limit / time_step as step cap.
    horizon_steps: Optional[int] = None
    algo: str = "ppo"
    # None → resolve_num_processes() at train time (min(16, cpu_count-1)).
    num_processes: Optional[int] = None
    num_steps: int = 30  # PPO rollout length (matches seq_length)
    # None → default_num_mini_batch(resolved num_processes).
    num_mini_batch: Optional[int] = None
    ppo_epoch: int = 5
    seed: int = 425
    # Choice (a): GST-inferred obs parity with CrowdNav++ (AUDIT.md §8.2).
    env_name: str = "CrowdSimPredRealGST-v0"
    predict_method: str = "inferred"
    randomization_regime: str = "without_random"
    output_root: str = "trained_models/stage3"
    device: str = "cpu"
    train_human_num: int = 20  # training population size (obs / Policy width)
    human_counts: Tuple[int, ...] = STAGE3_HUMAN_COUNTS
    # Alg. 1 has no elitism — defaults off; opt in via pipeline ``elitism=True``.
    protect_elite_refine: bool = False
    inject_elite: bool = False
    # Crash-safe resume (``{run}/stage3/checkpoint.json`` + mid-PPO progress).
    resume: bool = True
    # Write policy weights every N PPO updates (0 = only final save).
    save_interval_updates: int = 50


@dataclass
class Stage3RoundRecord:
    round_index: int
    candidate_id: str
    metrics: ProxyMetrics
    refined: bool
    kept_previous: bool
    validation_error: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass
class TrainEvalBundle:
    """Result of one Stage III train+eval (metrics + artifacts for H-sweep)."""

    metrics: ProxyMetrics
    checkpoint_path: Optional[str] = None
    algo_args: Any = None
    env_config: Any = None
    actor_critic: Any = None
    device: Any = None


@dataclass
class HumanSweepReport:
    """Table 6 generalization: SR/CR/TR (and full metrics) per H."""

    candidate_id: str
    by_human_count: Dict[int, ProxyMetrics]

    def summary_table(self) -> str:
        lines = [f"candidate={self.candidate_id}", "H\tSR\tCR\tTR"]
        for h in sorted(self.by_human_count):
            m = self.by_human_count[h]
            lines.append(f"{h}\t{m.sr:.4f}\t{m.cr:.4f}\t{m.tr:.4f}")
        return "\n".join(lines)


def _stable_code_hash(code: str) -> int:
    return int(hashlib.md5(code.encode("utf-8")).hexdigest(), 16) % 10_000


def _next_revision_id(candidate_id: str, *, stage_tag: str = "v3") -> str:
    m = re.search(r"_v(\d+)$", candidate_id)
    if m:
        return f"{candidate_id[: m.start()]}_v{int(m.group(1)) + 1}"
    n = 3 if stage_tag == "v3" else 2
    return f"{candidate_id}_v{n}"


def _v3_candidate_id(candidate_id: str) -> str:
    """Tag a successfully refined Stage-III revision (incrementing ``_vN``)."""
    return _next_revision_id(candidate_id, stage_tag="v3")


def _unique_trained_snapshot_id(candidate_id: str, *, round_index: int, snap_index: int) -> str:
    return f"{candidate_id}__r{int(round_index)}_s{int(snap_index)}"


def _resolve_horizon(config: Stage3Config, env_config) -> int:
    if config.horizon_steps is not None:
        return int(config.horizon_steps)
    return max(
        1, int(float(env_config.env.time_limit) / float(env_config.env.time_step))
    )


class PolicyTrainer(ABC):
    """Train + evaluate a full policy under a candidate reward."""

    @abstractmethod
    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        resume_update: int = 0,
        resume_weights_path: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> TrainEvalBundle:
        raise NotImplementedError

    @abstractmethod
    def evaluate_at_human_counts(
        self,
        candidate: RewardCandidate,
        bundle: TrainEvalBundle,
        *,
        config: Stage3Config,
        human_counts: Optional[Sequence[int]] = None,
    ) -> HumanSweepReport:
        raise NotImplementedError


class StubPolicyTrainer(PolicyTrainer):
    """Deterministic no-op trainer for pytest (seconds, not GPU-days)."""

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        resume_update: int = 0,
        resume_weights_path: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> TrainEvalBundle:
        del resume_update, resume_weights_path, progress_callback
        base = (_stable_code_hash(candidate.code) % 100) / 100.0
        jitter = 0.01 * (round_index % 3)
        sr = min(1.0, max(0.0, 0.5 + 0.4 * base + jitter))
        cr = min(1.0, max(0.0, 0.3 * (1.0 - base)))
        tr = max(0.0, 1.0 - sr - cr)
        metrics = ProxyMetrics(
            sr=sr,
            cr=cr,
            tr=tr,
            nt=12.0 + 4.0 * (1.0 - base),
            pl=14.0 + 6.0 * (1.0 - base),
            itr=4.0 + 15.0 * (1.0 - base),
            sd=0.25 + 0.25 * base,
        )
        return TrainEvalBundle(metrics=metrics, checkpoint_path=None)

    def evaluate_at_human_counts(
        self,
        candidate: RewardCandidate,
        bundle: TrainEvalBundle,
        *,
        config: Stage3Config,
        human_counts: Optional[Sequence[int]] = None,
    ) -> HumanSweepReport:
        counts = tuple(human_counts or config.human_counts)
        by_h: Dict[int, ProxyMetrics] = {}
        for h in counts:
            dens = h / 20.0
            sr = min(1.0, max(0.0, bundle.metrics.sr - 0.05 * dens))
            cr = min(1.0, max(0.0, bundle.metrics.cr + 0.04 * dens))
            tr = max(0.0, 1.0 - sr - cr)
            by_h[int(h)] = ProxyMetrics(
                sr=sr,
                cr=cr,
                tr=tr,
                nt=bundle.metrics.nt,
                pl=bundle.metrics.pl,
                itr=bundle.metrics.itr + 2.0 * dens,
                sd=max(0.05, bundle.metrics.sd - 0.02 * dens),
            )
        return HumanSweepReport(candidate_id=candidate.candidate_id, by_human_count=by_h)


def _stage3_train_argv(
    config: Stage3Config, candidate_id: str, round_index: int
) -> list:
    out_dir = os.path.join(
        config.output_root, f"r{round_index:02d}_{candidate_id}"
    )
    nproc = resolve_num_processes(config.num_processes)
    nbatch = (
        max(1, int(config.num_mini_batch))
        if config.num_mini_batch is not None
        else default_num_mini_batch(nproc)
    )
    if nbatch > nproc:
        nbatch = nproc
    argv = [
        "stage3_train",
        "--algo",
        config.algo,
        "--num-env-steps",
        str(int(config.train_env_steps)),
        "--num-processes",
        str(nproc),
        "--num-steps",
        str(config.num_steps),
        "--seq_length",
        str(config.num_steps),
        "--num-mini-batch",
        str(nbatch),
        "--ppo-epoch",
        str(config.ppo_epoch),
        "--seed",
        str(config.seed + round_index),
        "--env-name",
        config.env_name,
        "--output_dir",
        out_dir,
        "--overwrite",
    ]
    if config.device != "cuda" or not torch.cuda.is_available():
        if config.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Stage III: --device cuda requested but PyTorch has no CUDA "
                "(version=%s); forcing --no-cuda.",
                torch.__version__,
            )
        argv.append("--no-cuda")
    return argv


def _parse_stage3_algo_args(config: Stage3Config, candidate_id: str, round_index: int):
    from arguments import get_args

    argv = _stage3_train_argv(config, candidate_id, round_index)
    saved = list(sys.argv)
    try:
        sys.argv = argv
        algo_args = get_args()
        from crowd_nav.configs.config import Config  # noqa: F401

        return algo_args
    finally:
        sys.argv = saved


def _make_full_env_config(config: Stage3Config):
    from copy import copy

    from crowd_nav.configs.config import Config
    from crowd_nav.reward_search.regime import (
        apply_regime_to_config,
        env_name_for_predict_method,
    )

    cfg = Config()
    apply_regime_to_config(
        cfg,
        config.randomization_regime,
        predict_method=config.predict_method,
        entry_point="stage3",
    )
    expected = env_name_for_predict_method(config.predict_method)
    if config.env_name != expected:
        logger.warning(
            "Stage3Config.env_name=%s overridden to %s for predict_method=%s",
            config.env_name,
            expected,
            config.predict_method,
        )
        config.env_name = expected
    cfg.robot.policy = "selfAttn_merge_srnn"
    if "env" not in cfg.__dict__:
        cfg.env = copy(cfg.env)
    if "sim" not in cfg.__dict__:
        cfg.sim = copy(cfg.sim)
    cfg.sim.human_num = int(config.train_human_num)
    cfg.sim.human_num_range = 0
    cfg.env.test_size = int(config.eval_episodes)
    return cfg


class RealPolicyTrainer(PolicyTrainer):
    """Fresh PPO for fixed K3 steps + full-episode eval. No early stopping."""

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        resume_update: int = 0,
        resume_weights_path: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ) -> TrainEvalBundle:
        if candidate.reward_fn is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no reward_fn")

        from crowd_nav.reward_search.stage3_checkpoint import (
            clear_train_progress,
            save_train_progress,
        )
        from rl import ppo
        from rl.networks.envs import make_vec_envs
        from rl.networks.model import Policy
        from rl.networks.storage import RolloutStorage

        algo_args = _parse_stage3_algo_args(
            config, candidate.candidate_id, round_index
        )
        env_config = _make_full_env_config(config)
        horizon = _resolve_horizon(config, env_config)

        out_dir = algo_args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        device = torch.device(
            "cuda" if algo_args.cuda and torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(algo_args.seed)
        torch.set_num_threads(1)

        envs = make_vec_envs(
            algo_args.env_name,
            algo_args.seed,
            algo_args.num_processes,
            algo_args.gamma,
            None,
            device,
            False,
            config=env_config,
            pretext_wrapper=env_config.env.use_wrapper,
            reward_fn=candidate.reward_fn,
        )

        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=env_config.robot.policy,
        )
        nn.DataParallel(actor_critic).to(device)

        start_j = max(0, int(resume_update))
        if resume_weights_path and os.path.isfile(resume_weights_path):
            state = torch.load(resume_weights_path, map_location=device)
            actor_critic.load_state_dict(state)
            console.status(
                f"resume PPO {candidate.candidate_id} from update={start_j} "
                f"({resume_weights_path})",
                stage="Stage III",
            )

        agent = ppo.PPO(
            actor_critic,
            algo_args.clip_param,
            algo_args.ppo_epoch,
            algo_args.num_mini_batch,
            algo_args.value_loss_coef,
            algo_args.entropy_coef,
            lr=algo_args.lr,
            eps=algo_args.eps,
            max_grad_norm=algo_args.max_grad_norm,
        )

        rollouts = RolloutStorage(
            algo_args.num_steps,
            algo_args.num_processes,
            envs.observation_space.spaces,
            envs.action_space,
            algo_args.human_node_rnn_size,
            algo_args.human_human_edge_rnn_size,
        )

        obs = envs.reset()
        for key in rollouts.obs:
            rollouts.obs[key][0].copy_(obs[key])
        rollouts.to(device)

        num_updates = (
            int(algo_args.num_env_steps)
            // algo_args.num_steps
            // algo_args.num_processes
        )
        save_every = max(0, int(getattr(config, "save_interval_updates", 50) or 0))
        console.status(
            f"training {candidate.candidate_id} round={round_index} "
            f"PPO K3={config.train_env_steps} updates={num_updates} "
            f"start_j={start_j} nproc={algo_args.num_processes}",
            stage="Stage III",
        )
        pbar = console.progress(
            total=max(1, num_updates),
            desc=f"[Stage III] {candidate.candidate_id} PPO",
            unit="upd",
        )
        if start_j > 0 and hasattr(pbar, "update"):
            try:
                pbar.update(min(start_j, max(1, num_updates)))
            except Exception:  # noqa: BLE001
                pass
        t_train = time.perf_counter()
        last_ckpt_path = os.path.join(ckpt_dir, f"{max(num_updates - 1, 0):05d}.pt")
        try:
            for j in range(start_j, num_updates):
                for step in range(algo_args.num_steps):
                    with torch.no_grad():
                        rollouts_obs = {k: rollouts.obs[k][step] for k in rollouts.obs}
                        rollouts_hidden_s = {
                            k: rollouts.recurrent_hidden_states[k][step]
                            for k in rollouts.recurrent_hidden_states
                        }
                        value, action, action_log_prob, recurrent_hidden_states = (
                            actor_critic.act(
                                rollouts_obs, rollouts_hidden_s, rollouts.masks[step]
                            )
                        )
                    obs, reward, done, infos = envs.step(action)
                    masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if "bad_transition" in info else [1.0] for info in infos]
                    )
                    rollouts.insert(
                        obs,
                        recurrent_hidden_states,
                        action,
                        action_log_prob,
                        value,
                        reward,
                        masks,
                        bad_masks,
                    )

                with torch.no_grad():
                    rollouts_obs = {k: rollouts.obs[k][-1] for k in rollouts.obs}
                    rollouts_hidden_s = {
                        k: rollouts.recurrent_hidden_states[k][-1]
                        for k in rollouts.recurrent_hidden_states
                    }
                    next_value = actor_critic.get_value(
                        rollouts_obs, rollouts_hidden_s, rollouts.masks[-1]
                    ).detach()

                rollouts.compute_returns(
                    next_value,
                    algo_args.use_gae,
                    algo_args.gamma,
                    algo_args.gae_lambda,
                    algo_args.use_proper_time_limits,
                )
                agent.update(rollouts)
                rollouts.after_update()
                pbar.update(1)

                done_update = j + 1
                should_save = (
                    save_every > 0
                    and (
                        done_update % save_every == 0
                        or done_update >= num_updates
                    )
                )
                if should_save:
                    mid_path = os.path.join(ckpt_dir, f"{j:05d}.pt")
                    torch.save(actor_critic.state_dict(), mid_path)
                    last_ckpt_path = mid_path
                    save_train_progress(
                        out_dir,
                        round_index=round_index,
                        candidate_id=candidate.candidate_id,
                        update_j=done_update,
                        num_updates=num_updates,
                        weights_path=mid_path,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "round_index": int(round_index),
                                "candidate_id": str(candidate.candidate_id),
                                "update_j": int(done_update),
                                "num_updates": int(num_updates),
                                "weights_path": mid_path,
                            }
                        )
        except Exception as exc:  # noqa: BLE001
            console.fail(
                f"train crashed for {candidate.candidate_id} round={round_index}: {exc}",
                stage="Stage III",
            )
            raise
        finally:
            pbar.close()
        console.status(
            f"train done for {candidate.candidate_id} in "
            f"{console.format_seconds(time.perf_counter() - t_train)}; evaluating...",
            stage="Stage III",
        )

        ckpt_path = os.path.join(ckpt_dir, f"{max(num_updates - 1, 0):05d}.pt")
        torch.save(actor_critic.state_dict(), ckpt_path)
        clear_train_progress(out_dir)
        envs.close()

        metrics = evaluate_proxy_policy(
            actor_critic,
            candidate.reward_fn,
            algo_args,
            env_config,
            device,
            n_episodes=config.eval_episodes,
            horizon_steps=horizon,
            human_num=config.train_human_num,
        )
        with open(os.path.join(out_dir, "full_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(metrics.feedback_text() + "\n")
            f.write(f"checkpoint={ckpt_path}\n")
            f.write(f"K3={config.train_env_steps} (paper={STAGE3_PAPER_STEPS})\n")
        console.status(
            f"eval {candidate.candidate_id}: {metrics.feedback_text()}",
            stage="Stage III",
        )

        return TrainEvalBundle(
            metrics=metrics,
            checkpoint_path=ckpt_path,
            algo_args=algo_args,
            env_config=env_config,
            actor_critic=actor_critic,
            device=device,
        )

    def evaluate_at_human_counts(
        self,
        candidate: RewardCandidate,
        bundle: TrainEvalBundle,
        *,
        config: Stage3Config,
        human_counts: Optional[Sequence[int]] = None,
    ) -> HumanSweepReport:
        if candidate.reward_fn is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no reward_fn")
        if bundle.actor_critic is None or bundle.algo_args is None:
            raise ValueError("TrainEvalBundle missing policy artifacts for H-sweep")

        counts = tuple(human_counts or config.human_counts)
        env_config = bundle.env_config or _make_full_env_config(config)
        horizon = _resolve_horizon(config, env_config)
        by_h: Dict[int, ProxyMetrics] = {}
        for h in counts:
            console.status(
                f"H-sweep {candidate.candidate_id} H={h} E={config.eval_episodes}",
                stage="Stage III",
            )
            by_h[int(h)] = evaluate_proxy_policy(
                bundle.actor_critic,
                candidate.reward_fn,
                bundle.algo_args,
                env_config,
                bundle.device,
                n_episodes=config.eval_episodes,
                horizon_steps=horizon,
                human_num=int(h),
            )
        report = HumanSweepReport(
            candidate_id=candidate.candidate_id, by_human_count=by_h
        )
        if bundle.checkpoint_path:
            sweep_path = os.path.join(
                os.path.dirname(os.path.dirname(bundle.checkpoint_path)),
                "h_sweep.txt",
            )
            with open(sweep_path, "w", encoding="utf-8") as f:
                f.write(report.summary_table() + "\n")
                for h, m in sorted(by_h.items()):
                    f.write(f"H={h}: {m.feedback_text()}\n")
        return report


class Stage3Runner:
    """G3 rounds of full PPO train/eval + D.3 refinement + final H-sweep."""

    def __init__(
        self,
        llm: LLMClient,
        trainer: PolicyTrainer,
        *,
        validator: Optional[RewardValidator] = None,
        config: Optional[Stage3Config] = None,
    ) -> None:
        self.llm = llm
        self.trainer = trainer
        self.validator = validator or RewardValidator()
        self.config = config or Stage3Config()
        self.history: List[Stage3RoundRecord] = []
        self.validation_failures: List[Dict[str, Any]] = []
        self.last_bundles: Dict[str, TrainEvalBundle] = {}
        self.sweep_reports: List[HumanSweepReport] = []
        self.trained_snapshots: List[RewardCandidate] = []
        self.best_trained: Optional[RewardCandidate] = None
        # Optional paper-scale resume (set by PaperScaleRunner).
        self.checkpoint_store = None  # type: ignore[assignment]
        self.checkpoint_seed: int = int(self.config.seed)
        # Run-dir crash-safe resume (``{run}/stage3/checkpoint.json``).
        self._ckpt_dir: Optional[str] = None
        self._run_h_sweep: bool = True
        self._history_payload: List[Dict[str, Any]] = []

    def _resolve_ckpt_dir(self) -> str:
        from crowd_nav.reward_search.stage3_checkpoint import stage3_dir_from_train_root

        if self._ckpt_dir:
            return self._ckpt_dir
        return stage3_dir_from_train_root(self.config.output_root)

    def _config_snapshot(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "population_size": int(cfg.population_size),
            "rounds": int(cfg.rounds),
            "train_env_steps": int(cfg.train_env_steps),
            "eval_episodes": int(cfg.eval_episodes),
            "seed": int(cfg.seed),
            "device": str(cfg.device),
            "num_processes": cfg.num_processes,
            "train_human_num": int(cfg.train_human_num),
            "predict_method": str(cfg.predict_method),
            "randomization_regime": str(cfg.randomization_regime),
            "output_root": str(cfg.output_root),
            "save_interval_updates": int(getattr(cfg, "save_interval_updates", 50)),
        }

    def _persist(
        self,
        *,
        status: str,
        phase: str,
        round_index: int,
        candidate_index: int,
        round_input_population: Sequence[RewardCandidate],
        round_output_partial: Sequence[RewardCandidate],
        train_progress: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not bool(getattr(self.config, "resume", True)):
            return
        from crowd_nav.reward_search.stage3_checkpoint import (
            build_checkpoint,
            save_checkpoint,
        )

        payload = build_checkpoint(
            status=status,
            phase=phase,
            round_index=round_index,
            candidate_index=candidate_index,
            rounds=int(self.config.rounds),
            round_input_population=round_input_population,
            round_output_partial=round_output_partial,
            history=self._history_payload,
            best_trained=self.best_trained,
            trained_snapshots=self.trained_snapshots,
            run_h_sweep=bool(self._run_h_sweep),
            train_progress=train_progress,
            config=self._config_snapshot(),
        )
        path = save_checkpoint(self._resolve_ckpt_dir(), payload)
        logger.info("Stage III checkpoint → %s (%s/%s)", path, status, phase)

    def _restore_history_from_payload(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.history = []
        self._history_payload = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            md = row.get("metrics") or {}
            metrics = ProxyMetrics(
                sr=float(md.get("SR", md.get("sr", 0.0))),
                cr=float(md.get("CR", md.get("cr", 0.0))),
                tr=float(md.get("TR", md.get("tr", 0.0))),
                nt=float(md.get("NT", md.get("nt", 0.0))),
                pl=float(md.get("PL", md.get("pl", 0.0))),
                itr=float(md.get("ITR", md.get("itr", 0.0))),
                sd=float(md.get("SD", md.get("sd", 0.0))),
            )
            rec = Stage3RoundRecord(
                round_index=int(row.get("round_index", 0)),
                candidate_id=str(row.get("candidate_id", "")),
                metrics=metrics,
                refined=bool(row.get("refined")),
                kept_previous=bool(row.get("kept_previous")),
                validation_error=row.get("validation_error"),
                checkpoint_path=row.get("checkpoint_path"),
            )
            self.history.append(rec)
            self._history_payload.append(dict(row))
            # Lightweight bundle so H-sweep can reload weights after process restart.
            if rec.checkpoint_path:
                self.last_bundles[rec.candidate_id] = TrainEvalBundle(
                    metrics=metrics,
                    checkpoint_path=rec.checkpoint_path,
                )

    def _append_history_record(self, rec: Stage3RoundRecord) -> None:
        from crowd_nav.reward_search.stage3_checkpoint import history_record_to_dict

        self.history.append(rec)
        self._history_payload.append(
            history_record_to_dict(
                round_index=rec.round_index,
                candidate_id=rec.candidate_id,
                metrics=rec.metrics.as_dict(),
                refined=rec.refined,
                kept_previous=rec.kept_previous,
                checkpoint_path=rec.checkpoint_path,
                validation_error=rec.validation_error,
            )
        )

    def _reload_policy_into_bundle(
        self, candidate: RewardCandidate, bundle: TrainEvalBundle
    ) -> TrainEvalBundle:
        """Load ``.pt`` into a fresh Policy when actor_critic was not kept in memory."""
        if bundle.actor_critic is not None and bundle.algo_args is not None:
            return bundle
        if not bundle.checkpoint_path or not os.path.isfile(bundle.checkpoint_path):
            return bundle
        from rl.networks.envs import make_vec_envs
        from rl.networks.model import Policy

        algo_args = _parse_stage3_algo_args(
            self.config, candidate.candidate_id, int(
                (candidate.metadata or {}).get("trained_round", 0)
            )
        )
        env_config = _make_full_env_config(self.config)
        device = torch.device(
            "cuda" if algo_args.cuda and torch.cuda.is_available() else "cpu"
        )
        envs = make_vec_envs(
            algo_args.env_name,
            algo_args.seed,
            1,
            algo_args.gamma,
            None,
            device,
            False,
            config=env_config,
            pretext_wrapper=env_config.env.use_wrapper,
            reward_fn=candidate.reward_fn,
        )
        try:
            actor_critic = Policy(
                envs.observation_space.spaces,
                envs.action_space,
                base_kwargs=algo_args,
                base=env_config.robot.policy,
            )
            nn.DataParallel(actor_critic).to(device)
            state = torch.load(bundle.checkpoint_path, map_location=device)
            actor_critic.load_state_dict(state)
        finally:
            envs.close()
        return TrainEvalBundle(
            metrics=bundle.metrics,
            checkpoint_path=bundle.checkpoint_path,
            algo_args=algo_args,
            env_config=env_config,
            actor_critic=actor_critic,
            device=device,
        )

    def _record_trained_snapshot(
        self,
        candidate: RewardCandidate,
        bundle: TrainEvalBundle,
        *,
        round_index: int,
    ) -> RewardCandidate:
        """Freeze the genome that produced ``bundle.metrics`` (pre-refine)."""
        from crowd_nav.reward_search.proxy_consistency import genome_key

        source_id = str(candidate.candidate_id)
        snap_index = len(self.trained_snapshots)
        snapshot = replace(
            candidate,
            candidate_id=_unique_trained_snapshot_id(
                source_id, round_index=round_index, snap_index=snap_index
            ),
            metadata={
                **(candidate.metadata or {}),
                "last_metrics": bundle.metrics.as_dict(),
                "checkpoint_path": bundle.checkpoint_path,
                "trained_round": int(round_index),
                "trained_snapshot": True,
                "source_candidate_id": source_id,
                "genome_key": genome_key(candidate.code),
            },
        )
        self.trained_snapshots.append(snapshot)
        self.last_bundles[source_id] = bundle
        self.last_bundles[snapshot.candidate_id] = bundle
        score = bundle.metrics.scalar_score()
        prev = (
            candidate_nav_scalar(self.best_trained)
            if self.best_trained is not None
            else float("-inf")
        )
        if score > prev:
            self.best_trained = snapshot
            logger.info(
                "Stage III new best-ever %s scalar=%.4f (round=%s)",
                snapshot.candidate_id,
                score,
                round_index,
            )
            console.status(
                f"new best-ever {snapshot.candidate_id} "
                f"SR-CR-0.5TR={score:.3f}",
                stage="Stage III",
            )
        return snapshot

    def _skip_refine_for_elite(self, candidate: RewardCandidate) -> bool:
        if not bool(getattr(self.config, "protect_elite_refine", False)):
            return False
        return self.best_trained is not None and is_same_genome(
            candidate, self.best_trained
        )

    def _inject_elite(
        self, population: List[RewardCandidate]
    ) -> List[RewardCandidate]:
        if not bool(getattr(self.config, "inject_elite", False)):
            return population
        elite = self.best_trained
        if elite is None or not population:
            return population
        if any(is_same_genome(c, elite) for c in population):
            return population
        worst_i = min(
            range(len(population)),
            key=lambda i: candidate_nav_scalar(population[i]),
        )
        logger.info(
            "Elitism: injecting best-ever %s (replacing %s)",
            elite.candidate_id,
            population[worst_i].candidate_id,
        )
        population[worst_i] = replace(
            elite,
            metadata={
                **(elite.metadata or {}),
                "elite_injected": True,
                "refine_kept_previous": True,
                "refine_skipped_elite": False,
            },
        )
        return population

    def _repair_invalid_code(
        self,
        candidate_id: str,
        bad_code: str,
        validation_error: str,
        metrics: ProxyMetrics,
    ) -> tuple[Optional[str], Optional[str]]:
        """One D.3 repair attempt (same contract as Stage II)."""
        repair_prompt = format_d3_repair(
            bad_code=bad_code,
            validation_error=validation_error,
        )
        full_prompt = f"{D3_SYSTEM_PROMPT}\n\n{repair_prompt}"
        try:
            raw = self.llm.complete(full_prompt)
            repaired_code = normalize_to_compute_reward(extract_python_code(raw))
            return repaired_code, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def refine_candidate(
        self,
        candidate: RewardCandidate,
        metrics: ProxyMetrics,
    ) -> RewardCandidate:
        feedback = metrics.feedback_text()
        user_prompt = format_d3_refinement(
            candidate.code,
            last_score=metrics.scalar_score(),
            feedback=feedback,
            extra_context_if_any="",
        )
        full_prompt = f"{D3_SYSTEM_PROMPT}\n\n{user_prompt}"
        try:
            raw = self.llm.complete(full_prompt)
            new_code = normalize_to_compute_reward(extract_python_code(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM refinement failed for %s: %s — keeping previous code",
                candidate.candidate_id,
                exc,
            )
            self.validation_failures.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": f"llm_error: {exc}",
                    "kept_previous": True,
                }
            )
            return replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "refine_kept_previous": True,
                    "refine_error": f"llm_error: {exc}",
                },
            )

        reward_fn, err = self.validator.try_validate(new_code)
        if reward_fn is None:
            logger.info(
                "Sandbox rejected refinement for %s: %s — attempting repair",
                candidate.candidate_id,
                err,
            )
            repaired_code, repair_err = self._repair_invalid_code(
                candidate.candidate_id, new_code, err or "unknown", metrics
            )
            if repaired_code is not None:
                reward_fn, repair_validation_err = self.validator.try_validate(
                    repaired_code
                )
                if reward_fn is not None:
                    logger.info(
                        "Repair succeeded for %s: validation passed",
                        candidate.candidate_id,
                    )
                    new_code = repaired_code
                    err = None
                else:
                    logger.warning(
                        "Repair attempted but failed validation for %s: %s",
                        candidate.candidate_id,
                        repair_validation_err,
                    )
            else:
                logger.warning(
                    "Repair LLM call failed for %s: %s",
                    candidate.candidate_id,
                    repair_err,
                )

        if reward_fn is None:
            logger.warning(
                "Sandbox rejected refinement for %s (and repair failed): %s — "
                "keeping previous code",
                candidate.candidate_id,
                err,
            )
            self.validation_failures.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": err,
                    "kept_previous": True,
                    "rejected_code": new_code,
                }
            )
            return replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "refine_kept_previous": True,
                    "refine_error": err,
                },
            )

        # New code is untested — do not attach parent train metrics as its own.
        from crowd_nav.reward_search.proxy_consistency import genome_key

        parent_md = {
            k: v
            for k, v in (candidate.metadata or {}).items()
            if k
            not in (
                "last_metrics",
                "checkpoint_path",
                "trained_snapshot",
                "trained_round",
                "refine_kept_previous",
                "refine_error",
                "refine_skipped_elite",
            )
        }
        return replace(
            candidate,
            candidate_id=_v3_candidate_id(candidate.candidate_id),
            code=new_code,
            reward_fn=reward_fn,
            valid=True,
            validation_error=None,
            origin="refinement",
            parent_ids=(candidate.candidate_id,),
            metadata={
                **parent_md,
                "refine_kept_previous": False,
                "parent_metrics": metrics.as_dict(),
                "parent_id": candidate.candidate_id,
                "parent_genome_key": genome_key(candidate.code),
            },
        )

    def run_round(
        self,
        population: Sequence[RewardCandidate],
        round_index: int,
        *,
        start_index: int = 0,
        partial_next: Optional[Sequence[RewardCandidate]] = None,
        train_progress: Optional[Dict[str, Any]] = None,
    ) -> List[RewardCandidate]:
        if len(population) != self.config.population_size:
            logger.warning(
                "Expected N=%d candidates, got %d",
                self.config.population_size,
                len(population),
            )
        next_pop: List[RewardCandidate] = list(partial_next or [])
        round_records: List[Stage3RoundRecord] = []
        n = len(population)
        for i, cand in enumerate(population):
            if i < int(start_index):
                continue
            console.status(
                f"Round {round_index + 1}/{self.config.rounds} - "
                f"candidate {cand.candidate_id} ({i + 1}/{n})",
                stage="Stage III",
            )
            mid = train_progress if (
                train_progress
                and int(train_progress.get("round_index", -1)) == int(round_index)
                and str(train_progress.get("candidate_id", "")) == str(cand.candidate_id)
            ) else None
            # Clear mid-train hint after first use so later candidates start fresh.
            train_progress = None

            def _on_progress(prog: Dict[str, Any], _i=i, _inp=population, _out=next_pop) -> None:
                self._persist(
                    status="running",
                    phase="training",
                    round_index=round_index,
                    candidate_index=_i,
                    round_input_population=_inp,
                    round_output_partial=_out,
                    train_progress=prog,
                )

            refined, bundle, kept = self._train_refine_one(
                cand,
                round_index=round_index,
                resume_update=int((mid or {}).get("update_j", 0) or 0),
                resume_weights_path=(mid or {}).get("weights_path"),
                progress_callback=_on_progress if bool(getattr(self.config, "resume", True)) else None,
            )
            rec = Stage3RoundRecord(
                round_index=round_index,
                candidate_id=cand.candidate_id,
                metrics=bundle.metrics,
                refined=not kept,
                kept_previous=kept,
                validation_error=refined.metadata.get("refine_error"),
                checkpoint_path=bundle.checkpoint_path,
            )
            self._append_history_record(rec)
            round_records.append(rec)
            self.last_bundles[cand.candidate_id] = bundle
            self.last_bundles[refined.candidate_id] = bundle
            next_pop.append(refined)
            self._persist(
                status="running",
                phase="training",
                round_index=round_index,
                candidate_index=i + 1,
                round_input_population=population,
                round_output_partial=next_pop,
                train_progress=None,
            )
        next_pop = self._inject_elite(next_pop)
        console.stage_round_summary(
            "Stage III", round_index, self.config.rounds, round_records
        )
        return next_pop

    def _train_refine_one(
        self,
        cand: RewardCandidate,
        *,
        round_index: int,
        resume_update: int = 0,
        resume_weights_path: Optional[str] = None,
        progress_callback: Optional[Any] = None,
    ):
        """Train+refine one candidate with optional paper-scale / mid-PPO resume."""
        import time

        from crowd_nav.reward_search.checkpointing import CheckpointKey, CostEvent
        from crowd_nav.reward_search.reporting import candidate_to_dict, load_candidate_dict

        store = self.checkpoint_store
        key = None
        if store is not None:
            key = CheckpointKey(
                seed=int(self.checkpoint_seed),
                stage="stage3",
                round=int(round_index),
                candidate_id=str(cand.candidate_id),
            )
            cached = store.load(key)
            if cached is not None:
                payload = cached.get("payload") or {}
                refined = load_candidate_dict(payload["candidate"])
                md = payload.get("metrics") or {}
                metrics = ProxyMetrics(
                    sr=float(md.get("SR", md.get("sr", 0.0))),
                    cr=float(md.get("CR", md.get("cr", 0.0))),
                    tr=float(md.get("TR", md.get("tr", 0.0))),
                    nt=float(md.get("NT", md.get("nt", 0.0))),
                    pl=float(md.get("PL", md.get("pl", 0.0))),
                    itr=float(md.get("ITR", md.get("itr", 0.0))),
                    sd=float(md.get("SD", md.get("sd", 0.0))),
                )
                bundle = TrainEvalBundle(
                    metrics=metrics,
                    checkpoint_path=payload.get("checkpoint_path"),
                )
                kept = bool(payload.get("kept_previous"))
                logger.info(
                    "Resume Stage III seed=%s round=%s cand=%s",
                    key.seed,
                    key.round,
                    key.candidate_id,
                )
                if store.cost_logger is not None:
                    store.cost_logger.record(
                        CostEvent(
                            seed=key.seed,
                            stage=key.stage,
                            round=key.round,
                            candidate_id=key.candidate_id,
                            wall_seconds=0.0,
                            device=store.device,
                            resumed=True,
                        )
                    )
                self._record_trained_snapshot(
                    cand, bundle, round_index=round_index
                )
                return refined, bundle, kept

        t0 = time.perf_counter()
        bundle = self.trainer.train_and_eval(
            cand,
            round_index=round_index,
            config=self.config,
            resume_update=int(resume_update or 0),
            resume_weights_path=resume_weights_path,
            progress_callback=progress_callback,
        )
        self._record_trained_snapshot(cand, bundle, round_index=round_index)
        if self._skip_refine_for_elite(cand):
            logger.info(
                "Skipping D.3 refine for elite genome %s (protect best-ever)",
                cand.candidate_id,
            )
            refined = replace(
                cand,
                metadata={
                    **(cand.metadata or {}),
                    "refine_kept_previous": True,
                    "refine_skipped_elite": True,
                    "checkpoint_path": bundle.checkpoint_path,
                    "last_metrics": bundle.metrics.as_dict(),
                },
            )
            kept = True
        else:
            refined = self.refine_candidate(cand, bundle.metrics)
            kept = bool(refined.metadata.get("refine_kept_previous"))
            if kept:
                # Kept previous genome — attach this round's train metrics.
                refined = replace(
                    refined,
                    metadata={
                        **refined.metadata,
                        "checkpoint_path": bundle.checkpoint_path,
                        "last_metrics": bundle.metrics.as_dict(),
                    },
                )
            else:
                # New untested genome: keep parent pointers only.
                refined = replace(
                    refined,
                    metadata={
                        **refined.metadata,
                        "parent_checkpoint_path": bundle.checkpoint_path,
                        "parent_metrics": bundle.metrics.as_dict(),
                    },
                )
        wall_s = float(time.perf_counter() - t0)
        if store is not None and key is not None:
            store.save(
                key,
                {
                    "candidate": candidate_to_dict(refined),
                    "metrics": bundle.metrics.as_dict(),
                    "kept_previous": kept,
                    "checkpoint_path": bundle.checkpoint_path,
                },
                wall_seconds=wall_s,
                resumed=False,
            )
        return refined, bundle, kept

    def run_generalization_sweep(
        self,
        population: Sequence[RewardCandidate],
        *,
        human_counts: Optional[Sequence[int]] = None,
    ) -> List[HumanSweepReport]:
        reports: List[HumanSweepReport] = []
        for cand in population:
            bundle = self.last_bundles.get(cand.candidate_id)
            if bundle is None:
                for pid in cand.parent_ids:
                    bundle = self.last_bundles.get(pid)
                    if bundle is not None:
                        break
            if bundle is None:
                # Fall back to metadata checkpoint_path after crash-resume.
                ckpt = (cand.metadata or {}).get("checkpoint_path")
                md = (cand.metadata or {}).get("last_metrics") or {}
                if ckpt and md:
                    bundle = TrainEvalBundle(
                        metrics=ProxyMetrics(
                            sr=float(md.get("SR", md.get("sr", 0.0))),
                            cr=float(md.get("CR", md.get("cr", 0.0))),
                            tr=float(md.get("TR", md.get("tr", 0.0))),
                            nt=float(md.get("NT", md.get("nt", 0.0))),
                            pl=float(md.get("PL", md.get("pl", 0.0))),
                            itr=float(md.get("ITR", md.get("itr", 0.0))),
                            sd=float(md.get("SD", md.get("sd", 0.0))),
                        ),
                        checkpoint_path=str(ckpt),
                    )
            if bundle is None:
                logger.warning(
                    "No train bundle for %s — skipping H-sweep", cand.candidate_id
                )
                continue
            if bundle.actor_critic is None and bundle.checkpoint_path:
                bundle = self._reload_policy_into_bundle(cand, bundle)
                if bundle.actor_critic is None:
                    logger.warning(
                        "Could not load policy for %s — skipping H-sweep",
                        cand.candidate_id,
                    )
                    continue
            report = self.trainer.evaluate_at_human_counts(
                cand, bundle, config=self.config, human_counts=human_counts
            )
            reports.append(report)
            logger.info("H-sweep\n%s", report.summary_table())
        self.sweep_reports = reports
        return reports

    def run(
        self,
        population: Sequence[RewardCandidate],
        *,
        run_h_sweep: bool = True,
    ) -> List[RewardCandidate]:
        from crowd_nav.reward_search.stage3_checkpoint import (
            deserialize_population,
            load_checkpoint,
            load_train_progress,
        )
        from crowd_nav.reward_search.reporting import load_candidate_dict

        console.banner(
            f"Stage III - full PPO refinement (K3={self.config.train_env_steps}, "
            f"paper={STAGE3_PAPER_STEPS})"
        )
        self._run_h_sweep = bool(run_h_sweep)
        self._ckpt_dir = self._resolve_ckpt_dir()
        pop = list(population)
        start_round = 0
        start_index = 0
        partial_next: List[RewardCandidate] = []
        train_progress: Optional[Dict[str, Any]] = None
        resume_phase = "training"

        if bool(getattr(self.config, "resume", True)):
            payload = load_checkpoint(self._ckpt_dir)
            if payload and str(payload.get("status", "")) != "done":
                console.status(
                    f"resuming Stage III from {self._ckpt_dir} "
                    f"(round={payload.get('round_index')} "
                    f"cand_idx={payload.get('candidate_index')} "
                    f"phase={payload.get('phase')})",
                    stage="Stage III",
                )
                self._restore_history_from_payload(payload.get("history") or [])
                if payload.get("best_trained"):
                    self.best_trained = load_candidate_dict(payload["best_trained"])
                snaps = deserialize_population(payload.get("trained_snapshots") or [])
                if snaps:
                    self.trained_snapshots = snaps
                start_round = int(payload.get("round_index", 0))
                start_index = int(payload.get("candidate_index", 0))
                round_in = deserialize_population(
                    payload.get("round_input_population") or []
                )
                if round_in:
                    pop = round_in
                partial_next = deserialize_population(
                    payload.get("round_output_partial") or []
                )
                train_progress = payload.get("train_progress")
                if not train_progress and partial_next is not None:
                    # Also check on-disk mid-PPO file for the next candidate.
                    if start_round < int(self.config.rounds) and start_index < len(pop):
                        nxt = pop[start_index]
                        cand_out = os.path.join(
                            self.config.output_root,
                            f"r{start_round:02d}_{nxt.candidate_id}",
                        )
                        train_progress = load_train_progress(cand_out)
                resume_phase = str(payload.get("phase") or "training")
                if resume_phase == "h_sweep":
                    start_round = int(self.config.rounds)
            elif payload and str(payload.get("status", "")) == "done":
                console.status(
                    "Stage III checkpoint already done — reusing finished population",
                    stage="Stage III",
                )
                finished = deserialize_population(
                    payload.get("round_output_partial")
                    or payload.get("round_input_population")
                    or []
                )
                self._restore_history_from_payload(payload.get("history") or [])
                if payload.get("best_trained"):
                    self.best_trained = load_candidate_dict(payload["best_trained"])
                snaps = deserialize_population(payload.get("trained_snapshots") or [])
                if snaps:
                    self.trained_snapshots = snaps
                if finished:
                    return finished

        for r in range(start_round, self.config.rounds):
            console.status(
                f"starting round {r + 1}/{self.config.rounds} "
                f"(N={len(pop)}, K3={self.config.train_env_steps})",
                stage="Stage III",
            )
            si = start_index if r == start_round else 0
            pn = partial_next if r == start_round else []
            tp = train_progress if r == start_round else None
            pop = self.run_round(
                pop,
                round_index=r,
                start_index=si,
                partial_next=pn,
                train_progress=tp,
            )
            partial_next = []
            start_index = 0
            train_progress = None
            self._persist(
                status="running",
                phase="h_sweep" if (run_h_sweep and r + 1 >= self.config.rounds) else "training",
                round_index=min(r + 1, self.config.rounds),
                candidate_index=0,
                round_input_population=pop,
                round_output_partial=pop,
                train_progress=None,
            )

        if run_h_sweep:
            console.status("running H-sweep generalization...", stage="Stage III")
            self._persist(
                status="running",
                phase="h_sweep",
                round_index=int(self.config.rounds),
                candidate_index=0,
                round_input_population=pop,
                round_output_partial=pop,
            )
            # Prefer best-ever trained policy (matches final selection).
            sweep_pop: List[RewardCandidate]
            if self.best_trained is not None:
                sweep_pop = [self.best_trained]
            else:
                sweep_pop = pop
            self.run_generalization_sweep(sweep_pop)

        self._persist(
            status="done",
            phase="done",
            round_index=int(self.config.rounds),
            candidate_index=0,
            round_input_population=pop,
            round_output_partial=pop,
            train_progress=None,
        )
        return pop

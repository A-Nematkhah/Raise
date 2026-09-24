"""
RAISE Stage II — lightweight proxy A2C rollouts + D.3 LLM refinement.

For each of N Stage-I candidates, each of G2 rounds:
  1. Train a FRESH A2C policy for fixed K2 env steps (Table 5: 8000).
  2. Evaluate E2 episodes with horizon T_short (Table 5: 50 eps, 100 steps).
  3. Collect SR, CR, TR, NT, PL, ITR, SD (ITR/SD from Danger / dmin tracking).
  4. Ask the LLM for a D.3 refinement; sandbox-validate; keep prior code if
     the revision fails validation (logged, never silently dropped).

Training mirrors ``train.py`` (Policy / RolloutStorage / make_vec_envs /
agent.update) with ``--algo a2c`` and reward injection via
``make_vec_envs(..., reward_fn=...)``. ``train.py`` itself is not modified.
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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
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
from crowd_nav.reward_search.proxy_consistency import genome_key
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
from crowd_nav.reward_search.state import RewardFunction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config / metrics (Table 5)
# ---------------------------------------------------------------------------


@dataclass
class Stage2Config:
    """RAISE Table 5 Stage II defaults."""

    population_size: int = 8
    rounds: int = 16  # G2
    # When k2_unit=env_steps: env interaction steps. When gradient_steps: K2 updates.
    # Paper §4.3.2 default: K2=8000 gradient steps.
    train_env_steps: int = 8_000
    k2_unit: str = "gradient_steps"  # env_steps | gradient_steps
    eval_episodes: int = 50  # E2
    horizon_steps: int = 100  # T_short
    algo: str = "a2c"
    # None → resolve_num_processes() at train time (min(16, cpu_count-1)).
    num_processes: Optional[int] = None
    num_steps: int = 5  # A2C n-step (arguments.py help text default)
    # None → default_num_mini_batch(resolved num_processes).
    num_mini_batch: Optional[int] = None
    seed: int = 425
    # Choice (a): GST-inferred obs parity with CrowdNav++ (AUDIT.md §8.2).
    env_name: str = "CrowdSimPredRealGST-v0"
    predict_method: str = "inferred"
    human_num: int = 20
    randomization_regime: str = "without_random"
    output_root: str = "trained_models/stage2"
    device: str = "cpu"
    # Alg. 1 has no elitism — defaults off; opt in via pipeline ``elitism=True``.
    protect_elite_refine: bool = False
    inject_elite: bool = False


@dataclass
class ProxyMetrics:
    """Multi-objective tuple M(r) collected after proxy evaluation."""

    sr: float = 0.0
    cr: float = 0.0
    tr: float = 0.0
    nt: float = 0.0
    pl: float = 0.0
    itr: float = 0.0  # mean intrusion time ratio (%), same as rl.evaluation
    sd: float = 0.0  # mean min distance during Danger frames

    def as_dict(self) -> Dict[str, float]:
        return {
            "SR": float(self.sr),
            "CR": float(self.cr),
            "TR": float(self.tr),
            "NT": float(self.nt),
            "PL": float(self.pl),
            "ITR": float(self.itr),
            "SD": float(self.sd),
        }

    def feedback_text(self) -> str:
        """Raw metric values for D.3 (not rankings)."""
        d = self.as_dict()
        return (
            f"SR={d['SR']:.4f}, CR={d['CR']:.4f}, TR={d['TR']:.4f}, "
            f"NT={d['NT']:.4f}, PL={d['PL']:.4f}, ITR={d['ITR']:.4f}, SD={d['SD']:.4f}"
        )

    def scalar_score(self) -> float:
        """Simple scalar for D.3 last_score field (higher better)."""
        return float(self.sr - self.cr - 0.5 * self.tr)


@dataclass
class Stage2RoundRecord:
    round_index: int
    candidate_id: str
    metrics: ProxyMetrics
    refined: bool
    kept_previous: bool
    validation_error: Optional[str] = None
    checkpoint_path: Optional[str] = None


def _stable_code_hash(code: str) -> int:
    return int(hashlib.md5(code.encode("utf-8")).hexdigest(), 16) % 10_000


def _next_revision_id(candidate_id: str, *, stage_tag: str = "v2") -> str:
    """
    Tag a successful refine as ``*_v{n}`` with a monotonically increasing n.

    Previously always forced ``_v2``, so later refinements of the same lineage
    collided on one id and broke ``pick_candidate_by_ranking``.
    """
    m = re.search(r"_v(\d+)$", candidate_id)
    if m:
        return f"{candidate_id[: m.start()]}_v{int(m.group(1)) + 1}"
    # Prefer stage-default first suffix (v2 for Stage II, v3 for Stage III).
    n = 2 if stage_tag == "v2" else 3 if stage_tag == "v3" else 2
    return f"{candidate_id}_v{n}"


def _v2_candidate_id(candidate_id: str) -> str:
    """Tag a successfully refined Stage-II revision (incrementing ``_vN``)."""
    return _next_revision_id(candidate_id, stage_tag="v2")


def _unique_trained_snapshot_id(candidate_id: str, *, round_index: int, snap_index: int) -> str:
    """Disambiguate trained snapshots that share a genome id across rounds."""
    return f"{candidate_id}__r{int(round_index)}_s{int(snap_index)}"


# ---------------------------------------------------------------------------
# Trainer interface (real vs stub)
# ---------------------------------------------------------------------------


class PolicyTrainer(ABC):
    """Train + evaluate a proxy policy under a candidate reward."""

    @abstractmethod
    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage2Config,
    ) -> ProxyMetrics:
        raise NotImplementedError


class StubPolicyTrainer(PolicyTrainer):
    """
    Deterministic no-op trainer for pytest (seconds, not GPU-hours).

    Metrics depend only on a stable hash of ``candidate.code`` so ranking
    is reproducible across rounds unless the code changes.
    """

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage2Config,
    ) -> ProxyMetrics:
        base = (_stable_code_hash(candidate.code) % 100) / 100.0
        jitter = 0.01 * (round_index % 5)
        sr = min(1.0, max(0.0, 0.4 + 0.5 * base + jitter))
        cr = min(1.0, max(0.0, 0.4 * (1.0 - base)))
        tr = max(0.0, 1.0 - sr - cr)
        return ProxyMetrics(
            sr=sr,
            cr=cr,
            tr=tr,
            nt=10.0 + 5.0 * (1.0 - base),
            pl=12.0 + 8.0 * (1.0 - base),
            itr=5.0 + 20.0 * (1.0 - base),  # percent, like rl.evaluation
            sd=0.2 + 0.3 * base,
        )


def resolve_stage2_env_steps(config: Stage2Config, num_processes: int) -> int:
    """
    Map configured K2 budget to env interaction steps.

    - ``env_steps``: ``train_env_steps`` is already env interactions.
    - ``gradient_steps`` (paper §4.3.2 default): ``train_env_steps`` holds K2
      *updates*; env steps = K2 * num_steps * num_processes so that
      ``num_updates = env_steps // num_steps // num_processes == K2``.
    """
    unit = str(getattr(config, "k2_unit", "gradient_steps")).strip().lower()
    k2 = max(1, int(config.train_env_steps))
    nproc = max(1, int(num_processes))
    nsteps = max(1, int(config.num_steps))
    if unit in ("env_steps", "env", "environment"):
        return k2
    if unit in ("gradient_steps", "gradient", "updates"):
        return max(k2 * nsteps * nproc, k2)
    raise ValueError(
        f"Unknown k2_unit={config.k2_unit!r}; use env_steps or gradient_steps"
    )


def _stage2_train_argv(
    config: Stage2Config, candidate_id: str, round_index: int
) -> list:
    out_dir = os.path.join(
        config.output_root, f"r{round_index:02d}_{candidate_id}"
    )
    nproc = resolve_num_processes(config.num_processes)
    env_steps = resolve_stage2_env_steps(config, nproc)
    # A2C.update() always feeds a single mini-batch; SRNN uses
    # nenv // num_mini_batch, so these must match (same as train.py).
    if str(config.algo).lower() == "a2c":
        nbatch = 1
    elif config.num_mini_batch is not None:
        nbatch = max(1, int(config.num_mini_batch))
    else:
        nbatch = default_num_mini_batch(nproc)
    if nbatch > nproc:
        nbatch = nproc
    argv = [
        "stage2_train",
        "--algo",
        config.algo,
        "--num-env-steps",
        str(int(env_steps)),
        "--num-processes",
        str(nproc),
        "--num-steps",
        str(config.num_steps),
        "--seq_length",
        str(config.num_steps),
        "--num-mini-batch",
        str(nbatch),
        "--seed",
        str(config.seed + round_index),
        "--env-name",
        config.env_name,
        "--output_dir",
        out_dir,
        "--overwrite",
    ]
    # Policy code checks ``args.no_cuda`` (not ``args.cuda``) for tensor placement.
    if config.device != "cuda" or not torch.cuda.is_available():
        if config.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "Stage II: --device cuda requested but PyTorch has no CUDA "
                "(version=%s); forcing --no-cuda.",
                torch.__version__,
            )
        argv.append("--no-cuda")
    return argv


def _parse_stage2_algo_args(config: Stage2Config, candidate_id: str, round_index: int):
    """
    Call ``get_args()`` / import ``Config`` under a controlled argv.

    ``crowd_nav.configs.config.Config`` evaluates ``get_args()`` at class-body
    time, so the first import must not see caller CLI flags (e.g. smoke script).
    """
    from arguments import get_args

    argv = _stage2_train_argv(config, candidate_id, round_index)
    saved = list(sys.argv)
    try:
        sys.argv = argv
        algo_args = get_args()
        # Import under the same argv so Config.args matches algo_args.
        from crowd_nav.configs.config import Config  # noqa: F401
        return algo_args
    finally:
        sys.argv = saved


def _make_proxy_env_config(config: Stage2Config):
    """
    Stage II env Config: short horizon + regime randomization + GST (choice a).

    ``Config.env`` is a shared class-level object. We shallow-copy it onto the
    instance before mutating ``time_limit`` so Stage III still sees the paper
    default (50s) in the same process.
    """
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
        entry_point="stage2",
    )
    # Keep env_name consistent with predict_method (GST vs VarNum).
    expected = env_name_for_predict_method(config.predict_method)
    if config.env_name != expected:
        logger.warning(
            "Stage2Config.env_name=%s overridden to %s for predict_method=%s",
            config.env_name,
            expected,
            config.predict_method,
        )
        config.env_name = expected
    cfg.robot.policy = "selfAttn_merge_srnn"
    # apply_regime_to_config already detaches namespaces; keep env/sim on the
    # instance dict so Windows spawn workers see the same human_num / horizon.
    if "env" not in cfg.__dict__:
        cfg.env = copy(cfg.env)
    if "sim" not in cfg.__dict__:
        cfg.sim = copy(cfg.sim)
    cfg.env.time_limit = float(config.horizon_steps) * float(cfg.env.time_step)
    cfg.env.test_size = int(config.eval_episodes)
    cfg.sim.human_num = max(1, int(config.human_num))
    cfg.sim.human_num_range = 0
    return cfg


class RealPolicyTrainer(PolicyTrainer):
    """
    Fresh A2C (or PPO) training for fixed K2 steps + short-horizon evaluation.

    Thin wrapper around the same building blocks as ``train.py``; injects
    ``candidate.reward_fn`` into envs. No adaptive early-stopping.
    """

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage2Config,
    ) -> ProxyMetrics:
        if candidate.reward_fn is None:
            raise ValueError(f"Candidate {candidate.candidate_id} has no reward_fn")

        from rl.a2c import A2C
        from rl import ppo
        from rl.networks.envs import make_vec_envs
        from rl.networks.model import Policy
        from rl.networks.storage import RolloutStorage

        algo_args = _parse_stage2_algo_args(
            config, candidate.candidate_id, round_index
        )
        env_config = _make_proxy_env_config(config)

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

        if config.algo.lower() == "a2c":
            agent = A2C(
                actor_critic,
                algo_args.value_loss_coef,
                algo_args.entropy_coef,
                lr=algo_args.lr,
                eps=algo_args.eps,
                alpha=algo_args.alpha,
                max_grad_norm=algo_args.max_grad_norm,
            )
        else:
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

        # Fixed schedule — no early stopping (paper Table 5).
        num_updates = (
            int(algo_args.num_env_steps)
            // algo_args.num_steps
            // algo_args.num_processes
        )
        console.status(
            f"training {candidate.candidate_id} round={round_index} "
            f"algo={config.algo} K2={config.train_env_steps} "
            f"k2_unit={getattr(config, 'k2_unit', 'gradient_steps')} "
            f"env_steps={algo_args.num_env_steps} "
            f"updates={num_updates} nproc={algo_args.num_processes}",
            stage="Stage II",
        )
        pbar = console.progress(
            total=max(1, num_updates),
            desc=f"[Stage II] {candidate.candidate_id} A2C",
            unit="upd",
        )
        t_train = time.perf_counter()
        try:
            for _j in range(num_updates):
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
        except Exception as exc:  # noqa: BLE001
            console.fail(
                f"train crashed for {candidate.candidate_id} round={round_index}: {exc}",
                stage="Stage II",
            )
            raise
        finally:
            pbar.close()
        console.status(
            f"train done for {candidate.candidate_id} in "
            f"{console.format_seconds(time.perf_counter() - t_train)}; evaluating...",
            stage="Stage II",
        )

        ckpt_path = os.path.join(ckpt_dir, f"{max(num_updates - 1, 0):05d}.pt")
        torch.save(actor_critic.state_dict(), ckpt_path)
        envs.close()

        metrics = evaluate_proxy_policy(
            actor_critic,
            candidate.reward_fn,
            algo_args,
            env_config,
            device,
            n_episodes=config.eval_episodes,
            horizon_steps=config.horizon_steps,
        )
        with open(os.path.join(out_dir, "proxy_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(metrics.feedback_text() + "\n")
            f.write(f"checkpoint={ckpt_path}\n")
        console.status(
            f"eval {candidate.candidate_id}: {metrics.feedback_text()}",
            stage="Stage II",
        )
        return metrics


def pin_episode_human_count(base_env, human_num: int) -> None:
    """
    Force each episode to spawn exactly ``human_num`` humans while keeping
    ``max_human_num`` (and thus Policy obs dims) unchanged.
    """
    if human_num < 1 or human_num > int(base_env.max_human_num):
        raise ValueError(
            f"human_num={human_num} out of range [1, {base_env.max_human_num}]"
        )
    # Allow H < training population without shrinking obs / max_human_num.
    base_env.min_human_num = 1
    orig = base_env.generate_robot_humans

    def _forced(phase, human_num_arg=None):
        old_h = base_env.config.sim.human_num
        old_r = base_env.human_num_range
        base_env.config.sim.human_num = int(human_num)
        base_env.human_num_range = 0
        try:
            return orig(phase, human_num=human_num)
        finally:
            base_env.config.sim.human_num = old_h
            base_env.human_num_range = old_r

    base_env.generate_robot_humans = _forced


def evaluate_proxy_policy(
    actor_critic,
    reward_fn: RewardFunction,
    algo_args,
    env_config,
    device,
    *,
    n_episodes: int,
    horizon_steps: int,
    human_num: Optional[int] = None,
) -> ProxyMetrics:
    """
    Evaluate like ``rl.evaluation.evaluate``, returning metrics.

    Caps each episode at ``horizon_steps`` (T_short). ITR/SD reuse Danger /
    ``min_dist`` tracking from the env infos (same as test.py path).

    If ``human_num`` is set, pin episode population to that count without
    shrinking ``max_human_num`` (needed for Table 6 H-sweeps).
    """
    from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout
    from rl.networks.envs import make_vec_envs

    # Eval always uses a single env; Policy was trained with N processes.
    # Match test.py: overwrite nenv so infer-time reshapeT matches obs batch.
    if hasattr(actor_critic, "base") and hasattr(actor_critic.base, "nenv"):
        actor_critic.base.nenv = 1

    envs = make_vec_envs(
        algo_args.env_name,
        algo_args.seed,
        1,
        algo_args.gamma,
        None,
        device,
        True,
        config=env_config,
        pretext_wrapper=env_config.env.use_wrapper,
        reward_fn=reward_fn,
    )

    if hasattr(envs.venv, "envs"):
        base_env = envs.venv.envs[0].env
    else:
        base_env = envs.venv.unwrapped.envs[0].env

    if human_num is not None:
        pin_episode_human_count(base_env, int(human_num))

    eval_recurrent_hidden_states = {
        "human_node_rnn": torch.zeros(
            1, 1, actor_critic.base.human_node_rnn_size, device=device
        ),
        "human_human_edge_rnn": torch.zeros(
            1,
            actor_critic.base.human_num + 1,
            actor_critic.base.human_human_edge_rnn_size,
            device=device,
        ),
    }
    eval_masks = torch.zeros(1, 1, device=device)

    success = collision = timeout = 0
    success_times: List[float] = []
    all_path_len: List[float] = []
    too_close_ratios: List[float] = []
    min_dist: List[float] = []
    time_limit = float(base_env.time_limit)

    for _k in range(n_episodes):
        done = False
        step_counter = 0
        obs = envs.reset()
        global_time = 0.0
        path_len = 0.0
        too_close = 0.0
        last_pos = obs["robot_node"][0, 0, :2].cpu().numpy()
        infos = [{"info": None}]

        while not done and step_counter < horizon_steps:
            step_counter += 1
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs, eval_recurrent_hidden_states, eval_masks, deterministic=True
                )
            if not done:
                global_time = float(base_env.global_time)
            obs, _rew, done, infos = envs.step(action)
            path_len += float(
                np.linalg.norm(obs["robot_node"][0, 0, :2].cpu().numpy() - last_pos)
            )
            last_pos = obs["robot_node"][0, 0, :2].cpu().numpy()
            if isinstance(infos[0]["info"], Danger):
                too_close += 1
                min_dist.append(float(infos[0]["info"].min_dist))
            eval_masks = torch.tensor(
                [[0.0] if d else [1.0] for d in done],
                dtype=torch.float32,
                device=device,
            )

        all_path_len.append(path_len)
        # Match rl.evaluation: intrusion ratio as percent of steps.
        too_close_ratios.append(too_close / max(step_counter, 1) * 100.0)

        info = infos[0]["info"]
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(global_time)
        elif isinstance(info, Collision):
            collision += 1
        elif isinstance(info, Timeout) or step_counter >= horizon_steps or not done:
            timeout += 1
        else:
            timeout += 1

    envs.close()
    n = float(n_episodes)
    return ProxyMetrics(
        sr=success / n,
        cr=collision / n,
        tr=timeout / n,
        nt=(sum(success_times) / len(success_times)) if success_times else time_limit,
        pl=float(np.mean(all_path_len)) if all_path_len else 0.0,
        itr=float(np.mean(too_close_ratios)) if too_close_ratios else 0.0,
        sd=float(np.mean(min_dist)) if min_dist else 0.0,
    )


# ---------------------------------------------------------------------------
# Stage II runner
# ---------------------------------------------------------------------------


class Stage2Runner:
    """G2 rounds of proxy train/eval + Appendix D.3 LLM refinement."""

    def __init__(
        self,
        llm: LLMClient,
        trainer: PolicyTrainer,
        *,
        validator: Optional[RewardValidator] = None,
        config: Optional[Stage2Config] = None,
        prompts: Any = None,
    ) -> None:
        self.llm = llm
        self.trainer = trainer
        self.validator = validator or RewardValidator()
        self.config = config or Stage2Config()
        self._prompts = prompts
        self.history: List[Stage2RoundRecord] = []
        self.validation_failures: List[Dict[str, Any]] = []
        # Trained (pre-refine) snapshots; best_trained is best-ever by SR-CR-0.5TR.
        self.trained_snapshots: List[RewardCandidate] = []
        self.best_trained: Optional[RewardCandidate] = None
        # Optional paper-scale resume (set by PaperScaleRunner).
        self.checkpoint_store = None  # type: ignore[assignment]
        self.checkpoint_seed: int = int(self.config.seed)

    def _d3_system_prompt(self) -> str:
        if self._prompts is not None:
            return str(getattr(self._prompts, "D3_SYSTEM_PROMPT", D3_SYSTEM_PROMPT))
        return D3_SYSTEM_PROMPT

    def _format_d3_refinement(self, *args: Any, **kwargs: Any) -> str:
        fn = (
            getattr(self._prompts, "format_d3_refinement", None)
            if self._prompts is not None
            else None
        )
        return (fn or format_d3_refinement)(*args, **kwargs)

    def _format_d3_repair(self, **kwargs: Any) -> str:
        fn = (
            getattr(self._prompts, "format_d3_repair", None)
            if self._prompts is not None
            else None
        )
        return (fn or format_d3_repair)(**kwargs)

    def _record_trained_snapshot(
        self,
        candidate: RewardCandidate,
        metrics: ProxyMetrics,
        *,
        round_index: int,
        checkpoint_path: Optional[str] = None,
    ) -> RewardCandidate:
        """Freeze the genome that produced ``metrics`` (before any LLM refine)."""
        source_id = str(candidate.candidate_id)
        snap_index = len(self.trained_snapshots)
        snapshot = replace(
            candidate,
            candidate_id=_unique_trained_snapshot_id(
                source_id, round_index=round_index, snap_index=snap_index
            ),
            metadata={
                **(candidate.metadata or {}),
                "last_metrics": metrics.as_dict(),
                "checkpoint_path": checkpoint_path,
                "trained_round": int(round_index),
                "trained_snapshot": True,
                "source_candidate_id": source_id,
                "genome_key": genome_key(candidate.code),
            },
        )
        self.trained_snapshots.append(snapshot)
        score = metrics.scalar_score()
        prev = (
            candidate_nav_scalar(self.best_trained)
            if self.best_trained is not None
            else float("-inf")
        )
        if score > prev:
            self.best_trained = snapshot
            logger.info(
                "Stage II new best-ever %s scalar=%.4f (round=%s)",
                snapshot.candidate_id,
                score,
                round_index,
            )
            console.status(
                f"new best-ever {snapshot.candidate_id} "
                f"SR-CR-0.5TR={score:.3f}",
                stage="Stage II",
            )
        return snapshot

    def _skip_refine_for_elite(self, candidate: RewardCandidate) -> bool:
        """Do not LLM-mutate the current best-ever genome."""
        if not bool(getattr(self.config, "protect_elite_refine", False)):
            return False
        return self.best_trained is not None and is_same_genome(
            candidate, self.best_trained
        )

    def _repair_invalid_code(
        self,
        candidate_id: str,
        bad_code: str,
        validation_error: str,
        metrics: ProxyMetrics,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Single repair attempt: feed back the exact error and ask LLM to fix.
        Returns (repaired_code, error) or (None, error_reason) if repair fails.
        """
        repair_prompt = self._format_d3_repair(
            bad_code=bad_code,
            validation_error=validation_error,
        )
        full_prompt = f"{self._d3_system_prompt()}\n\n{repair_prompt}"
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
        """
        D.3 refinement → ``*_v2``. On sandbox failure, attempt ONE repair before keeping previous.
        """
        feedback = metrics.feedback_text()
        user_prompt = self._format_d3_refinement(
            candidate.code,
            last_score=metrics.scalar_score(),
            feedback=feedback,
            extra_context_if_any="",
        )
        full_prompt = f"{self._d3_system_prompt()}\n\n{user_prompt}"
        try:
            raw = self.llm.complete(full_prompt)
            new_code = normalize_to_compute_reward(extract_python_code(raw))
        except Exception as exc:  # noqa: BLE001
            console.fail(
                f"LLM refinement error for {candidate.candidate_id}: {exc}",
                stage="Stage II",
            )
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
            # Attempt repair once before giving up
            logger.info(
                "Sandbox rejected refinement for %s: %s — attempting repair",
                candidate.candidate_id,
                err,
            )
            repaired_code, repair_err = self._repair_invalid_code(
                candidate.candidate_id, new_code, err, metrics
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
                    return replace(
                        candidate,
                        candidate_id=_v2_candidate_id(candidate.candidate_id),
                        code=repaired_code,
                        reward_fn=reward_fn,
                        valid=True,
                        validation_error=None,
                        origin="refinement",
                        parent_ids=(candidate.candidate_id,),
                        metadata={
                            **candidate.metadata,
                            "refine_kept_previous": False,
                            "last_metrics": metrics.as_dict(),
                            "repair_attempted": True,
                            "repair_succeeded": True,
                            "parent_genome_key": genome_key(candidate.code),
                            "parent_id": candidate.candidate_id,
                        },
                    )
                else:
                    logger.warning(
                        "Repair attempted but failed validation for %s: %s",
                        candidate.candidate_id,
                        repair_validation_err,
                    )
            elif repair_err is not None:
                logger.warning(
                    "Repair LLM call failed for %s: %s",
                    candidate.candidate_id,
                    repair_err,
                )

            logger.warning(
                "Sandbox rejected refinement for %s (and repair failed): %s — keeping previous code",
                candidate.candidate_id,
                err,
            )
            self.validation_failures.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": err,
                    "kept_previous": True,
                    "rejected_code": new_code,
                    "repair_attempted": True,
                }
            )
            return replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "refine_kept_previous": True,
                    "refine_error": err,
                    "repair_attempted": True,
                    "repair_succeeded": False,
                },
            )

        # New code is untested — do not attach parent train metrics as its own.
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
            candidate_id=_v2_candidate_id(candidate.candidate_id),
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
    ) -> List[RewardCandidate]:
        """One G2 round: train/eval all, then refine each via D.3."""
        if len(population) != self.config.population_size:
            logger.warning(
                "Expected N=%d candidates, got %d",
                self.config.population_size,
                len(population),
            )
        next_pop: List[RewardCandidate] = []
        round_records: List[Stage2RoundRecord] = []
        n = len(population)
        for i, cand in enumerate(population, start=1):
            console.status(
                f"Round {round_index + 1}/{self.config.rounds} - "
                f"candidate {cand.candidate_id} ({i}/{n})",
                stage="Stage II",
            )
            refined, metrics, kept, wall_s, resumed = self._train_refine_one(
                cand, round_index=round_index
            )
            rec = Stage2RoundRecord(
                round_index=round_index,
                candidate_id=cand.candidate_id,
                metrics=metrics,
                refined=not kept,
                kept_previous=kept,
                validation_error=refined.metadata.get("refine_error"),
            )
            self.history.append(rec)
            round_records.append(rec)
            if wall_s is not None:
                refined.metadata = {
                    **(refined.metadata or {}),
                    "wall_seconds": wall_s,
                    "resumed": resumed,
                }
            next_pop.append(refined)
        next_pop = self._inject_elite(next_pop)
        console.stage_round_summary(
            "Stage II", round_index, self.config.rounds, round_records
        )
        return next_pop

    def _inject_elite(
        self, population: List[RewardCandidate]
    ) -> List[RewardCandidate]:
        """Keep best-ever genome in the next round population (elitism)."""
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
        # Carry elite code forward; drop stale refine flags for a clean retrain.
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

    def _train_refine_one(
        self,
        cand: RewardCandidate,
        *,
        round_index: int,
    ):
        """Train+refine one candidate, with optional (seed, stage, round, id) resume."""
        import time

        from crowd_nav.reward_search.checkpointing import CheckpointKey
        from crowd_nav.reward_search.reporting import candidate_to_dict, load_candidate_dict

        store = self.checkpoint_store
        key = None
        if store is not None:
            key = CheckpointKey(
                seed=int(self.checkpoint_seed),
                stage="stage2",
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
                kept = bool(payload.get("kept_previous"))
                logger.info(
                    "Resume Stage II seed=%s round=%s cand=%s",
                    key.seed,
                    key.round,
                    key.candidate_id,
                )
                if store.cost_logger is not None:
                    from crowd_nav.reward_search.checkpointing import CostEvent

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
                    cand, metrics, round_index=round_index
                )
                return refined, metrics, kept, 0.0, True

        t0 = time.perf_counter()
        metrics = self.trainer.train_and_eval(
            cand, round_index=round_index, config=self.config
        )
        self._record_trained_snapshot(cand, metrics, round_index=round_index)
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
                    "last_metrics": metrics.as_dict(),
                },
            )
            kept = True
        else:
            refined = self.refine_candidate(cand, metrics)
            kept = bool(refined.metadata.get("refine_kept_previous"))
        wall_s = float(time.perf_counter() - t0)
        if store is not None and key is not None:
            store.save(
                key,
                {
                    "candidate": candidate_to_dict(refined),
                    "metrics": metrics.as_dict(),
                    "kept_previous": kept,
                },
                wall_seconds=wall_s,
                resumed=False,
            )
        return refined, metrics, kept, wall_s, False

    def run(self, population: Sequence[RewardCandidate]) -> List[RewardCandidate]:
        """Run G2 refinement rounds; returns final population."""
        console.banner("Stage II - proxy A2C refinement")
        pop = list(population)
        for r in range(self.config.rounds):
            console.status(
                f"starting round {r + 1}/{self.config.rounds} "
                f"(N={len(pop)}, K2={self.config.train_env_steps})",
                stage="Stage II",
            )
            pop = self.run_round(pop, round_index=r)
        return pop

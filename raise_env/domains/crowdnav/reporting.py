"""
Shared evaluation + JSON reporting for RAISE Table 1 / Table 2.

Returns raw per-episode records (not just aggregates) so later AMFRS-style
analyses can reuse the same baseline JSON without re-running.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from raise_core.refine import ProxyMetrics, pin_episode_human_count

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    """One evaluation episode (raw, for JSON archival)."""

    episode_index: int
    seed: int
    outcome: str  # success | collision | timeout | other
    success: int
    collision: int
    timeout: int
    nav_time: float
    path_length: float
    intrusion_ratio_pct: float
    min_dist_during_intrusion: Optional[float]
    steps: int
    human_num: Optional[int] = None
    method: str = ""
    randomize: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricSummary:
    mean: float
    std: float
    n: int

    def as_dict(self) -> Dict[str, float]:
        return {"mean": float(self.mean), "std": float(self.std), "n": int(self.n)}

    def format_pct(self) -> str:
        return f"{100.0 * self.mean:.2f}+/-{100.0 * self.std:.2f}%"

    def format_raw(self) -> str:
        return f"{self.mean:.2f}+/-{self.std:.2f}"


@dataclass
class EvalBundle:
    """Per-method evaluation: episodes + aggregates (paper Table format)."""

    method: str
    randomize: bool
    episodes: List[EpisodeRecord]
    sr: MetricSummary
    cr: MetricSummary
    tr: MetricSummary
    nt: MetricSummary
    pl: MetricSummary
    itr: MetricSummary
    sd: MetricSummary
    metadata: Dict[str, Any] = field(default_factory=dict)

    def aggregate_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "randomize": self.randomize,
            "SR": self.sr.as_dict(),
            "CR": self.cr.as_dict(),
            "TR": self.tr.as_dict(),
            "NT": self.nt.as_dict(),
            "PL": self.pl.as_dict(),
            "ITR": self.itr.as_dict(),
            "SD": self.sd.as_dict(),
            "n_episodes": len(self.episodes),
            "metadata": self.metadata,
            "format": {
                "SR": self.sr.format_pct(),
                "CR": self.cr.format_pct(),
                "TR": self.tr.format_pct(),
                "NT": self.nt.format_raw(),
                "PL": self.pl.format_raw(),
                "ITR": self.itr.format_raw(),
                "SD": self.sd.format_raw(),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.aggregate_dict(),
            "episodes": [e.to_dict() for e in self.episodes],
        }


def _summary(values: Sequence[float]) -> MetricSummary:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return MetricSummary(mean=0.0, std=0.0, n=0)
    if arr.size == 1:
        return MetricSummary(mean=float(arr[0]), std=0.0, n=1)
    return MetricSummary(mean=float(arr.mean()), std=float(arr.std(ddof=1)), n=int(arr.size))


def summarize_episodes(
    episodes: Sequence[EpisodeRecord],
    *,
    method: str,
    randomize: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvalBundle:
    """
    Paper-style mean±std.

    Rates (SR/CR/TR) are summarized across evaluation seeds when multiple
    seeds are present (group by seed first); otherwise across episodes.
    Continuous metrics use successful-episode NT and all-episode PL/ITR/SD.
    """
    by_seed: Dict[int, List[EpisodeRecord]] = {}
    for ep in episodes:
        by_seed.setdefault(ep.seed, []).append(ep)

    sr_vals: List[float] = []
    cr_vals: List[float] = []
    tr_vals: List[float] = []
    for seed_eps in by_seed.values():
        n = max(len(seed_eps), 1)
        sr_vals.append(sum(e.success for e in seed_eps) / n)
        cr_vals.append(sum(e.collision for e in seed_eps) / n)
        tr_vals.append(sum(e.timeout for e in seed_eps) / n)

    nt_vals = [e.nav_time for e in episodes if e.success]
    pl_vals = [e.path_length for e in episodes]
    itr_vals = [e.intrusion_ratio_pct for e in episodes]
    sd_vals = [
        e.min_dist_during_intrusion
        for e in episodes
        if e.min_dist_during_intrusion is not None
    ]

    return EvalBundle(
        method=method,
        randomize=randomize,
        episodes=list(episodes),
        sr=_summary(sr_vals),
        cr=_summary(cr_vals),
        tr=_summary(tr_vals),
        nt=_summary(nt_vals if nt_vals else [0.0]),
        pl=_summary(pl_vals),
        itr=_summary(itr_vals),
        sd=_summary(sd_vals if sd_vals else [0.0]),
        metadata=dict(metadata or {}),
    )


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def candidate_to_dict(cand) -> Dict[str, Any]:
    return {
        "candidate_id": cand.candidate_id,
        "code": cand.code,
        "score": cand.score,
        "valid": cand.valid,
        "origin": cand.origin,
        "parent_ids": list(cand.parent_ids),
        "validation_error": cand.validation_error,
        "metadata": dict(cand.metadata or {}),
    }


def load_candidate_dict(d: Dict[str, Any]):
    from raise_core.explore import RewardCandidate
    from raise_core.sandbox import RewardValidator

    code = d["code"]
    reward_fn, err = RewardValidator().try_validate(code)
    return RewardCandidate(
        candidate_id=d.get("candidate_id", "loaded"),
        code=code,
        reward_fn=reward_fn,
        score=d.get("score"),
        valid=reward_fn is not None,
        origin=d.get("origin", "loaded"),
        parent_ids=tuple(d.get("parent_ids") or ()),
        validation_error=err or d.get("validation_error"),
        metadata=dict(d.get("metadata") or {}),
    )


def _import_model_config(model_dir: str):
    """Load Config + get_args from a saved model directory (like test.py)."""
    from importlib import import_module

    model_dir_temp = model_dir.rstrip("/\\")
    saved = list(sys.argv)
    try:
        sys.argv = [sys.argv[0], "--no-cuda", "--num-processes", "1"]
        try:
            mod = import_module(model_dir_temp.replace("/", ".").replace("\\", ".") + ".arguments")
            get_args = getattr(mod, "get_args")
        except Exception:  # noqa: BLE001
            from arguments import get_args

        algo_args = get_args()
        try:
            mod = import_module(model_dir_temp.replace("/", ".").replace("\\", ".") + ".configs.config")
            Config = getattr(mod, "Config")
        except Exception:  # noqa: BLE001
            from crowd_nav.configs.config import Config

        env_config = Config()
        return algo_args, env_config
    finally:
        sys.argv = saved


def evaluate_saved_model(
    model_dir: str,
    *,
    checkpoint: str = "00000.pt",
    n_episodes: int = 500,
    seed: int = 425,
    method: str = "",
    randomize: Optional[bool] = None,
    device: str = "cpu",
    human_num: Optional[int] = None,
) -> EvalBundle:
    """
    Evaluate a checkpoint / ORCA / SF policy from ``trained_models/...``.

    Matches ``test.py`` + ``rl.evaluation.evaluate``, but returns raw episodes.
    """
    from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout
    from rl.networks.envs import make_vec_envs
    from rl.networks.model import Policy

    algo_args, env_config = _import_model_config(model_dir)
    # Some shipped GST_* configs have predict_method accidentally set to
    # 'none' / wrong env_name; checkpoints still expect the GST wrapper
    # (spatial edge dim 12). Force the paper CrowdNav++ evaluation setup.
    if "GST" in model_dir.replace("\\", "/"):
        env_config.sim.predict_method = "inferred"
        env_config.env.use_wrapper = True
        algo_args.env_name = "CrowdSimPredRealGST-v0"
        predict_steps = int(getattr(env_config.sim, "predict_steps", 5))
        # current (x,y) + predicted steps → 2 * (1 + predict_steps)
        algo_args.human_human_edge_input_size = 2 * (1 + predict_steps)

    if randomize is not None:
        env_config.env.randomize_attributes = bool(randomize)
        env_config.humans.random_goal_changing = bool(randomize)

    # Final-evaluation entry: GST policy trees must match the requested regime.
    from domains.crowdnav.regime import (
        REGIME_WITH_RANDOM,
        REGIME_WITHOUT_RANDOM,
        assert_gst_matches_regime,
        gst_model_dir_for_regime,
    )

    regime = (
        REGIME_WITH_RANDOM
        if (
            randomize
            if randomize is not None
            else bool(env_config.env.randomize_attributes)
        )
        else REGIME_WITHOUT_RANDOM
    )
    pred_method = str(getattr(env_config.sim, "predict_method", "none"))
    if pred_method == "inferred":
        # Policy checkpoint dirs (GST_predictor_*) are not GST predictor trees;
        # assert the *predictor* path that the wrapper will load.
        assert_gst_matches_regime(
            getattr(env_config.pred, "model_dir", "")
            or gst_model_dir_for_regime(regime),
            regime,
            predict_method=pred_method,
            entry_point="final_evaluation",
        )
        # Force canonical predictor path for the active regime.
        env_config.pred.model_dir = gst_model_dir_for_regime(regime)
        assert_gst_matches_regime(
            env_config.pred.model_dir,
            regime,
            predict_method=pred_method,
            entry_point="final_evaluation",
        )
    else:
        assert_gst_matches_regime(
            getattr(getattr(env_config, "pred", None), "model_dir", "") or "",
            regime,
            predict_method=pred_method,
            entry_point="final_evaluation",
        )

    env_config.env.test_size = int(n_episodes)

    torch_device = torch.device(
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    # Keep GST pretext wrapper buffers on the same device as Policy / obs.
    if hasattr(env_config, "training"):
        env_config.training.device = str(torch_device)
    torch.manual_seed(seed)

    envs = make_vec_envs(
        algo_args.env_name,
        seed,
        1,
        algo_args.gamma,
        None,
        torch_device,
        True,
        config=env_config,
        pretext_wrapper=env_config.env.use_wrapper,
    )

    if hasattr(envs.venv, "envs"):
        base_env = envs.venv.envs[0].env
    else:
        base_env = envs.venv.unwrapped.envs[0].env
    if human_num is not None:
        pin_episode_human_count(base_env, int(human_num))

    policy_name = env_config.robot.policy
    actor_critic = None
    if policy_name not in ("orca", "social_force"):
        load_path = os.path.join(model_dir, "checkpoints", checkpoint)
        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=policy_name,
        )
        actor_critic.load_state_dict(torch.load(load_path, map_location=torch_device))
        actor_critic.base.nenv = 1
        # Avoid DataParallel for single-env eval (device mismatch with GST wrappers).
        actor_critic.to(torch_device)
        actor_critic.eval()

    episodes = _rollout_episodes(
        actor_critic,
        envs,
        base_env,
        env_config,
        algo_args,
        torch_device,
        n_episodes=n_episodes,
        seed=seed,
        method=method or os.path.basename(model_dir.rstrip("/\\")),
        randomize=bool(
            env_config.env.randomize_attributes
            if randomize is None
            else randomize
        ),
        human_num=human_num,
    )
    envs.close()
    return summarize_episodes(
        episodes,
        method=method or os.path.basename(model_dir.rstrip("/\\")),
        randomize=bool(
            env_config.env.randomize_attributes if randomize is None else randomize
        ),
        metadata={"model_dir": model_dir, "checkpoint": checkpoint},
    )


def evaluate_actor_critic(
    actor_critic,
    reward_fn,
    algo_args,
    env_config,
    device,
    *,
    n_episodes: int,
    seed: int,
    method: str,
    randomize: bool = False,
    human_num: Optional[int] = None,
    horizon_steps: Optional[int] = None,
) -> EvalBundle:
    """Evaluate an in-memory policy (Stage II/III trainers) with raw episodes."""
    from rl.networks.envs import make_vec_envs
    from domains.crowdnav.regime import (
        REGIME_WITH_RANDOM,
        REGIME_WITHOUT_RANDOM,
        assert_gst_matches_regime,
        gst_model_dir_for_regime,
    )

    env_config.env.randomize_attributes = bool(randomize)
    env_config.humans.random_goal_changing = bool(randomize)
    regime = REGIME_WITH_RANDOM if randomize else REGIME_WITHOUT_RANDOM
    pred_method = str(getattr(env_config.sim, "predict_method", "none"))
    if pred_method == "inferred":
        env_config.pred.model_dir = gst_model_dir_for_regime(regime)
        env_config.env.use_wrapper = True
    assert_gst_matches_regime(
        getattr(getattr(env_config, "pred", None), "model_dir", "") or "",
        regime,
        predict_method=pred_method,
        entry_point="final_evaluation",
    )

    envs = make_vec_envs(
        algo_args.env_name,
        seed,
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

    if horizon_steps is None:
        horizon_steps = max(
            1, int(float(base_env.time_limit) / float(env_config.env.time_step))
        )

    episodes = _rollout_episodes(
        actor_critic,
        envs,
        base_env,
        env_config,
        algo_args,
        device,
        n_episodes=n_episodes,
        seed=seed,
        method=method,
        randomize=randomize,
        human_num=human_num,
        horizon_steps=horizon_steps,
    )
    envs.close()
    return summarize_episodes(
        episodes, method=method, randomize=randomize, metadata={}
    )


def _rollout_episodes(
    actor_critic,
    envs,
    base_env,
    env_config,
    algo_args,
    device,
    *,
    n_episodes: int,
    seed: int,
    method: str,
    randomize: bool,
    human_num: Optional[int] = None,
    horizon_steps: Optional[int] = None,
) -> List[EpisodeRecord]:
    from crowd_sim.envs.utils.info import Collision, Danger, ReachGoal, Timeout

    if horizon_steps is None:
        horizon_steps = 10**9

    policy_name = env_config.robot.policy
    use_nn = actor_critic is not None and policy_name not in ("orca", "social_force")

    if use_nn:
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
    else:
        eval_recurrent_hidden_states = {}
    eval_masks = torch.zeros(1, 1, device=device)

    records: List[EpisodeRecord] = []
    time_limit = float(base_env.time_limit)

    for k in range(n_episodes):
        done = False
        step_counter = 0
        obs = envs.reset()
        global_time = 0.0
        path_len = 0.0
        too_close = 0.0
        min_dists: List[float] = []
        last_pos = obs["robot_node"][0, 0, :2].cpu().numpy()
        infos = [{"info": None}]

        while not done and step_counter < horizon_steps:
            step_counter += 1
            if use_nn:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True,
                    )
            else:
                action = torch.zeros([1, 2], device=device)

            if not done:
                global_time = float(base_env.global_time)

            if (
                getattr(algo_args, "env_name", "") == "CrowdSimPredRealGST-v0"
                and env_config.env.use_wrapper
            ):
                out_pred = obs["spatial_edges"][:, :, 2:].to("cpu").numpy()
                ack = envs.talk2Env(out_pred)
                assert all(ack)

            obs, _rew, done, infos = envs.step(action)
            path_len += float(
                np.linalg.norm(obs["robot_node"][0, 0, :2].cpu().numpy() - last_pos)
            )
            last_pos = obs["robot_node"][0, 0, :2].cpu().numpy()
            if isinstance(infos[0]["info"], Danger):
                too_close += 1
                min_dists.append(float(infos[0]["info"].min_dist))
            eval_masks = torch.tensor(
                [[0.0] if d else [1.0] for d in done],
                dtype=torch.float32,
                device=device,
            )

        info = infos[0]["info"]
        success = collision = timeout = 0
        outcome = "other"
        nav_time = time_limit
        if isinstance(info, ReachGoal):
            success, outcome, nav_time = 1, "success", global_time
        elif isinstance(info, Collision):
            collision, outcome = 1, "collision"
        elif isinstance(info, Timeout) or step_counter >= horizon_steps or not done:
            timeout, outcome = 1, "timeout"
        else:
            timeout, outcome = 1, "timeout"

        records.append(
            EpisodeRecord(
                episode_index=k,
                seed=seed,
                outcome=outcome,
                success=success,
                collision=collision,
                timeout=timeout,
                nav_time=float(nav_time),
                path_length=float(path_len),
                intrusion_ratio_pct=float(too_close / max(step_counter, 1) * 100.0),
                min_dist_during_intrusion=(
                    float(np.mean(min_dists)) if min_dists else None
                ),
                steps=step_counter,
                human_num=human_num,
                method=method,
                randomize=randomize,
            )
        )
    return records


def proxy_metrics_from_bundle(bundle: EvalBundle) -> ProxyMetrics:
    return ProxyMetrics(
        sr=bundle.sr.mean,
        cr=bundle.cr.mean,
        tr=bundle.tr.mean,
        nt=bundle.nt.mean,
        pl=bundle.pl.mean,
        itr=bundle.itr.mean,
        sd=bundle.sd.mean,
    )


def format_table2_row(name: str, bundle: EvalBundle) -> str:
    return (
        f"{name:24s}  SR={bundle.sr.format_pct()}  "
        f"CR={bundle.cr.format_pct()}  TR={bundle.tr.format_pct()}"
    )


def format_table1_row(bundle: EvalBundle) -> str:
    f = bundle.aggregate_dict()["format"]
    return (
        f"{bundle.method:16s} rand={bundle.randomize}  "
        f"SR={f['SR']} CR={f['CR']} TR={f['TR']} "
        f"NT={f['NT']} PL={f['PL']} ITR={f['ITR']} SD={f['SD']}"
    )

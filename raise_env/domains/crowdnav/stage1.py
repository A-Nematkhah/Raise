"""CrowdNav Stage I — Score1 over recorded CrowdNav trajectories (pack-local)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from domains.crowdnav.objective_constants import PPO_GAMMA
from domains.crowdnav.settings import CrowdNavSettings, get_settings
from domains.crowdnav.state import (
    CrowdNavRewardState,
    HumanState,
    RobotState,
    smoke_action,
    with_kinematics,
)
from raise_core.rules import (
    TrajectoryCategory,
    label_to_category,
    spearman_correlation,
)
from raise_core.scoring import Score1Result

logger = logging.getLogger(__name__)

DEFAULT_STAGE1_DATASET = "domains/crowdnav/data/stage1_dataset"
STATE_SCHEMA_VERSION = 2


@dataclass
class CrowdNavTrajectoryRecord:
    trajectory_id: str
    scenario_id: str
    seed: Optional[int]
    states: Tuple[CrowdNavRewardState, ...]
    # Upstream episode outcome: success (ReachGoal) | collision | timeout.
    label: str
    behavior: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = tuple(self.states)
        if len(self.states) < 1:
            raise ValueError("CrowdNavTrajectoryRecord needs >=1 state")
        key = str(self.label).strip().lower()
        if key in ("reachgoal", "reach_goal", "reaching goal", "goal"):
            key = "success"
        if key not in ("success", "collision", "timeout"):
            raise ValueError(f"Invalid label: {self.label!r}")
        self.label = key

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def category(self) -> TrajectoryCategory:
        return label_to_category(self.label)

    @property
    def nav_time(self) -> float:
        return float(self.states[-1].global_time)


def _state_to_dict(s: CrowdNavRewardState) -> Dict[str, Any]:
    return {
        "robot": asdict(s.robot),
        "humans": [asdict(h) for h in s.humans],
        "danger": s.danger,
        "success_reward": s.success_reward,
        "collision_penalty": s.collision_penalty,
        "discomfort_penalty_factor": s.discomfort_penalty_factor,
        "potential_delta": s.potential_delta,
        "potential_factor": s.potential_factor,
        "action": list(s.action),
        "collision": s.collision,
        "reached_goal": s.reached_goal,
        "timeout": s.timeout,
        "min_separation": s.min_separation,
        "discomfort_dist": s.discomfort_dist,
        "goal_distance": s.goal_distance,
        "progress": s.progress,
        "time_step": s.time_step,
        "global_time": s.global_time,
        "time_limit": s.time_limit,
    }


def _state_from_dict(d: Dict[str, Any]) -> CrowdNavRewardState:
    r = d["robot"]
    return CrowdNavRewardState(
        robot=RobotState(
            px=float(r["px"]),
            py=float(r["py"]),
            vx=float(r["vx"]),
            vy=float(r["vy"]),
            gx=float(r["gx"]),
            gy=float(r["gy"]),
            theta=float(r["theta"]),
            radius=float(r["radius"]),
            v_pref=float(r["v_pref"]),
            kinematics=str(r.get("kinematics", "holonomic")),
        ),
        humans=tuple(
            HumanState(
                px=float(h["px"]),
                py=float(h["py"]),
                vx=float(h["vx"]),
                vy=float(h["vy"]),
                radius=float(h["radius"]),
                visible=bool(h.get("visible", False)),
            )
            for h in d.get("humans") or []
        ),
        action=(float(d["action"][0]), float(d["action"][1])),
        collision=bool(d["collision"]),
        reached_goal=bool(d["reached_goal"]),
        timeout=bool(d["timeout"]),
        min_separation=float(d["min_separation"]),
        discomfort_dist=float(d["discomfort_dist"]),
        goal_distance=float(d["goal_distance"]),
        progress=float(d["progress"]),
        time_step=float(d["time_step"]),
        global_time=float(d["global_time"]),
        time_limit=float(d["time_limit"]),
        danger=bool(d.get("danger", False)),
        success_reward=float(d.get("success_reward", 1.0)),
        collision_penalty=float(d.get("collision_penalty", -0.25)),
        discomfort_penalty_factor=float(d.get("discomfort_penalty_factor", 0.5)),
        potential_delta=float(d.get("potential_delta", 0.0)),
        potential_factor=float(d.get("potential_factor", 2.0)),
    )


def save_crowdnav_dataset(
    path: str,
    trajectories: Sequence[CrowdNavTrajectoryRecord],
    *,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "trajectories.jsonl"), "w", encoding="utf-8") as fh:
        for t in trajectories:
            rec = {
                "trajectory_id": t.trajectory_id,
                "scenario_id": t.scenario_id,
                "seed": t.seed,
                "label": t.label,
                "behavior": t.behavior,
                "metadata": t.metadata,
                "states": [_state_to_dict(s) for s in t.states],
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {
        "domain": "crowdnav",
        "n_trajectories": len(trajectories),
        "labels": {
            lab: sum(1 for t in trajectories if t.label == lab)
            for lab in ("success", "collision", "timeout")
        },
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    manifest.update(extra_manifest or {})
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def read_dataset_manifest(path: str) -> Dict[str, Any]:
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def load_crowdnav_dataset(path: str) -> List[CrowdNavTrajectoryRecord]:
    traj_path = os.path.join(path, "trajectories.jsonl")
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(traj_path)
    manifest = read_dataset_manifest(path)
    if manifest and manifest.get("domain") != "crowdnav":
        raise ValueError(f"{path} is not a CrowdNav Stage I dataset (domain={manifest.get('domain')!r})")
    out: List[CrowdNavTrajectoryRecord] = []
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            out.append(
                CrowdNavTrajectoryRecord(
                    trajectory_id=str(raw["trajectory_id"]),
                    scenario_id=str(raw["scenario_id"]),
                    seed=raw.get("seed"),
                    states=tuple(_state_from_dict(s) for s in raw["states"]),
                    label=str(raw["label"]),
                    behavior=str(raw.get("behavior") or ""),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
    return out


def with_settings(
    trajectories: Sequence[CrowdNavTrajectoryRecord], settings: CrowdNavSettings
) -> List[CrowdNavTrajectoryRecord]:
    """Express recorded actions in ``settings``' kinematics without reconverting."""
    converted = []
    for trajectory in trajectories:
        source_kinematics = {state.robot.kinematics for state in trajectory.states}
        if not source_kinematics:
            converted.append(trajectory)
        elif source_kinematics == {settings.kinematics}:
            converted.append(trajectory)
        elif source_kinematics == {"holonomic"} and settings.kinematics == "unicycle":
            converted.append(
                replace(trajectory, states=with_kinematics(trajectory.states, settings))
            )
        else:
            raise ValueError(
                "Cannot convert Stage I trajectory from "
                f"{sorted(source_kinematics)} to {settings.kinematics!r} kinematics"
            )
    return converted


# ------------------------------------------------------------------ scoring


def _step_rewards(reward_fn: Any, traj: CrowdNavTrajectoryRecord) -> List[float]:
    if hasattr(reward_fn, "reset"):
        reward_fn.reset()
    return [float(reward_fn.compute(st)) for st in traj.states]


def _discounted_return(rewards: Sequence[float], gamma: float = PPO_GAMMA) -> float:
    """The quantity PPO maximizes from the first frame of the episode."""
    if not rewards:
        return 0.0
    return float(np.dot(np.asarray(rewards, dtype=float), gamma ** np.arange(len(rewards))))


def _step_rule(st: CrowdNavRewardState) -> float:
    """Analytical per-step preference on a non-terminal frame (higher = better).

    Progress toward the goal per step at preferred speed, minus intrusion
    into the discomfort zone (the two shaping signals of the upstream task).
    """
    max_progress = max(1e-6, float(st.robot.v_pref) * float(st.time_step))
    intrusion = max(0.0, float(st.discomfort_dist) - float(st.min_separation))
    return float(st.progress) / max_progress - 2.0 * intrusion / max(1e-6, float(st.discomfort_dist))


def _step_spearman(
    steps: Dict[str, List[float]], trajectories: Sequence[CrowdNavTrajectoryRecord]
) -> Tuple[float, float, int, int]:
    """Mean within-trajectory Spearman(step reward, step rule) on non-terminal frames.

    Returns (mean_rho, deg_frac, n_trajs, n_deg); a reward that is constant
    along a trajectory has no defined rank correlation there (degenerate).
    """
    rhos: List[float] = []
    n_deg = 0
    for traj in trajectories:
        frames = [
            (r, _step_rule(st))
            for r, st in zip(steps[traj.trajectory_id], traj.states)
            if not (st.collision or st.reached_goal or st.timeout)
        ]
        rho = spearman_correlation([f[0] for f in frames], [f[1] for f in frames])
        if rho is None or rho != rho:
            n_deg += 1
            continue
        rhos.append(float(rho))
    n = len(trajectories)
    return (float(np.mean(rhos)) if rhos else 0.0), float(n_deg) / float(max(1, n)), n, n_deg


def _preference_auc(
    returns: Dict[str, float],
    trajectories: Sequence[CrowdNavTrajectoryRecord],
) -> float:
    """
    Pairwise preference success ≻ timeout ≻ collision on ``returns``, as ``2·AUC − 1``.

    The order is the upstream reward's (+1 goal, 0 timeout, −0.25 collision).
    With discounted returns this asks whether PPO's objective ranks reaching
    the goal above freezing until the time limit, and both above a collision.
    """
    by_label: Dict[str, List[float]] = {"success": [], "timeout": [], "collision": []}
    for traj in trajectories:
        by_label[traj.label].append(float(returns[traj.trajectory_id]))
    correct = 0.0
    total = 0
    for better, worse in (("success", "timeout"), ("success", "collision"), ("timeout", "collision")):
        for a in by_label[better]:
            for b in by_label[worse]:
                total += 1
                if a > b:
                    correct += 1.0
                elif a == b:
                    correct += 0.5
    if total < 1:
        return 0.0
    return float(2.0 * correct / float(total) - 1.0)


def _success_alignment(
    returns: Dict[str, float],
    trajectories: Sequence[CrowdNavTrajectoryRecord],
    key: Callable[[CrowdNavTrajectoryRecord], float],
) -> float:
    """Among successes, Spearman(episode return, ``key``); 0 when undefined."""
    succ = [t for t in trajectories if t.label == "success"]
    if len(succ) < 3:
        return 0.0
    values = [float(key(t)) for t in succ]
    if max(values) - min(values) < 1e-9:
        return 0.0
    rho = spearman_correlation([float(returns[t.trajectory_id]) for t in succ], values)
    if rho is None or rho != rho:
        return 0.0
    return float(rho)


def _danger_fraction(traj: CrowdNavTrajectoryRecord) -> float:
    n = sum(1 for st in traj.states if st.min_separation < st.discomfort_dist)
    return float(n) / float(max(1, traj.length))


# Score1 = W_SAFETY·safety + W_SAFETY·max(0, safety)·quality.
# Like the Stage II fitness, navigation quality only counts once the reward
# ranks reaching the goal above failure. Quality is a convex mix in [-1, 1]:
# efficiency (faster successes earn more) and comfort (fewer discomfort-zone
# steps earn more) rank whole successful episodes, the step term checks
# per-frame shaping. The weights are a design choice, not fitted.
_W_SAFETY = 0.5
_W_EFFICIENCY = 0.40
_W_COMFORT = 0.25
_W_STEP = 0.35
# Hard reject: PPO's discounted objective ranks failed episodes above
# successful ones more often than the reverse.
PREFERS_FAILURE_SAFETY = 0.0


def score_crowdnav_dataset(
    reward_fn: Any,
    trajectories: Sequence[CrowdNavTrajectoryRecord],
    *,
    candidate_id: str = "",
) -> Score1Result:
    """
    CrowdNav Score1: a cheap sanity filter for a candidate reward (in [-1, 1]).

    ::

        safety     = 2·AUC − 1 of success ≻ timeout ≻ collision on discounted returns (γ = PPO's)
        efficiency = Spearman(return, −nav_time) over successes
        comfort    = Spearman(return, −danger_fraction) over successes
        quality    = 0.40·efficiency + 0.25·comfort + 0.35·step_spearman
        Score1     = 0.5·safety + 0.5·max(0, safety)·quality

    Rejected: ``no_signal`` (no component carries information) or
    ``prefers_failure`` (safety < PREFERS_FAILURE_SAFETY).
    """
    del candidate_id  # reserved for logging / future per-candidate caches
    if not trajectories:
        return Score1Result(score=0.0, degenerate_fraction=1.0, rejected=True, reject_reason="empty")

    steps = {t.trajectory_id: _step_rewards(reward_fn, t) for t in trajectories}
    returns = {tid: _discounted_return(r) for tid, r in steps.items()}
    step_rho, deg_frac, n_trajs, n_deg = _step_spearman(steps, trajectories)
    safety = _preference_auc(returns, trajectories)
    efficiency = _success_alignment(returns, trajectories, lambda t: -t.nav_time)
    comfort = _success_alignment(returns, trajectories, lambda t: -_danger_fraction(t))

    quality = _W_EFFICIENCY * efficiency + _W_COMFORT * comfort + _W_STEP * step_rho
    raw = float(np.clip(_W_SAFETY * safety + _W_SAFETY * max(0.0, safety) * quality, -1.0, 1.0))

    reject_reason = None
    if deg_frac >= 0.95 and max(abs(safety), abs(efficiency), abs(comfort)) < 1e-9:
        reject_reason = "no_signal"
    elif safety < PREFERS_FAILURE_SAFETY:
        reject_reason = "prefers_failure"

    return Score1Result(
        score=-1.0 if reject_reason else raw,
        degenerate_fraction=deg_frac,
        n_pairs=n_trajs,
        n_degenerate=n_deg,
        rejected=reject_reason is not None,
        reject_reason=reject_reason,
        raw_score=raw,
        scenario_scores={
            "safety": float(safety),
            "quality": float(quality),
            "step_spearman": float(step_rho),
            "efficiency": float(efficiency),
            "comfort": float(comfort),
        },
    )


# ------------------------------------------------------------------ smoke fixture


def _smoke_frame(
    i: int,
    *,
    config: Any,
    py: float,
    vy: float,
    progress: float,
    min_sep: float = 1.0,
    collision: bool = False,
    reached: bool = False,
    timeout: bool = False,
    global_time: Optional[float] = None,
) -> CrowdNavRewardState:
    from domains.crowdnav.settings import get_settings

    settings = get_settings()
    goal_y = float(config.sim.circle_radius)
    time_step = float(config.env.time_step)
    time_limit = float(config.env.time_limit)
    discomfort_dist = float(config.reward.discomfort_dist)
    robot_radius = float(config.robot.radius)
    human_radius = float(config.humans.radius)
    v_pref = float(config.robot.v_pref)
    return CrowdNavRewardState(
        robot=RobotState(
            px=0.0,
            py=py,
            vx=0.0,
            vy=vy,
            gx=0.0,
            gy=goal_y,
            theta=np.pi / 2,
            radius=robot_radius,
            v_pref=v_pref,
            kinematics=settings.kinematics,
        ),
        humans=(
            HumanState(
                px=goal_y / 4.0,
                py=py + goal_y / 8.0,
                vx=-0.5,
                vy=0.0,
                radius=human_radius,
                visible=True,
            ),
        ),
        action=smoke_action(0.0, vy),
        collision=collision,
        reached_goal=reached,
        timeout=timeout,
        min_separation=min_sep,
        discomfort_dist=discomfort_dist,
        goal_distance=abs(goal_y - py),
        progress=progress,
        time_step=time_step,
        global_time=time_step * (i + 1) if global_time is None else global_time,
        time_limit=time_limit,
        danger=min_sep < discomfort_dist,
        success_reward=float(config.reward.success_reward),
        collision_penalty=float(config.reward.collision_penalty),
        discomfort_penalty_factor=float(config.reward.discomfort_penalty_factor),
        potential_delta=float(progress),
        potential_factor=3.0 if settings.kinematics == "unicycle" else 2.0,
    )


def smoke_trajectories() -> List[CrowdNavTrajectoryRecord]:
    """Fixture trajectories behind :func:`make_smoke_score_fn`."""
    from domains.crowdnav.env_wrapper import load_env_config

    config = load_env_config(get_settings().env_config_source)
    circle_radius = float(config.sim.circle_radius)
    time_step = float(config.env.time_step)
    time_limit = float(config.env.time_limit)
    discomfort_dist = float(config.reward.discomfort_dist)
    v_pref = float(config.robot.v_pref)

    def _traj(tid: str, label: str, frames: Sequence[CrowdNavRewardState]) -> CrowdNavTrajectoryRecord:
        return CrowdNavTrajectoryRecord(
            trajectory_id=tid, scenario_id="smoke", seed=0, states=tuple(frames), label=label, behavior="smoke"
        )

    def _arrival_frames(speed_fraction: float, clearance: float):
        speed = v_pref * speed_fraction
        steps = max(1, int(np.ceil(2.0 * circle_radius / (speed * time_step))))
        frames = []
        previous_distance = 2.0 * circle_radius
        for i in range(steps):
            py = min(circle_radius, -circle_radius + speed * time_step * (i + 1))
            if i == steps - 1:
                py = circle_radius
            current_distance = abs(circle_radius - py)
            frames.append(
                _smoke_frame(
                    i,
                    config=config,
                    py=py,
                    vy=speed,
                    progress=previous_distance - current_distance,
                    min_sep=clearance,
                    reached=i == steps - 1,
                )
            )
            previous_distance = current_distance
        return frames

    fast = _arrival_frames(1.0, discomfort_dist * 2.0)
    slow = _arrival_frames(0.6, discomfort_dist * 1.5)
    medium = _arrival_frames(0.8, discomfort_dist * 1.75)
    crash = [
        _smoke_frame(
            i,
            config=config,
            py=-circle_radius + v_pref * time_step * (i + 1),
            vy=v_pref,
            progress=v_pref * time_step,
            min_sep=discomfort_dist * (0.5 - 0.5 * i),
            collision=(i == 2),
        )
        for i in range(3)
    ]
    frozen = [
        _smoke_frame(
            0,
            config=config,
            py=-circle_radius,
            vy=0.0,
            progress=0.0,
            min_sep=discomfort_dist * 2.0,
            timeout=True,
            global_time=max(0.0, time_limit - time_step),
        )
    ]
    return [
        _traj("smoke_fast", "success", fast),
        _traj("smoke_slow", "success", slow),
        _traj("smoke_medium", "success", medium),
        _traj("smoke_crash", "collision", crash),
        _traj("smoke_frozen", "timeout", frozen),
    ]


def make_smoke_score_fn() -> Callable[..., Score1Result]:
    """Tiny fixture Score1 for --fast / smoke mode."""
    trajs = smoke_trajectories()

    def _fn(reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        result = score_crowdnav_dataset(reward_fn, trajs, candidate_id=candidate_id)
        # Smoke fixture: never hard-reject; keep raw_score for ranking.
        if result.raw_score is not None:
            return replace(result, score=float(result.raw_score), rejected=False, reject_reason=None)
        return result

    return _fn


def resolve_dataset_path(dataset_path: Optional[str] = None) -> str:
    if dataset_path is not None and str(dataset_path).strip():
        return str(dataset_path).strip()
    return DEFAULT_STAGE1_DATASET


def make_score_fn(
    *,
    mode: str = "dataset",
    dataset_path: Optional[str] = None,
    settings: Optional[CrowdNavSettings] = None,
) -> Tuple[Callable[..., Any], Optional[Any]]:
    """Score1 over the dataset with actions in ``settings``' kinematics (default: active)."""
    key = str(mode).strip().lower()
    if key == "smoke":
        logger.warning("CrowdNav Stage I using smoke score (not real dataset)")
        return make_smoke_score_fn(), None
    if key != "dataset":
        raise ValueError(f"Unknown score1_mode: {mode!r}")
    path = resolve_dataset_path(dataset_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CrowdNav Stage I dataset not found at {path}. "
            f"Run: python scripts/collect_crowdnav_stage1_dataset.py --out {path}"
        )
    dataset = with_settings(load_crowdnav_dataset(path), settings or get_settings())
    logger.info("Loaded CrowdNav Stage I dataset from %s (%d trajs)", path, len(dataset))

    def _fn(reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        return score_crowdnav_dataset(reward_fn, dataset, candidate_id=candidate_id)

    return _fn, dataset

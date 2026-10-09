"""Highway Stage I — Score1 over highway trajectories (pack-local)."""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from domains.highway.action_config import ActionModeSettings, get_action_settings
from domains.highway.objective_constants import LAG_SPEED_MPS, PPO_GAMMA
from domains.highway.state import (
    EgoVehicle,
    HighwayRewardState,
    NearbyVehicle,
    action_for_mode,
    lane_index_from_y,
)
from raise_core.rules import (
    TrajectoryCategory,
    label_to_category,
    spearman_correlation,
)
from raise_core.scoring import Score1Result

logger = logging.getLogger(__name__)

DEFAULT_STAGE1_DATASET = "domains/highway/data/stage1_dataset"
# 2: lane_index = nearest highway-env lane; off-road frames carry timeout=False.
STATE_SCHEMA_VERSION = 2


@dataclass
class HighwayTrajectoryRecord:
    trajectory_id: str
    scenario_id: str
    seed: Optional[int]
    states: Tuple[HighwayRewardState, ...]
    # success | collision | timeout. Label "timeout" is the shared rules name
    # for a non-collision failure (off-road); it is unrelated to state.timeout,
    # which is True only on the last frame of a survived (truncated) episode.
    label: str
    behavior: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = tuple(self.states)
        if len(self.states) < 1:
            raise ValueError("HighwayTrajectoryRecord needs >=1 state")
        key = str(self.label).strip().lower()
        if key in ("reachgoal", "reach_goal", "goal", "survive"):
            key = "success"
        elif key in ("collide", "fail", "crash"):
            key = "collision"
        elif key in ("time_out", "other", "truncation", "off_road", "offroad"):
            key = "timeout"
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
    def nav_length(self) -> float:
        last = self.states[-1]
        if last.global_time > 0:
            return float(last.global_time)
        return float(self.length) * float(last.time_step)


def _ego_from_dict(d: Dict[str, Any]) -> EgoVehicle:
    return EgoVehicle(
        x=float(d["x"]),
        y=float(d["y"]),
        vx=float(d["vx"]),
        vy=float(d["vy"]),
        heading=float(d["heading"]),
        speed=float(d["speed"]),
        lane_index=float(d["lane_index"]),
        on_road=bool(d["on_road"]),
    )


def _state_from_dict(d: Dict[str, Any]) -> HighwayRewardState:
    others = tuple(
        NearbyVehicle(
            x=float(o["x"]),
            y=float(o["y"]),
            vx=float(o["vx"]),
            vy=float(o["vy"]),
            heading=float(o["heading"]),
        )
        for o in (d.get("others") or [])
    )
    return HighwayRewardState(
        ego=_ego_from_dict(d["ego"]),
        others=others,
        collision=bool(d["collision"]),
        off_road=bool(d["off_road"]),
        timeout=bool(d["timeout"]),
        action=d.get("action"),
        time_step=float(d["time_step"]),
        global_time=float(d["global_time"]),
        time_limit=float(d["time_limit"]),
        progress=float(d.get("progress", 0.0)),
        speed=float(d.get("speed", d["ego"]["speed"])),
    )


def _state_to_dict(s: HighwayRewardState) -> Dict[str, Any]:
    return {
        "ego": asdict(s.ego),
        "others": [asdict(o) for o in s.others],
        "collision": s.collision,
        "off_road": s.off_road,
        "timeout": s.timeout,
        "action": s.action,
        "time_step": s.time_step,
        "global_time": s.global_time,
        "time_limit": s.time_limit,
        "progress": s.progress,
        "speed": s.speed,
    }


def save_highway_dataset(
    path: str, trajectories: Sequence[HighwayTrajectoryRecord]
) -> None:
    os.makedirs(path, exist_ok=True)
    traj_path = os.path.join(path, "trajectories.jsonl")
    with open(traj_path, "w", encoding="utf-8") as fh:
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
        "domain": "highway",
        "n_trajectories": len(trajectories),
        "labels": sorted({t.label for t in trajectories}),
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def read_dataset_manifest(path: str) -> Dict[str, Any]:
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest_path):
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def migrate_state_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade one serialized state to ``STATE_SCHEMA_VERSION`` (returns a new dict)."""
    out = dict(d)
    ego = dict(out["ego"])
    ego["lane_index"] = float(lane_index_from_y(float(ego["y"])))
    out["ego"] = ego
    if bool(out.get("off_road")):
        out["timeout"] = False
    return out


def migrate_highway_dataset(path: str, *, backup: bool = True) -> int:
    """Migrate ``path`` in place; return trajectories rewritten (0 if already current)."""
    traj_path = os.path.join(path, "trajectories.jsonl")
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(traj_path)
    manifest = read_dataset_manifest(path)
    if int(manifest.get("state_schema_version") or 1) >= STATE_SCHEMA_VERSION:
        return 0

    if backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        abs_path = os.path.abspath(path)
        shutil.copytree(
            abs_path,
            os.path.join(os.path.dirname(abs_path), f"{os.path.basename(abs_path)}_backup_{stamp}"),
        )

    records = []
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["states"] = [migrate_state_dict(s) for s in rec["states"]]
            records.append(rec)

    tmp_path = traj_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, traj_path)

    manifest["state_schema_version"] = STATE_SCHEMA_VERSION
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return len(records)


def load_highway_dataset(path: str) -> List[HighwayTrajectoryRecord]:
    traj_path = os.path.join(path, "trajectories.jsonl")
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(traj_path)
    version = int(read_dataset_manifest(path).get("state_schema_version") or 1)
    if version < STATE_SCHEMA_VERSION:
        logger.warning(
            "Highway Stage I dataset %s has state_schema_version=%d (current %d): "
            "lane_index / timeout fields are stale. Run "
            "scripts/migrate_highway_stage1_dataset.py --dataset %s",
            path,
            version,
            STATE_SCHEMA_VERSION,
            path,
        )
    out: List[HighwayTrajectoryRecord] = []
    with open(traj_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            states = tuple(_state_from_dict(s) for s in raw["states"])
            out.append(
                HighwayTrajectoryRecord(
                    trajectory_id=str(raw["trajectory_id"]),
                    scenario_id=str(raw["scenario_id"]),
                    seed=raw.get("seed"),
                    states=states,
                    label=str(raw["label"]),
                    behavior=str(raw.get("behavior") or ""),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
    return out


def with_action_mode(
    trajectories: Sequence[HighwayTrajectoryRecord], settings: ActionModeSettings
) -> List[HighwayTrajectoryRecord]:
    """Map the recorded meta action indices to ``state.action`` of ``settings``' mode.

    The dataset is collected with DiscreteMetaAction; a continuous-mode reward
    reads ``state.action[0]`` and would fail on every int frame.
    """
    if settings.mode != "continuous":
        return list(trajectories)
    return [
        replace(
            traj,
            states=tuple(
                replace(st, action=action_for_mode(st.action, settings)) for st in traj.states
            ),
        )
        for traj in trajectories
    ]


def _traj_mean_speed(traj: HighwayTrajectoryRecord) -> float:
    if traj.metadata.get("mean_speed") is not None:
        try:
            return float(traj.metadata["mean_speed"])
        except (TypeError, ValueError):
            pass
    speeds = [float(s.speed) for s in traj.states]
    return float(np.mean(speeds)) if speeds else 0.0


def _traj_mean_progress(traj: HighwayTrajectoryRecord) -> float:
    if traj.metadata.get("mean_progress") is not None:
        try:
            return float(traj.metadata["mean_progress"])
        except (TypeError, ValueError):
            pass
    # Sum of per-frame progress ≈ net forward travel.
    return float(sum(float(s.progress) for s in traj.states))


def _step_rule(st: HighwayRewardState) -> float:
    """Analytical per-step preference on a surviving frame (higher = better)."""
    speed = float(st.speed)
    speed_term = float(np.clip(speed / 25.0, 0.0, 1.5))
    # Lag behind traffic (~20 m/s) is the crawl failure mode.
    crawl = max(0.0, 18.0 - speed) / 18.0
    return float(3.0 * float(st.progress) + 2.0 * speed_term - 1.5 * crawl)


def _step_rewards(reward_fn: Any, traj: HighwayTrajectoryRecord) -> List[float]:
    if hasattr(reward_fn, "reset"):
        reward_fn.reset()
    return [float(reward_fn.compute(st)) for st in traj.states]


def _discounted_return(rewards: Sequence[float], gamma: float = PPO_GAMMA) -> float:
    """The quantity PPO maximizes from the first frame of the episode."""
    if not rewards:
        return 0.0
    return float(np.dot(np.asarray(rewards, dtype=float), gamma ** np.arange(len(rewards))))


def _step_spearman(
    steps: Dict[str, List[float]], trajectories: Sequence[HighwayTrajectoryRecord]
) -> Tuple[float, float, int, int]:
    """Mean within-trajectory Spearman(step reward, step rule) on surviving frames.

    Returns (mean_rho, deg_frac, n_trajs, n_deg); a reward that is constant
    along a trajectory has no defined rank correlation there (degenerate).
    """
    rhos: List[float] = []
    n_deg = 0
    for traj in trajectories:
        frames = [
            (r, _step_rule(st))
            for r, st in zip(steps[traj.trajectory_id], traj.states)
            if not (st.collision or st.off_road)
        ]
        rho = spearman_correlation([f[0] for f in frames], [f[1] for f in frames])
        if rho is None or rho != rho:
            n_deg += 1
            continue
        rhos.append(float(rho))
    n = len(trajectories)
    deg_frac = float(n_deg) / float(max(1, n))
    return (float(np.mean(rhos)) if rhos else 0.0), deg_frac, n, n_deg


def _preference_auc(
    returns: Dict[str, float],
    trajectories: Sequence[HighwayTrajectoryRecord],
) -> float:
    """
    Pairwise preference success ≻ timeout ≻ collision on ``returns``, as ``2·AUC − 1``.

    With discounted returns this asks whether PPO's objective ranks a survived
    episode above a failed one. A positive per-step reward already favours
    survival (termination forfeits future reward); a crash bonus that outweighs
    the discounted tail does not.
    """
    by_label: Dict[str, List[float]] = {"success": [], "timeout": [], "collision": []}
    for traj in trajectories:
        if traj.label in by_label:
            by_label[traj.label].append(float(returns[traj.trajectory_id]))

    pairs = (
        ("success", "timeout"),
        ("success", "collision"),
        ("timeout", "collision"),
    )
    correct = 0
    total = 0
    for better, worse in pairs:
        for a in by_label[better]:
            for b in by_label[worse]:
                total += 1
                if a > b:
                    correct += 1
                elif a == b:
                    correct += 0.5
    if total < 1:
        return 0.0
    auc = float(correct) / float(total)
    return float(2.0 * auc - 1.0)  # [-1, 1]


def _throughput_alignment(
    returns: Dict[str, float],
    trajectories: Sequence[HighwayTrajectoryRecord],
) -> float:
    """
    Among success trajs, Spearman(episode_return, throughput_score).

    throughput = 0.6·norm(progress) + 0.4·norm(speed). Rewards that only
    like crawling lose here against progress/speed-aligned rewards.
    """
    succ = [t for t in trajectories if t.label == "success"]
    if len(succ) < 3:
        return 0.0
    rets = [float(returns[t.trajectory_id]) for t in succ]
    speeds = [_traj_mean_speed(t) for t in succ]
    progs = [_traj_mean_progress(t) for t in succ]
    s_max = max(speeds) if max(speeds) > 1e-6 else 1.0
    p_max = max(progs) if max(progs) > 1e-6 else 1.0
    thr = [0.6 * (p / p_max) + 0.4 * (s / s_max) for p, s in zip(progs, speeds)]
    rho = spearman_correlation(rets, thr)
    if rho is None or (isinstance(rho, float) and rho != rho):
        return 0.0
    return float(rho)


def _traj_overtake_count(traj: HighwayTrajectoryRecord) -> float:
    meta = traj.metadata or {}
    try:
        return float(meta.get("overtakes", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _overtake_alignment(
    returns: Dict[str, float],
    trajectories: Sequence[HighwayTrajectoryRecord],
) -> float:
    """
    Among success trajs, Spearman(episode_return, overtake_count).

    Rewards that value skillful passing over passive crawl / reckless speed
    score higher when genuine overtake trajs exist in the dataset. Returns
    0 when fewer than 3 successes or when overtake counts are constant.
    """
    succ = [t for t in trajectories if t.label == "success"]
    if len(succ) < 3:
        return 0.0
    rets = [float(returns[t.trajectory_id]) for t in succ]
    overtakes = [_traj_overtake_count(t) for t in succ]
    if max(overtakes) - min(overtakes) < 1e-9:
        return 0.0
    rho = spearman_correlation(rets, overtakes)
    if rho is None or (isinstance(rho, float) and rho != rho):
        return 0.0
    return float(rho)


TRAJECTORY_ROLES = ("positive", "neutral", "lag", "failure")


def trajectory_role(traj: HighwayTrajectoryRecord) -> str:
    """``metadata["role"]`` if set, else inferred for datasets collected before roles.

    positive: a success at traffic speed (the behaviour we want); neutral: a
    success that is neither wanted nor a failure (e.g. lowest-gear cruise);
    lag: a success well below traffic speed; failure: collision or off-road.
    """
    role = str((traj.metadata or {}).get("role") or "").strip().lower()
    if role in TRAJECTORY_ROLES:
        return role
    if traj.label != "success":
        return "failure"
    beh = str(traj.behavior or "").lower()
    if any(tag in beh for tag in ("crawl", "lag")) or _traj_mean_speed(traj) < LAG_SPEED_MPS:
        return "lag"
    if "cruise" in beh:
        return "neutral"
    return "positive"


def _crawl_penalty(
    returns: Dict[str, float],
    trajectories: Sequence[HighwayTrajectoryRecord],
) -> float:
    """Positive when lag successes earn at least the mean return of positive ones."""
    lag_rets: List[float] = []
    pos_rets: List[float] = []
    for t in trajectories:
        role = trajectory_role(t)
        if role == "lag":
            lag_rets.append(float(returns[t.trajectory_id]))
        elif role == "positive":
            pos_rets.append(float(returns[t.trajectory_id]))
    if not lag_rets or not pos_rets:
        return 0.0
    gap = float(np.mean(lag_rets) - np.mean(pos_rets))
    return float(max(0.0, gap) / (abs(float(np.mean(pos_rets))) + 1.0))


# Score1 = W_SAFETY·safety + W_SAFETY·max(0, safety)·quality − W_CRAWL·crawl.
# Like the Stage II fitness, driving quality only counts once the reward ranks
# survival above failure. Quality is a convex mix in [-1, 1]: throughput and
# overtakes rank whole successful episodes, the step term checks per-frame
# shaping. The weights are a design choice, not fitted (no labels to fit to).
_W_SAFETY = 0.5
_W_THROUGHPUT = 0.40
_W_OVERTAKE = 0.25
_W_STEP = 0.35
_W_CRAWL_PEN = 0.5
# Hard reject: PPO's discounted objective ranks failed episodes above survived
# ones more often than the reverse (e.g. a crash bonus, or a negative per-step
# reward, which makes ending the episode early pay).
PREFERS_CRASH_SAFETY = 0.0


def score_highway_dataset(
    reward_fn: Any,
    trajectories: Sequence[HighwayTrajectoryRecord],
    *,
    candidate_id: str = "",
) -> Score1Result:
    """
    Highway Score1: a cheap sanity filter for a candidate reward (in [-1, 1]).

    ::

        safety  = 2·AUC − 1 of success ≻ timeout ≻ collision on discounted returns (γ = PPO's)
        quality = 0.40·throughput + 0.25·overtake + 0.35·step_spearman
        Score1  = 0.5·safety + 0.5·max(0, safety)·quality − 0.5·crawl_penalty

    Rejected: ``no_signal`` (no component carries information) or
    ``prefers_crash`` (safety < PREFERS_CRASH_SAFETY).
    """
    del candidate_id  # reserved for logging / future per-candidate caches
    if not trajectories:
        return Score1Result(score=0.0, degenerate_fraction=1.0, rejected=True, reject_reason="empty")

    steps = {t.trajectory_id: _step_rewards(reward_fn, t) for t in trajectories}
    returns = {tid: _discounted_return(r) for tid, r in steps.items()}
    step_rho, deg_frac, n_trajs, n_deg = _step_spearman(steps, trajectories)
    safety = _preference_auc(returns, trajectories)
    thr = _throughput_alignment(returns, trajectories)
    ovt = _overtake_alignment(returns, trajectories)
    crawl = _crawl_penalty(returns, trajectories)

    quality = _W_THROUGHPUT * thr + _W_OVERTAKE * ovt + _W_STEP * step_rho
    raw = float(
        np.clip(
            _W_SAFETY * safety + _W_SAFETY * max(0.0, safety) * quality - _W_CRAWL_PEN * crawl,
            -1.0,
            1.0,
        )
    )

    reject_reason = None
    if deg_frac >= 0.95 and max(abs(safety), abs(thr), abs(ovt)) < 1e-9:
        reject_reason = "no_signal"
    elif safety < PREFERS_CRASH_SAFETY:
        reject_reason = "prefers_crash"

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
            "throughput": float(thr),
            "overtake_alignment": float(ovt),
            "crawl_penalty": float(crawl),
        },
    )


def smoke_trajectories() -> List[HighwayTrajectoryRecord]:
    """Fixture trajectories behind :func:`make_smoke_score_fn`."""

    def _traj(
        tid: str,
        scenario: str,
        label: str,
        frames: Sequence[HighwayRewardState],
    ) -> HighwayTrajectoryRecord:
        return HighwayTrajectoryRecord(
            trajectory_id=tid,
            scenario_id=scenario,
            seed=0,
            states=tuple(frames),
            label=label,
            behavior="smoke",
        )

    # Vary progress / x across frames so Spearman is defined.
    ok_frames = [
        HighwayRewardState(
            ego=EgoVehicle(
                x=float(i) * 5.0,
                y=0.0,
                vx=20.0 + float(i),
                vy=0.0,
                heading=0.0,
                speed=20.0 + float(i),
                lane_index=1.0,
                on_road=True,
            ),
            others=(),
            collision=False,
            off_road=False,
            timeout=False,
            action=1,
            time_step=0.2,
            global_time=0.2 * (i + 1),
            time_limit=40.0,
            progress=5.0,
            speed=20.0 + float(i),
        )
        for i in range(4)
    ]
    crash_frames = [
        HighwayRewardState(
            ego=EgoVehicle(
                x=float(i),
                y=0.0,
                vx=5.0,
                vy=0.0,
                heading=0.0,
                speed=5.0,
                lane_index=1.0,
                on_road=True,
            ),
            others=(),
            collision=(i == 2),
            off_road=False,
            timeout=False,
            action=1,
            time_step=0.2,
            global_time=0.2 * (i + 1),
            time_limit=40.0,
            progress=1.0,
            speed=5.0,
        )
        for i in range(3)
    ]
    # Label "timeout" = non-collision failure (off-road); state.timeout stays
    # False because the wrapper sets it only for survived, truncated episodes.
    to_frames = [
        HighwayRewardState(
            ego=EgoVehicle(
                x=float(i) * 2.0,
                y=0.0,
                vx=10.0,
                vy=0.0,
                heading=0.0,
                speed=10.0,
                lane_index=1.0,
                on_road=(i != 3),
            ),
            others=(),
            collision=False,
            off_road=(i == 3),
            timeout=False,
            action=1,
            time_step=0.2,
            global_time=0.2 * (i + 1),
            time_limit=40.0,
            progress=2.0,
            speed=10.0,
        )
        for i in range(4)
    ]
    return [
        _traj("smoke_ok", "s0", "success", ok_frames),
        _traj("smoke_ok2", "s0", "success", list(reversed(ok_frames))),
        _traj(
            "smoke_ok3",
            "s1",
            "success",
            [
                HighwayRewardState(
                    ego=EgoVehicle(
                        x=float(i) * 3.0,
                        y=0.0,
                        vx=15.0,
                        vy=0.0,
                        heading=0.0,
                        speed=15.0,
                        lane_index=1.0,
                        on_road=True,
                    ),
                    others=(),
                    collision=False,
                    off_road=False,
                    timeout=False,
                    action=1,
                    time_step=0.2,
                    global_time=0.2 * (i + 1),
                    time_limit=40.0,
                    progress=3.0 + 0.5 * float(i),
                    speed=15.0,
                )
                for i in range(5)
            ],
        ),
        _traj("smoke_crash", "s0", "collision", crash_frames),
        _traj("smoke_to", "s1", "timeout", to_frames),
    ]


def make_smoke_score_fn() -> Callable[..., Score1Result]:
    """Tiny fixture Score1 for --fast / smoke mode."""
    trajs = smoke_trajectories()

    def _fn(reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        result = score_highway_dataset(reward_fn, trajs, candidate_id=candidate_id)
        # Smoke fixture: never hard-reject; keep raw_score for ranking.
        if result.raw_score is not None:
            return Score1Result(
                score=float(result.raw_score),
                degenerate_fraction=result.degenerate_fraction,
                n_pairs=result.n_pairs,
                n_degenerate=result.n_degenerate,
                rejected=False,
                reject_reason=None,
                raw_score=result.raw_score,
                scenario_scores=result.scenario_scores,
            )
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
    action_settings: Optional[ActionModeSettings] = None,
) -> Tuple[Callable[..., Any], Optional[Any]]:
    """Score1 over the dataset with actions in ``action_settings``' mode (default: active)."""
    key = str(mode).strip().lower()
    if key == "smoke":
        logger.warning("Highway Stage I using smoke score (not real dataset)")
        return make_smoke_score_fn(), None
    if key != "dataset":
        raise ValueError(f"Unknown score1_mode: {mode!r}")
    path = resolve_dataset_path(dataset_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Highway Stage I dataset not found at {path}. "
            f"Run: python scripts/collect_highway_stage1_dataset.py --out {path}"
        )
    dataset = with_action_mode(load_highway_dataset(path), action_settings or get_action_settings())
    logger.info("Loaded highway Stage I dataset from %s (%d trajs)", path, len(dataset))

    def _fn(reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        return score_highway_dataset(reward_fn, dataset, candidate_id=candidate_id)

    return _fn, dataset

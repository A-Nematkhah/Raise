"""Highway Stage I — Score1 over highway trajectories (pack-local)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from crowd_nav.domains.highway.state import (
    EgoVehicle,
    HighwayRewardState,
    NearbyVehicle,
)
from crowd_nav.reward_search.rules import (
    TrajectoryCategory,
    label_to_category,
    rule_preference_score,
    spearman_correlation,
)
from crowd_nav.reward_search.scoring import Score1Result

logger = logging.getLogger(__name__)

DEFAULT_STAGE1_DATASET = "data/highway_stage1_dataset"


@dataclass
class HighwayTrajectoryRecord:
    trajectory_id: str
    scenario_id: str
    seed: Optional[int]
    states: Tuple[HighwayRewardState, ...]
    label: str  # success | collision | timeout
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

    @property
    def progress_proxy(self) -> float:
        """Higher = better progress (used like inverse dist_to_goal)."""
        last = self.states[-1]
        return float(last.ego.x) + 0.1 * float(last.speed)


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
    }
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_highway_dataset(path: str) -> List[HighwayTrajectoryRecord]:
    traj_path = os.path.join(path, "trajectories.jsonl")
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(traj_path)
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


def _rule_score_at_frame(traj: HighwayTrajectoryRecord, frame: int) -> float:
    idx = min(frame, len(traj.states) - 1)
    st = traj.states[idx]
    # Prefer progress: rule_preference uses -dist_goal for Other/Fail, so pass
    # negative progress so larger progress → better.
    dist_goal = -float(st.ego.x) - 0.1 * float(st.speed)
    nav = float(min(frame + 1, traj.length)) * float(st.time_step)
    return rule_preference_score(
        traj.category, nav_length=nav, dist_goal=dist_goal
    )


def _reward_cumulatives(reward_fn: Any, traj: HighwayTrajectoryRecord) -> List[float]:
    if hasattr(reward_fn, "reset"):
        reward_fn.reset()
    cum = 0.0
    vals: List[float] = []
    for st in traj.states:
        cum += float(reward_fn.compute(st))
        vals.append(cum)
    return vals


def score_highway_dataset(
    reward_fn: Any,
    trajectories: Sequence[HighwayTrajectoryRecord],
    *,
    candidate_id: str = "",
) -> Score1Result:
    """Mean Spearman(rule ranks, cumulative reward ranks) over trajectories."""
    if not trajectories:
        return Score1Result(score=0.0, degenerate_fraction=1.0, rejected=True, reject_reason="empty")

    pair_scores: List[float] = []
    n_deg = 0
    n_pairs = 0
    for traj in trajectories:
        n = len(traj.states)
        rules = [_rule_score_at_frame(traj, f) for f in range(n)]
        rewards = _reward_cumulatives(reward_fn, traj)
        # Pad freeze already implicit (one frame per state).
        rho = spearman_correlation(rules, rewards)
        n_pairs += 1
        if rho is None or (isinstance(rho, float) and (rho != rho)):
            n_deg += 1
            continue
        pair_scores.append(float(rho))

    deg_frac = float(n_deg) / float(max(1, n_pairs))
    if not pair_scores:
        return Score1Result(
            score=-1.0,
            degenerate_fraction=deg_frac,
            n_pairs=n_pairs,
            n_degenerate=n_deg,
            rejected=True,
            reject_reason="all_degenerate",
        )
    score = float(np.mean(pair_scores))
    rejected = deg_frac >= 0.5
    return Score1Result(
        score=-1.0 if rejected else score,
        degenerate_fraction=deg_frac,
        n_pairs=n_pairs,
        n_degenerate=n_deg,
        rejected=rejected,
        reject_reason="degenerate_fraction" if rejected else None,
        raw_score=score,
    )


def make_smoke_score_fn() -> Callable[..., Score1Result]:
    """Tiny fixture Score1 for --fast / smoke mode."""

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
                on_road=True,
            ),
            others=(),
            collision=False,
            off_road=False,
            timeout=(i == 3),
            action=1,
            time_step=0.2,
            global_time=0.2 * (i + 1),
            time_limit=0.8,
            progress=2.0,
            speed=10.0,
        )
        for i in range(4)
    ]
    trajs = [
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

    def _fn(reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        result = score_highway_dataset(reward_fn, trajs, candidate_id=candidate_id)
        # Smoke fixture: never hard-reject on degeneracy; keep raw_score for ranking.
        if result.raw_score is not None:
            return Score1Result(
                score=float(result.raw_score),
                degenerate_fraction=result.degenerate_fraction,
                n_pairs=result.n_pairs,
                n_degenerate=result.n_degenerate,
                rejected=False,
                reject_reason=None,
                raw_score=result.raw_score,
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
) -> Tuple[Callable[..., Any], Optional[Any]]:
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
    dataset = load_highway_dataset(path)
    logger.info("Loaded highway Stage I dataset from %s (%d trajs)", path, len(dataset))

    def _fn(reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        return score_highway_dataset(reward_fn, dataset, candidate_id=candidate_id)

    return _fn, dataset

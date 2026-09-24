"""
Stage I pre-collected trajectory dataset (load / save / records).

Each trajectory stores a sequence of Phase-1 ``RewardState`` snapshots plus a
final episode label. Stage I never reads env-logged reward scalars — candidates
recompute via ``reward_fn.compute(state)`` only.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from raise_core.rules import TrajectoryCategory
from domains.crowdnav.state import HumanObservable, RewardState, RobotRewardState

EpisodeLabel = str  # "success" | "collision" | "timeout"


@dataclass
class TrajectoryRecord:
    """
    One recorded episode for Stage I scoring.

    Field layout mirrors mobile_robot_env.trajectories.Trajectory closely
    enough for rule/tie-break logic (scenario_id, length, outcome stats) while
    storing CrowdNav ``RewardState`` frames instead of RewardContext arrays.
    """

    trajectory_id: str
    scenario_id: str
    seed: Optional[int]
    states: Tuple[RewardState, ...]
    label: EpisodeLabel
    behavior: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = tuple(self.states)
        if len(self.states) < 1:
            raise ValueError("TrajectoryRecord must contain at least one RewardState")
        self.label = str(self.label).strip().lower()
        if self.label in ("reachgoal", "reach_goal", "goal"):
            self.label = "success"
        elif self.label in ("collide", "fail"):
            self.label = "collision"
        elif self.label in ("time_out", "other", "truncation"):
            self.label = "timeout"
        if self.label not in ("success", "collision", "timeout"):
            raise ValueError(f"Invalid label: {self.label!r}")
        if self.metadata is None:
            self.metadata = {}

    @property
    def length(self) -> int:
        return len(self.states)

    @property
    def category(self) -> TrajectoryCategory:
        if self.label == "success":
            return TrajectoryCategory.SUCCESS
        if self.label == "collision":
            return TrajectoryCategory.FAIL
        return TrajectoryCategory.OTHER

    @property
    def nav_length(self) -> float:
        """Episode navigation length proxy: final global_time (or step count)."""
        last = self.states[-1]
        if last.global_time > 0:
            return float(last.global_time)
        return float(self.length) * float(last.time_step)

    def state_at(self, frame: int) -> RewardState:
        """Pad by holding the last available state (sister-project assumption)."""
        if frame < 0:
            raise IndexError(frame)
        if frame < len(self.states):
            return self.states[frame]
        return self.states[-1]


Stage1Dataset = Dict[str, List[TrajectoryRecord]]


def _action_to_json(action: Any) -> Any:
    if action is None:
        return None
    if hasattr(action, "_asdict"):
        return dict(action._asdict())
    if isinstance(action, (list, tuple)) and len(action) == 2:
        return {"vx": float(action[0]), "vy": float(action[1])}
    if isinstance(action, Mapping):
        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in action.items()}
    return {"repr": repr(action)}


def _action_from_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        if "vx" in obj and "vy" in obj:
            from crowd_sim.envs.utils.action import ActionXY

            return ActionXY(float(obj["vx"]), float(obj["vy"]))
        if "v" in obj and "r" in obj:
            from crowd_sim.envs.utils.action import ActionRot

            return ActionRot(float(obj["v"]), float(obj["r"]))
    return obj


def reward_state_to_dict(state: RewardState) -> Dict[str, Any]:
    return {
        "robot": asdict(state.robot),
        "humans": [asdict(h) for h in state.humans],
        "dmin": float(state.dmin),
        "discomfort_dist": float(state.discomfort_dist),
        "collision": bool(state.collision),
        "reaching_goal": bool(state.reaching_goal),
        "timeout": bool(state.timeout),
        "action": _action_to_json(state.action),
        "time_step": float(state.time_step),
        "global_time": float(state.global_time),
        "time_limit": float(state.time_limit),
    }


def reward_state_from_dict(d: Mapping[str, Any]) -> RewardState:
    robot = RobotRewardState(**{k: float(d["robot"][k]) for k in RobotRewardState.__dataclass_fields__})
    humans = tuple(
        HumanObservable(
            px=float(h["px"]),
            py=float(h["py"]),
            vx=float(h["vx"]),
            vy=float(h["vy"]),
            radius=float(h["radius"]),
        )
        for h in d.get("humans", ())
    )
    return RewardState(
        robot=robot,
        humans=humans,
        dmin=float(d["dmin"]),
        discomfort_dist=float(d["discomfort_dist"]),
        collision=bool(d["collision"]),
        reaching_goal=bool(d["reaching_goal"]),
        timeout=bool(d["timeout"]),
        action=_action_from_json(d.get("action")),
        time_step=float(d["time_step"]),
        global_time=float(d["global_time"]),
        time_limit=float(d["time_limit"]),
    )


def trajectory_to_dict(traj: TrajectoryRecord) -> Dict[str, Any]:
    return {
        "trajectory_id": traj.trajectory_id,
        "scenario_id": traj.scenario_id,
        "seed": traj.seed,
        "label": traj.label,
        "behavior": traj.behavior,
        "metadata": dict(traj.metadata),
        "states": [reward_state_to_dict(s) for s in traj.states],
    }


def trajectory_from_dict(d: Mapping[str, Any]) -> TrajectoryRecord:
    states = tuple(reward_state_from_dict(s) for s in d["states"])
    return TrajectoryRecord(
        trajectory_id=str(d["trajectory_id"]),
        scenario_id=str(d["scenario_id"]),
        seed=d.get("seed"),
        states=states,
        label=str(d["label"]),
        behavior=str(d.get("behavior") or ""),
        metadata=dict(d.get("metadata") or {}),
    )


def save_stage1_dataset(
    dataset: Stage1Dataset,
    path: str,
    *,
    fmt: str = "npz",
) -> None:
    """
    Persist dataset.

    ``fmt='jsonl'``: directory of ``scenario_{id}.jsonl`` (one traj JSON per line).
    ``fmt='npz'`` (default): single gzip+pickle archive (much smaller on disk).
    """
    fmt = fmt.lower()
    if fmt == "jsonl":
        os.makedirs(path, exist_ok=True)
        meta = {"n_scenarios": len(dataset), "format": "jsonl"}
        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
        for sid, trajs in dataset.items():
            if sid.startswith("scenario_"):
                fname = f"{sid}.jsonl"
            else:
                try:
                    fname = f"scenario_{int(sid):03d}.jsonl"
                except ValueError:
                    fname = f"scenario_{sid}.jsonl"
            fpath = os.path.join(path, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                for traj in trajs:
                    f.write(json.dumps(trajectory_to_dict(traj), sort_keys=True) + "\n")
        return
    if fmt in ("npz", "npz.gz", "pkl"):
        import gzip
        import pickle

        archive_exts = (".npz", ".npz.gz", ".pkl", ".pkl.gz")
        # Bare paths (no archive suffix) are treated as directories even if missing.
        if any(path.endswith(ext) for ext in archive_exts):
            out = path
            parent = os.path.dirname(os.path.abspath(out))
            if parent:
                os.makedirs(parent, exist_ok=True)
        else:
            os.makedirs(path, exist_ok=True)
            out = os.path.join(path, "stage1_dataset.npz")
        payload = {sid: [trajectory_to_dict(t) for t in trajs] for sid, trajs in dataset.items()}
        with gzip.open(out, "wb", compresslevel=6) as gz:
            pickle.dump(payload, gz, protocol=4)
        man = os.path.join(os.path.dirname(out) or ".", "manifest.json")
        with open(man, "w", encoding="utf-8") as f:
            json.dump(
                {"n_scenarios": len(dataset), "format": "npz", "path": os.path.basename(out)},
                f,
                indent=2,
            )
            f.write("\n")
        return
    raise ValueError(f"Unknown fmt: {fmt}")


def load_stage1_dataset(path: str) -> Stage1Dataset:
    """
    Load ``dict[scenario_id, list[TrajectoryRecord]]`` from a jsonl directory
    or a ``.npz`` / ``.pkl`` archive.
    """
    if os.path.isdir(path):
        # Prefer compact archive inside the directory if present.
        for name in ("stage1_dataset.npz", "dataset.npz", "stage1.pkl.gz"):
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                return load_stage1_dataset(cand)
        out: Stage1Dataset = {}
        for name in sorted(os.listdir(path)):
            if not name.endswith(".jsonl"):
                continue
            sid = name[: -len(".jsonl")]
            trajs: List[TrajectoryRecord] = []
            with open(os.path.join(path, name), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    trajs.append(trajectory_from_dict(json.loads(line)))
            if trajs:
                key = trajs[0].scenario_id or sid
                out[key] = trajs
        if not out:
            raise FileNotFoundError(f"No scenario_*.jsonl / stage1_dataset.npz under {path}")
        return out

    if path.endswith(".npz") or path.endswith(".npz.gz") or path.endswith(".pkl") or path.endswith(".pkl.gz"):
        import gzip
        import pickle

        with gzip.open(path, "rb") as gz:
            try:
                payload = pickle.load(gz)
            except OSError:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
        return {
            str(sid): [trajectory_from_dict(t) for t in trajs]
            for sid, trajs in payload.items()
        }

    raise FileNotFoundError(f"Stage I dataset not found: {path}")


def build_synthetic_scenario(
    scenario_id: str,
    trajectories: Sequence[TrajectoryRecord],
) -> List[TrajectoryRecord]:
    """Helper for unit tests — validates and returns a scenario list."""
    out = []
    for t in trajectories:
        if t.scenario_id != scenario_id:
            out.append(
                TrajectoryRecord(
                    trajectory_id=t.trajectory_id,
                    scenario_id=scenario_id,
                    seed=t.seed,
                    states=t.states,
                    label=t.label,
                    behavior=t.behavior,
                    metadata=t.metadata,
                )
            )
        else:
            out.append(t)
    return out


def split_stage1_dataset(
    dataset: Stage1Dataset,
    *,
    holdout_fraction: float = 0.3,
    seed: int = 425,
) -> Tuple[Stage1Dataset, Stage1Dataset]:
    """
    Split by **scenario id** (keeps within-scenario traj groups intact).

    With fewer than 2 scenarios, returns ``(dataset, {})`` — no holdout.
    Always keeps at least one scenario in train when a holdout is created.
    """
    import random

    if not dataset:
        return {}, {}
    frac = float(holdout_fraction)
    if frac <= 0.0:
        return dict(dataset), {}
    if frac >= 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")

    ids = sorted(str(k) for k in dataset.keys())
    if len(ids) < 2:
        return dict(dataset), {}

    rng = random.Random(int(seed))
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_hold = max(1, int(round(len(shuffled) * frac)))
    n_hold = min(n_hold, len(shuffled) - 1)
    hold_ids = set(shuffled[:n_hold])
    train = {k: list(v) for k, v in dataset.items() if str(k) not in hold_ids}
    holdout = {k: list(v) for k, v in dataset.items() if str(k) in hold_ids}
    return train, holdout

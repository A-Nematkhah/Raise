"""EvoNav Score_1 for CrowdNav++ Stage I (training-free ranking agreement).

Score_1(r) = (1/M) * sum_j (1/F_j) * sum_f Spearman(rank_rules[j,f], rank_r[j,f])

No policy training. NumPy only (scipy optional as a test oracle).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from domains.crowdnav.stage1 import (
    CrowdNavTrajectoryRecord,
    load_crowdnav_dataset,
    resolve_dataset_path,
)
from raise_core.rules import _rankdata, spearman_correlation
from raise_core.scoring import Score1Result

logger = logging.getLogger(__name__)

# Class offsets so within-class terms (metres-scale) cannot cross class boundaries.
_OFFSET_SUCCESS = 2.0e6
_OFFSET_OTHER = 1.0e6
_OFFSET_FAIL = 0.0


@dataclass
class Score1Config:
    """Tunables for EvoNav Score_1."""

    nan_policy: str = "zero"  # zero | skip
    reject_score: float = float("-inf")
    n_workers: int = 1
    # Documented rule-score offsets (see ``rule_score_at_frame``).
    offset_success: float = _OFFSET_SUCCESS
    offset_other: float = _OFFSET_OTHER
    offset_fail: float = _OFFSET_FAIL

    def __post_init__(self) -> None:
        key = str(self.nan_policy).strip().lower()
        if key not in ("zero", "skip"):
            raise ValueError("nan_policy must be 'zero' or 'skip'")
        self.nan_policy = key
        if int(self.n_workers) < 1:
            raise ValueError("n_workers must be >= 1")


# ------------------------------------------------------------------ adapters


def frame_outcome_features(
    traj: CrowdNavTrajectoryRecord, frame_index: int
) -> Tuple[str, float, float]:
    """Map dataset fields → ``(class, path_length, dist_to_goal)`` at ``frame_index``.

    Class priority (EvoNav): ``success`` > ``other`` (running/timeout) > ``fail``
    (collision). Early termination: if ``frame_index`` is past the last stored
    state, carry forward the final frame's class and geometry.

    ``path_length`` is the cumulative robot travel distance from frame 0 through
    the effective frame (not stored on disk; derived from positions).
    ``dist_to_goal`` is ``CrowdNavRewardState.goal_distance`` at that frame.
    """
    states = traj.states
    if not states:
        return "other", 0.0, float("inf")
    f = int(frame_index)
    if f < 0:
        f = 0
    # Effective index with carry-forward past termination.
    end = len(states) - 1
    use = end if f > end else f

    # Outcome class: first terminal flag at or before ``use``.
    cls = "other"
    for st in states[: use + 1]:
        if st.reached_goal:
            cls = "success"
            break
        if st.collision:
            cls = "fail"
            break
    if cls == "other" and states[use].timeout:
        cls = "other"

    path = 0.0
    prev = states[0].robot
    for st in states[1 : use + 1]:
        path += math.hypot(st.robot.px - prev.px, st.robot.py - prev.py)
        prev = st.robot
    dist = float(states[use].goal_distance)
    return cls, float(path), dist


def rule_score_at_frame(
    traj: CrowdNavTrajectoryRecord,
    frame_index: int,
    *,
    config: Optional[Score1Config] = None,
) -> float:
    """Scalar rule score at one frame (higher = better).

    Formula
    -------
    class offsets: success = ``offset_success``, other = ``offset_other``,
    fail = ``offset_fail`` (defaults 2e6 / 1e6 / 0).

    within-class term (always subtracted so shorter is better):
      success: ``-path_length``
      other / fail: ``-dist_to_goal``

    ``rule_score = offset[class] + within``. Offsets dominate metre-scale
    within terms so class order is absolute; ties inside a class follow path
    or distance. Average ranks are applied later via ``_rankdata``.
    """
    cfg = config or Score1Config()
    cls, path_len, dist = frame_outcome_features(traj, frame_index)
    if cls == "success":
        return float(cfg.offset_success) - path_len
    if cls == "fail":
        return float(cfg.offset_fail) - dist
    return float(cfg.offset_other) - dist


def _group_scenarios(
    trajectories: Sequence[CrowdNavTrajectoryRecord],
) -> List[Tuple[str, List[CrowdNavTrajectoryRecord]]]:
    order: List[str] = []
    buckets: Dict[str, List[CrowdNavTrajectoryRecord]] = {}
    for traj in trajectories:
        sid = str(traj.scenario_id)
        if sid not in buckets:
            order.append(sid)
            buckets[sid] = []
        buckets[sid].append(traj)
    return [(sid, buckets[sid]) for sid in order]


# ------------------------------------------------------------------ RuleRanker


@dataclass
class RuleRanker:
    """Precomputes ``rank_rules[j, f, n]`` once per dataset."""

    scenario_ids: Tuple[str, ...]
    rule_ranks: np.ndarray  # [M, F_max, N_max]
    valid_mask: np.ndarray  # [M, F_max] True where frame exists for the scenario
    n_traj: np.ndarray  # [M]
    f_len: np.ndarray  # [M]
    traj_ids: Tuple[Tuple[str, ...], ...]
    config: Score1Config = field(default_factory=Score1Config)

    @classmethod
    def build(
        cls,
        trajectories: Sequence[CrowdNavTrajectoryRecord],
        *,
        config: Optional[Score1Config] = None,
    ) -> "RuleRanker":
        cfg = config or Score1Config()
        groups = _group_scenarios(trajectories)
        if not groups:
            empty = np.zeros((0, 0, 0), dtype=np.float64)
            return cls(
                scenario_ids=(),
                rule_ranks=empty,
                valid_mask=np.zeros((0, 0), dtype=bool),
                n_traj=np.zeros(0, dtype=np.int32),
                f_len=np.zeros(0, dtype=np.int32),
                traj_ids=(),
                config=cfg,
            )
        n_max = max(len(g) for _, g in groups)
        f_max = max(max(t.length for t in g) for _, g in groups)
        m = len(groups)
        ranks = np.full((m, f_max, n_max), np.nan, dtype=np.float64)
        valid = np.zeros((m, f_max), dtype=bool)
        n_traj = np.zeros(m, dtype=np.int32)
        f_len = np.zeros(m, dtype=np.int32)
        ids: List[Tuple[str, ...]] = []
        sids: List[str] = []
        for j, (sid, group) in enumerate(groups):
            sids.append(sid)
            ids.append(tuple(t.trajectory_id for t in group))
            n = len(group)
            n_traj[j] = n
            # Scenario horizon: max trajectory length (carry-forward fills shorter).
            fj = max(t.length for t in group)
            f_len[j] = fj
            valid[j, :fj] = True
            scores = np.empty((n, fj), dtype=np.float64)
            for n_i, traj in enumerate(group):
                for f in range(fj):
                    scores[n_i, f] = rule_score_at_frame(traj, f, config=cfg)
            for f in range(fj):
                ranks[j, f, :n] = _rankdata(scores[:, f])
        return cls(
            scenario_ids=tuple(sids),
            rule_ranks=ranks,
            valid_mask=valid,
            n_traj=n_traj,
            f_len=f_len,
            traj_ids=tuple(ids),
            config=cfg,
        )


# ------------------------------------------------------------------ rewards / ranks


def _step_rewards(reward_fn: Any, traj: CrowdNavTrajectoryRecord) -> np.ndarray:
    if hasattr(reward_fn, "reset"):
        reward_fn.reset()
    out = np.empty(traj.length, dtype=np.float64)
    for i, st in enumerate(traj.states):
        val = reward_fn.compute(st)
        out[i] = float(val)
    return out


def _cum_reward_matrix(
    reward_fn: Any,
    group: Sequence[CrowdNavTrajectoryRecord],
    f_len: int,
) -> np.ndarray:
    """``[N, F]`` cumulative reward with carry-forward after early termination."""
    n = len(group)
    cum = np.zeros((n, f_len), dtype=np.float64)
    for i, traj in enumerate(group):
        steps = _step_rewards(reward_fn, traj)
        if steps.size == 0:
            continue
        c = np.cumsum(steps)
        L = c.size
        take = min(L, f_len)
        cum[i, :take] = c[:take]
        if take < f_len:
            cum[i, take:] = c[take - 1]
    return cum


def _spearman_frame(rule_ranks: np.ndarray, reward_ranks: np.ndarray) -> float:
    return float(spearman_correlation(rule_ranks, reward_ranks))


# ------------------------------------------------------------------ evaluator


@dataclass
class Score1Evaluator:
    """Caches rule ranks; scores candidate rewards with EvoNav Score_1."""

    trajectories: Sequence[CrowdNavTrajectoryRecord]
    config: Score1Config = field(default_factory=Score1Config)
    _ranker: RuleRanker = field(init=False, repr=False)
    _groups: List[Tuple[str, List[CrowdNavTrajectoryRecord]]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._groups = _group_scenarios(self.trajectories)
        self._ranker = RuleRanker.build(self.trajectories, config=self.config)

    def evaluate(self, reward_fn: Any, *, candidate_id: str = "") -> Score1Result:
        del candidate_id
        try:
            return self._evaluate_unsafe(reward_fn)
        except Exception as exc:  # noqa: BLE001
            return Score1Result(
                score=float(self.config.reject_score),
                per_scenario=None,
                n_nan_frames=0,
                n_frames=0,
                error=f"{type(exc).__name__}: {exc}"[:500],
                rejected=True,
                reject_reason=f"score1_error: {type(exc).__name__}",
            )

    def _evaluate_unsafe(self, reward_fn: Any) -> Score1Result:
        if not self._groups:
            return Score1Result(score=0.0, n_nan_frames=0, n_frames=0)

        per_scenario: Dict[str, float] = {}
        scenario_means: List[float] = []
        n_nan = 0
        n_frames = 0
        ranker = self._ranker

        for j, (sid, group) in enumerate(self._groups):
            fj = int(ranker.f_len[j])
            n = int(ranker.n_traj[j])
            if fj < 1 or n < 2:
                per_scenario[sid] = 0.0
                scenario_means.append(0.0)
                continue
            cum = _cum_reward_matrix(reward_fn, group, fj)
            if not np.all(np.isfinite(cum)):
                # Treat non-finite rewards as a hard error for this candidate.
                raise ValueError("non-finite reward value in Score_1 rollouts")
            rhos: List[float] = []
            for f in range(fj):
                n_frames += 1
                rr = ranker.rule_ranks[j, f, :n]
                rw = _rankdata(cum[:, f])
                rho = _spearman_frame(rr, rw)
                if not math.isfinite(rho):
                    n_nan += 1
                    if self.config.nan_policy == "zero":
                        rhos.append(0.0)
                    # skip: omit frame from the scenario mean
                    continue
                rhos.append(float(rho))
            if not rhos:
                mean_j = 0.0
            else:
                mean_j = float(np.mean(rhos))
            per_scenario[sid] = mean_j
            scenario_means.append(mean_j)

        score = float(np.mean(scenario_means)) if scenario_means else 0.0
        return Score1Result(
            score=score,
            per_scenario=per_scenario,
            n_nan_frames=int(n_nan),
            n_frames=int(n_frames),
            error=None,
            rejected=False,
            reject_reason=None,
        )

    def evaluate_population(
        self,
        candidates: Sequence[Any],
        *,
        n_workers: Optional[int] = None,
        reward_fn_of: Optional[Callable[[Any], Any]] = None,
        id_of: Optional[Callable[[Any], str]] = None,
    ) -> List[Tuple[Any, Score1Result]]:
        """Score candidates; return ``(candidate, result)`` sorted by score descending."""
        workers = int(n_workers if n_workers is not None else self.config.n_workers)
        get_fn = reward_fn_of or (lambda c: c)
        get_id = id_of or (lambda c: str(getattr(c, "candidate_id", id(c))))
        results: List[Tuple[Any, Score1Result]] = []
        if workers <= 1 or len(candidates) <= 1:
            for cand in candidates:
                results.append(
                    (cand, self.evaluate(get_fn(cand), candidate_id=get_id(cand)))
                )
        else:
            # Process pool needs picklable callables; fall back to serial on failure.
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futs = {
                        pool.submit(
                            _evaluate_pickled, self.trajectories, self.config, get_fn(cand)
                        ): cand
                        for cand in candidates
                    }
                    for fut in as_completed(futs):
                        cand = futs[fut]
                        results.append((cand, fut.result()))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Score_1 population parallel eval failed (%s); using serial",
                    exc,
                )
                results = [
                    (cand, self.evaluate(get_fn(cand), candidate_id=get_id(cand)))
                    for cand in candidates
                ]
        results.sort(key=lambda pair: float(pair[1].score), reverse=True)
        return results


def _evaluate_pickled(
    trajectories: Sequence[CrowdNavTrajectoryRecord],
    config: Score1Config,
    reward_fn: Any,
) -> Score1Result:
    return Score1Evaluator(trajectories, config=config).evaluate(reward_fn)


def score_crowdnav_dataset(
    reward_fn: Any,
    trajectories: Sequence[CrowdNavTrajectoryRecord],
    *,
    candidate_id: str = "",
    config: Optional[Score1Config] = None,
) -> Score1Result:
    """Public entry used by the pack / timeout worker (EvoNav Score_1 only)."""
    return Score1Evaluator(trajectories, config=config or Score1Config()).evaluate(
        reward_fn, candidate_id=candidate_id
    )


# ------------------------------------------------------------------ CLI


def _load_candidates(path: str) -> List[Tuple[str, str]]:
    """Return ``[(id, source_code), ...]`` from a JSON file or directory of ``*.py``."""
    path = os.path.abspath(path)
    out: List[Tuple[str, str]] = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if not name.endswith(".py"):
                continue
            fp = os.path.join(path, name)
            out.append((os.path.splitext(name)[0], open(fp, encoding="utf-8").read()))
        return out
    data = json.loads(open(path, encoding="utf-8").read())
    if isinstance(data, Mapping):
        data = data.get("candidates") or data.get("rewards") or []
    for i, row in enumerate(data):
        if isinstance(row, str):
            out.append((f"cand_{i}", row))
        else:
            out.append(
                (
                    str(row.get("id") or row.get("candidate_id") or f"cand_{i}"),
                    str(row.get("code") or row.get("source") or ""),
                )
            )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EvoNav Score_1 ranked table")
    parser.add_argument("--dataset", required=True, help="Stage I dataset directory")
    parser.add_argument(
        "--candidates",
        required=True,
        help="JSON list/object or directory of .py reward files",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--nan-policy", choices=("zero", "skip"), default="zero"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    from domains.crowdnav.settings import get_settings
    from domains.crowdnav.stage1 import with_settings
    from raise_core.sandbox import SandboxedReward, compile_compute_reward
    from domains.crowdnav.sandbox_config import crowdnav_sandbox_config

    ds_path = resolve_dataset_path(args.dataset)
    trajs = with_settings(load_crowdnav_dataset(ds_path), get_settings())
    cfg = Score1Config(nan_policy=args.nan_policy, n_workers=max(1, int(args.workers)))
    evaluator = Score1Evaluator(trajs, config=cfg)
    sandbox_cfg = crowdnav_sandbox_config()

    rows: List[Tuple[str, Score1Result]] = []
    for cid, code in _load_candidates(args.candidates):
        try:
            fn = SandboxedReward(
                compile_compute_reward(code, sandbox_cfg),
                sandbox_cfg,
                source_code=code,
            )
            result = evaluator.evaluate(fn, candidate_id=cid)
        except Exception as exc:  # noqa: BLE001
            result = Score1Result(
                score=float(cfg.reject_score),
                rejected=True,
                reject_reason="score1_error",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        rows.append((cid, result))
    rows.sort(key=lambda r: float(r[1].score), reverse=True)

    print(f"{'candidate':<24} {'Score_1':>10} {'n_nan':>8} {'error'}")
    print("-" * 72)
    for cid, res in rows:
        err = (res.error or res.reject_reason or "")[:40]
        print(f"{cid:<24} {res.score:10.4f} {res.n_nan_frames:8d} {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

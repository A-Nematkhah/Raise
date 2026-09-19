"""
Stage I analytical Score1 over a pre-collected trajectory dataset.

Score1(r) = mean_j mean_f Spearman(rank_rules(j,f), rank_reward(j,f))

Reward ranks use cumulative sum(reward_fn.compute(state)) — never env-logged
reward scalars. ``make_smoke_score_fn`` remains only as an opt-in fast-test
fixture (``--score1 smoke`` / ``fast`` profile).

Padding policy (baseline lock + Figure 3 fidelity):
  After a trajectory's real last frame, cumulative reward is **frozen** (no
  further ``compute`` on repeated terminal states). For Success rule
  tie-breaks, ``nav_length = min(f + 1, traj.length)``: no leak of a longer
  episode's *future* length into early frames, but once a short Success has
  ended its length stays short so Figure 3 (Success short ≻ Success long)
  can differentiate at later frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from crowd_nav.reward_search.dataset import Stage1Dataset, TrajectoryRecord
from crowd_nav.reward_search.rules import (
    dist_to_goal,
    rule_preference_score,
    spearman_correlation,
)
from crowd_nav.reward_search.state import RewardFunction


class Score1Fn(Protocol):
    def __call__(
        self, reward_fn: RewardFunction, *, candidate_id: str = ""
    ) -> "Score1Result":
        ...


@dataclass(frozen=True)
class Score1Result:
    """Score1 scalar plus degeneracy diagnostics for reporting."""

    score: float
    degenerate_fraction: float
    n_pairs: int = 0
    n_degenerate: int = 0

    def __float__(self) -> float:
        return float(self.score)

    def __gt__(self, other: object) -> bool:
        return float(self) > float(other)  # type: ignore[arg-type]

    def __ge__(self, other: object) -> bool:
        return float(self) >= float(other)  # type: ignore[arg-type]

    def __lt__(self, other: object) -> bool:
        return float(self) < float(other)  # type: ignore[arg-type]

    def __le__(self, other: object) -> bool:
        return float(self) <= float(other)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Score1Result):
            return self.as_dict() == other.as_dict()
        if isinstance(other, (int, float)):
            return float(self) == float(other)
        return NotImplemented

    def as_dict(self) -> Dict[str, Any]:
        return {
            "score": float(self.score),
            "degenerate_fraction": float(self.degenerate_fraction),
            "n_pairs": int(self.n_pairs),
            "n_degenerate": int(self.n_degenerate),
        }


def _cumulative_reward(
    reward_fn: RewardFunction,
    traj: TrajectoryRecord,
    *,
    max_frame: int,
) -> List[float]:
    """
    Prefix sums of recomputed rewards for frames 0..max_frame inclusive.

    For frames beyond the trajectory's real length, the last real cumulative
    value is repeated — ``compute()`` is **not** called on padded states.
    Resets the reward function once at the start of the trajectory.
    """
    reward_fn.reset()
    totals: List[float] = []
    running = 0.0
    n_pad = max_frame + 1
    real_len = int(traj.length)
    for f in range(n_pad):
        if f < real_len:
            state = traj.state_at(f)
            try:
                running += float(reward_fn.compute(state))
                totals.append(running)
            except Exception:  # noqa: BLE001
                totals.append(float("nan"))
        else:
            # Freeze: do not accumulate on repeated terminal / padded states.
            if totals:
                totals.append(totals[-1])
            else:
                totals.append(float("nan"))
    return totals


def _scenario_frame_correlations(
    trajs: Sequence[TrajectoryRecord],
    reward_fn: RewardFunction,
) -> Tuple[List[float], int, int]:
    """
    Spearman ρ for each frame f = 0..F_j-1 within one scenario.

    Returns ``(finite_correlations, n_pairs_evaluated, n_degenerate)`` where
    ``n_degenerate`` counts frames where Spearman returned NaN (e.g. constant
    reward ranks across trajectories).
    """
    if len(trajs) < 2:
        return [], 0, 0
    f_max = max(t.length for t in trajs)
    categories = [t.category for t in trajs]

    # Precompute cumulative rewards per trajectory (reset between trajs).
    cumuls: List[List[float]] = []
    for traj in trajs:
        cumuls.append(_cumulative_reward(reward_fn, traj, max_frame=f_max - 1))

    corrs: List[float] = []
    n_pairs = 0
    n_degenerate = 0
    for f in range(f_max):
        rule_scores = []
        reward_scores = []
        ok = True
        for i, traj in enumerate(trajs):
            state = traj.state_at(f)
            d_goal = dist_to_goal(
                state.robot.px, state.robot.py, state.robot.gx, state.robot.gy
            )
            # Steps so far, capped at real traj length (Figure 3 short≻long
            # after a short Success ends; no future-length leak while running).
            nav_so_far = float(min(f + 1, int(traj.length)))
            rule_scores.append(
                rule_preference_score(
                    categories[i], nav_length=nav_so_far, dist_goal=d_goal
                )
            )
            val = cumuls[i][f]
            if not math.isfinite(val):
                ok = False
                break
            reward_scores.append(val)
        if not ok:
            continue
        n_pairs += 1
        rho = spearman_correlation(rule_scores, reward_scores)
        if math.isfinite(rho):
            corrs.append(float(rho))
        else:
            n_degenerate += 1
    return corrs, n_pairs, n_degenerate


def score1_for_dataset(
    dataset: Union[Stage1Dataset, RewardFunction],
    reward_fn: Optional[RewardFunction] = None,
    *,
    candidate_id: str = "",
) -> Score1Result:
    """
    Analytical Score1 over pre-collected scenarios (EvoNav Eq. 1 / Figure 3).

    Call as ``score1_for_dataset(dataset, reward_fn)``. Scenarios with fewer
    than 2 trajectories are skipped. Raises ``ValueError`` only if every
    scenario is unusable.

    ``degenerate_fraction`` is the fraction of evaluated (scenario, frame)
    pairs where Spearman was undefined (typically constant reward ranks).
    """
    del candidate_id  # reserved for logging / future per-id caches
    # Back-compat: accidental score1_for_dataset(reward_fn) without dataset.
    if reward_fn is None:
        if isinstance(dataset, RewardFunction):
            raise ValueError(
                "score1_for_dataset requires a loaded Stage I dataset as the "
                "first argument: score1_for_dataset(dataset, reward_fn). "
                "Use make_score1_fn(dataset) for StageIEvolver.score_fn."
            )
        raise TypeError("reward_fn is required")

    if not isinstance(dataset, dict) or not dataset:
        raise ValueError("dataset must be a non-empty dict[scenario_id, trajectories]")

    scenario_means: List[float] = []
    n_pairs_total = 0
    n_degenerate_total = 0
    for _sid, trajs in dataset.items():
        usable = [t for t in trajs if t.length >= 1]
        if len(usable) < 2:
            continue
        frame_corrs, n_pairs, n_deg = _scenario_frame_correlations(usable, reward_fn)
        n_pairs_total += n_pairs
        n_degenerate_total += n_deg
        if not frame_corrs:
            continue
        scenario_means.append(float(np.mean(frame_corrs)))

    if not scenario_means:
        raise ValueError(
            "score1_for_dataset: no scoreable scenarios "
            "(need ≥2 trajectories with finite frame correlations)."
        )
    score = float(np.mean(scenario_means))
    # Spearman is in [-1, 1]; keep that contract even with float noise.
    score = float(max(-1.0, min(1.0, score)))
    if n_pairs_total > 0:
        degenerate_fraction = float(n_degenerate_total) / float(n_pairs_total)
    else:
        degenerate_fraction = 0.0
    return Score1Result(
        score=score,
        degenerate_fraction=degenerate_fraction,
        n_pairs=n_pairs_total,
        n_degenerate=n_degenerate_total,
    )


def make_score1_fn(dataset: Stage1Dataset) -> Callable[..., Score1Result]:
    """Bind a loaded dataset once for StageIEvolver.score_fn."""

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> Score1Result:
        return score1_for_dataset(dataset, reward_fn, candidate_id=candidate_id)

    return _fn


def make_constant_score_fn(scores: dict) -> Callable[..., Score1Result]:
    """Test helper: map candidate_id -> score (missing ids get -inf)."""

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> Score1Result:
        del reward_fn
        value = float(scores.get(candidate_id, float("-inf")))
        return Score1Result(score=value, degenerate_fraction=0.0)

    return _fn


def make_smoke_score_fn() -> Callable[..., Score1Result]:
    """
    Opt-in fast-test fixture only (``--score1 smoke`` / ``--fast``).

    Not used by the real Stage I loop. Prefer ``make_score1_fn(dataset)``.
    """
    from crowd_nav.reward_search.sandbox.runtime import default_smoke_states

    states = default_smoke_states()

    def _fn(reward_fn: RewardFunction, *, candidate_id: str = "") -> Score1Result:
        del candidate_id
        vals = []
        for s in states:
            try:
                vals.append(float(reward_fn.compute(s)))
            except Exception:  # noqa: BLE001
                return Score1Result(score=float("-inf"), degenerate_fraction=1.0)
        if not vals:
            return Score1Result(score=float("-inf"), degenerate_fraction=1.0)
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(len(vals), 1)
        polarity = (vals[0] - vals[1]) if len(vals) >= 2 else 0.0
        return Score1Result(
            score=float(var + 0.1 * polarity),
            degenerate_fraction=0.0,
        )

    return _fn

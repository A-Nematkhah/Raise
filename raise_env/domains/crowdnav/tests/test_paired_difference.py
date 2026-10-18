"""Paired per-case fitness differences (P1-2)."""

from __future__ import annotations

from domains.crowdnav.metrics import paired_difference
from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.candidate import RewardCandidate
from raise_core.selection import statistical_ties


def test_paired_se_tighter_than_independent():
    # Shared case effects: independent SE overstates uncertainty.
    eps_a = [
        {"case": i, "selection_score": 0.10 + 0.05 * (i % 3)} for i in range(20)
    ]
    eps_b = [
        {
            "case": i,
            "selection_score": row["selection_score"] + 0.02 + 0.001 * (i % 2),
        }
        for i, row in enumerate(eps_a)
    ]
    paired = paired_difference(eps_a, eps_b)
    assert paired is not None
    assert paired["n"] == 20
    assert paired["mean"] > 0
    # Independent SE of the two means.
    import math

    import numpy as np

    a = np.asarray([e["selection_score"] for e in eps_a], dtype=float)
    b = np.asarray([e["selection_score"] for e in eps_b], dtype=float)
    se_ind = math.hypot(a.std(ddof=1) / math.sqrt(len(a)), b.std(ddof=1) / math.sqrt(len(b)))
    assert paired["se"] < se_ind


def test_paired_difference_fallback_when_cases_differ():
    eps_a = [
        {"case": 0, "selection_score": 0.1},
        {"case": 1, "selection_score": 0.2},
    ]
    eps_b = [
        {"case": 0, "selection_score": 0.3},
        {"case": 9, "selection_score": 0.4},
    ]
    assert paired_difference(eps_a, eps_b) is None


def test_statistical_ties_uses_paired_band_when_episodes_match():
    validator = make_crowdnav_validator()
    code = "def compute_reward(state, memory):\n    return 0.0, {}\n"
    fn, err = validator.try_validate(code)
    assert fn is not None, err
    cases = list(range(30))
    shared = [0.1 * ((i * 7) % 5) for i in cases]

    def _from_returns(cid: str, returns, se: float) -> RewardCandidate:
        eps = [
            {"case": c, "selection_score": float(v)}
            for c, v in zip(cases, returns)
        ]
        fit = sum(returns) / len(returns)
        return RewardCandidate(
            candidate_id=cid,
            code=code,
            reward_fn=fn,
            valid=True,
            metadata={
                "fitness": fit,
                "last_metrics": {
                    "fitness": fit,
                    "fitness_se": se,
                    "holdout_episodes": eps,
                },
            },
        )

    # Large independent SE would mark everyone a tie; paired SE separates far.
    import numpy as np

    rng = np.random.default_rng(0)
    noise_b = rng.normal(0.0, 0.02, size=len(cases))
    noise_c = rng.normal(0.0, 0.02, size=len(cases))
    noise_f = rng.normal(0.0, 0.02, size=len(cases))
    best_r = [s + 0.05 + n for s, n in zip(shared, noise_b)]
    # Mean gap ~0.002 with paired residual sd ~0.028 → within 2 SE.
    close_r = [s + 0.048 + n for s, n in zip(shared, noise_c)]
    # Mean gap ~0.05 >> paired SE → not a tie.
    far_r = [s + 0.00 + n for s, n in zip(shared, noise_f)]
    best = _from_returns("best", best_r, se=1.0)
    close = _from_returns("close", close_r, se=1.0)
    far = _from_returns("far", far_r, se=1.0)
    ties = statistical_ties([best, close, far], best, z=2.0)
    assert "close" in ties
    assert "far" not in ties

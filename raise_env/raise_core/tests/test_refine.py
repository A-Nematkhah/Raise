"""
Stage II building blocks: ProxyMetrics + StubPolicyTrainer (no real RL).

Run: pytest raise_core/tests/test_refine.py -q
"""

from __future__ import annotations

from raise_core.explore import RewardCandidate
from raise_core.refine import ProxyMetrics, Stage2Config, StubPolicyTrainer


def _candidate(value: float) -> RewardCandidate:
    return RewardCandidate(
        candidate_id="c0",
        code=f"def compute_reward(state, memory):\n    return float({value})\n",
        valid=True,
        origin="initial",
    )


def test_proxy_metrics_feedback_is_raw_not_rankings():
    m = ProxyMetrics(sr=0.8, cr=0.1, tr=0.1, nt=12.0, pl=15.0, itr=20.0, sd=0.3)
    text = m.feedback_text()
    assert "SR=0.8000" in text
    assert "rank" not in text.lower()
    assert set(m.as_dict()) == {"SR", "CR", "TR", "NT", "PL", "ITR", "SD"}


def test_stub_trainer_is_deterministic():
    trainer = StubPolicyTrainer()
    cfg = Stage2Config()
    a = trainer.train_and_eval(_candidate(1.0), round_index=0, config=cfg)
    b = trainer.train_and_eval(_candidate(1.0), round_index=0, config=cfg)
    assert a.as_dict() == b.as_dict()


def test_stage2_defaults():
    cfg = Stage2Config()
    assert cfg.train_env_steps == 12_000
    assert cfg.eval_episodes == 30
    assert cfg.eval_seed_base is None

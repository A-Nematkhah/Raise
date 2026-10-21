"""Stage III multi-seed training / final eval with checkpoint-returning stubs."""

from __future__ import annotations

from typing import Any, List, Optional

from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.candidate import RewardCandidate
from raise_core.refine import ProxyMetrics
from raise_core.validate import (
    Stage3Config,
    Stage3Runner,
    TrainEvalBundle,
)


def _cand(cid: str) -> RewardCandidate:
    code = (
        "def compute_reward(state, memory):\n"
        f"    return float({len(cid)}.0)\n"
    )
    reward_fn, err = make_crowdnav_validator().try_validate(code)
    assert reward_fn is not None, err
    return RewardCandidate(
        candidate_id=cid, code=code, reward_fn=reward_fn, valid=True
    )


class _CheckpointStub:
    """Stub that returns a distinct checkpoint path per train seed."""

    def __init__(self, *, fail_seed: Optional[int] = None) -> None:
        self.fail_seed = fail_seed
        self.reeval_paths: List[str] = []

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        train_seed_index: Optional[int] = None,
        **kwargs: Any,
    ) -> TrainEvalBundle:
        seed = int(train_seed_index if train_seed_index is not None else round_index)
        if self.fail_seed is not None and seed == int(self.fail_seed):
            raise RuntimeError(f"seed {seed} failed")
        # Distinct fitness per seed so median ≠ max.
        fit_jitter = 0.1 * seed
        metrics = ProxyMetrics(
            sr=min(1.0, 0.6 + fit_jitter),
            cr=0.1,
            tr=0.0,
            nt=10.0,
            pl=8.0,
            itr=0.05,
            sd=0.2,
        )
        path = f"/tmp/{candidate.candidate_id}_seed{seed}.zip"
        return TrainEvalBundle(metrics=metrics, checkpoint_path=path)

    def reevaluate(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        train_seed_index: Optional[int] = None,
        **kwargs: Any,
    ) -> ProxyMetrics:
        path = (candidate.metadata or {}).get("checkpoint_path")
        self.reeval_paths.append(str(path))
        # Fitness from path seed index so means are deterministic.
        seed = 0
        if path and "_seed" in str(path):
            seed = int(str(path).rsplit("_seed", 1)[-1].split(".")[0])
        return ProxyMetrics(
            sr=min(1.0, 0.5 + 0.1 * seed),
            cr=0.1,
            tr=0.0,
            nt=10.0,
            pl=8.0,
            itr=0.05,
            sd=0.2,
        )


def test_final_eval_uses_three_seed_checkpoints(tmp_path):
    trainer = _CheckpointStub()
    runner = Stage3Runner(
        trainer,
        validator=make_crowdnav_validator(),
        config=Stage3Config(
            output_root=str(tmp_path / "s3"),
            resume=False,
            n_seeds=3,
            train_env_steps=8,
            eval_episodes=2,
        ),
    )
    pop = runner.run([_cand("c0")])
    assert len(runner.trained) == 1
    md = runner.trained[0].metadata or {}
    assert md["stage3_n_seeds"] == 3
    assert md["n_ok_seeds"] == 3
    assert md["partial_seeds"] is False
    assert md["checkpoint_selection"] == "median_seed"
    assert len(md["stage3_seed_checkpoints"]) == 3
    # Median of fitnesses 0.6, 0.7, 0.8 (sr-based) → middle seed path.
    assert md["checkpoint_path"] == "/tmp/c0_seed1.zip"

    final = runner.final_eval(pop[0], episodes=4, seed_base=99)
    assert final is not None
    assert final["status"] == "ok"
    assert final["n_ok_seeds"] == 3
    assert final["fitness_se_source"] == "across_seeds"
    assert len(final["seed_evals"]) == 3
    assert len(trainer.reeval_paths) == 3
    assert set(trainer.reeval_paths) == set(md["stage3_seed_checkpoints"])


def test_partial_seeds_flag_when_one_seed_fails(tmp_path):
    trainer = _CheckpointStub(fail_seed=1)
    runner = Stage3Runner(
        trainer,
        validator=make_crowdnav_validator(),
        config=Stage3Config(
            output_root=str(tmp_path / "s3_partial"),
            resume=False,
            n_seeds=3,
            train_env_steps=8,
            eval_episodes=2,
        ),
    )
    runner.run([_cand("c1")])
    assert len(runner.trained) == 1
    md = runner.trained[0].metadata or {}
    assert md["n_ok_seeds"] == 2
    assert md["n_failed_seeds"] == 1
    assert md["partial_seeds"] is True
    assert len(md["stage3_seed_checkpoints"]) == 2

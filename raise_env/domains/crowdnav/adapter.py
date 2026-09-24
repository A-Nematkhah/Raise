"""
CrowdNav EnvAdapter — façade over Stage II/III PolicyTrainers.

Pipeline (and future domains) should obtain trainers via
``make_stage2_trainer`` / ``make_stage3_trainer`` rather than importing
``RealPolicyTrainer`` directly. Budgets, GST, and metrics stay unchanged.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


class CrowdNavAdapter:
    """
    Domain-level train/eval surface for CrowdNav.

    Optional ``stage2_delegate`` / ``stage3_delegate`` inject Stub trainers
    (``--fast`` / tests). When omitted, real A2C/PPO trainers are used.
    """

    def __init__(
        self,
        *,
        stage2_delegate: Any = None,
        stage3_delegate: Any = None,
    ) -> None:
        self._stage2_delegate = stage2_delegate
        self._stage3_delegate = stage3_delegate

    def _stage2_trainer(self) -> Any:
        if self._stage2_delegate is not None:
            return self._stage2_delegate
        from raise_core.refine import RealPolicyTrainer

        return RealPolicyTrainer()

    def _stage3_trainer(self) -> Any:
        if self._stage3_delegate is not None:
            return self._stage3_delegate
        from raise_core.validate import RealPolicyTrainer

        return RealPolicyTrainer()

    def train_and_eval(
        self,
        candidate: Any,
        *,
        round_index: int,
        config: Any,
        stage: str = "stage2",
        **train_kwargs: Any,
    ) -> Any:
        stage_key = str(stage).strip().lower()
        if stage_key in ("2", "stage2", "ii"):
            return self._stage2_trainer().train_and_eval(
                candidate, round_index=round_index, config=config
            )
        if stage_key in ("3", "stage3", "iii"):
            return self._stage3_trainer().train_and_eval(
                candidate,
                round_index=round_index,
                config=config,
                **train_kwargs,
            )
        raise ValueError(f"Unknown stage for CrowdNavAdapter: {stage!r}")

    def evaluate_at_human_counts(
        self,
        candidate: Any,
        bundle: Any,
        *,
        config: Any,
        human_counts: Optional[Sequence[int]] = None,
    ) -> Any:
        """Stage III H-sweep; delegates to the Stage III PolicyTrainer."""
        return self._stage3_trainer().evaluate_at_human_counts(
            candidate,
            bundle,
            config=config,
            human_counts=human_counts,
        )


class CrowdNavStage2Trainer:
    """
    ``stage2.PolicyTrainer``-compatible bridge.

    Stage2Runner keeps calling ``trainer.train_and_eval(...)``; this class
    routes those calls through :class:`CrowdNavAdapter`.
    """

    def __init__(self, adapter: CrowdNavAdapter) -> None:
        self.adapter = adapter

    def train_and_eval(
        self,
        candidate: Any,
        *,
        round_index: int,
        config: Any,
    ) -> Any:
        return self.adapter.train_and_eval(
            candidate,
            round_index=round_index,
            config=config,
            stage="stage2",
        )


class CrowdNavStage3Trainer:
    """
    ``stage3.PolicyTrainer``-compatible bridge (train + H-sweep).
    """

    def __init__(self, adapter: CrowdNavAdapter) -> None:
        self.adapter = adapter

    def train_and_eval(
        self,
        candidate: Any,
        *,
        round_index: int,
        config: Any,
        **train_kwargs: Any,
    ) -> Any:
        return self.adapter.train_and_eval(
            candidate,
            round_index=round_index,
            config=config,
            stage="stage3",
            **train_kwargs,
        )

    def evaluate_at_human_counts(
        self,
        candidate: Any,
        bundle: Any,
        *,
        config: Any,
        human_counts: Optional[Sequence[int]] = None,
    ) -> Any:
        return self.adapter.evaluate_at_human_counts(
            candidate,
            bundle,
            config=config,
            human_counts=human_counts,
        )


def make_stage2_trainer(*, use_stub: bool = False) -> CrowdNavStage2Trainer:
    """Build the Stage II trainer used by the CrowdNav domain pack."""
    if use_stub:
        from raise_core.refine import StubPolicyTrainer

        adapter = CrowdNavAdapter(stage2_delegate=StubPolicyTrainer())
    else:
        adapter = CrowdNavAdapter()
    return CrowdNavStage2Trainer(adapter)


def make_stage3_trainer(*, use_stub: bool = False) -> CrowdNavStage3Trainer:
    """Build the Stage III trainer used by the CrowdNav domain pack."""
    if use_stub:
        from raise_core.validate import StubPolicyTrainer

        adapter = CrowdNavAdapter(stage3_delegate=StubPolicyTrainer())
    else:
        adapter = CrowdNavAdapter()
    return CrowdNavStage3Trainer(adapter)

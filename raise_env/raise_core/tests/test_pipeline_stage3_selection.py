"""Pipeline Stage III fallback labelling and LLM-ranker tie restriction."""

from __future__ import annotations

import json
from typing import Any, Optional

from raise_core import pipeline as pipeline_mod
from raise_core.candidate import RewardCandidate
from raise_core.evo_selection.config import SelectionConfig
from raise_core.evo_selection.llm_ranker import LLMRanker, RankResult
from raise_core.pipeline import RaisePipeline, RaiseRunConfig
from raise_core.refine import ProxyMetrics
from raise_core.validate import Stage3Config, TrainEvalBundle


class _FailAllStage3:
    """Stage III trainer that never produces a trained candidate."""

    def train_and_eval(self, candidate, *, round_index, config, **kwargs):
        raise RuntimeError("PPO diverged")

    def reevaluate(self, candidate, *, round_index, config, **kwargs):
        raise FileNotFoundError("no model")


def test_fallback_to_stage2_clears_best_stage3_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer",
        lambda pack, use_stub=False: _FailAllStage3(),
    )
    out = tmp_path / "fallback"
    cfg = RaiseRunConfig(output_dir=str(out))
    cfg.apply_fast_profile()
    arts = RaisePipeline(cfg).run()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    final = json.loads((out / "final_candidate.json").read_text(encoding="utf-8"))

    assert manifest["stage3_status"] == "fallback_to_stage2"
    assert manifest["best_stage3_id"] is None
    assert arts.best_stage3.candidate_id == manifest["best_stage2_id"]
    assert final["metadata"]["selection_stage"] == "stage2_fallback"
    assert manifest["final_eval"] is None


class _FixedFitnessStage3:
    """Deterministic Stage III with controllable per-candidate fitness."""

    def __init__(self, fits: dict) -> None:
        self.fits = dict(fits)
        self.inner = None

    def train_and_eval(
        self,
        candidate: RewardCandidate,
        *,
        round_index: int,
        config: Stage3Config,
        train_seed_index: Optional[int] = None,
        **kwargs: Any,
    ) -> TrainEvalBundle:
        fit = float(self.fits.get(candidate.candidate_id, 0.0))
        # Encode fitness via SR so Stub-style metrics_fitness also works.
        sr = max(0.0, min(1.0, (fit + 1.0) / 2.0))
        metrics = ProxyMetrics(sr=sr, cr=0.0, tr=0.0, nt=10.0, pl=8.0, itr=0.0, sd=0.2)
        path = f"/tmp/{candidate.candidate_id}.zip"
        md = dict(candidate.metadata or {})
        md["fitness"] = fit
        md["selection_scalar"] = fit
        md["last_metrics"] = {
            "fitness": fit,
            "fitness_se": 0.05,
            "fitness_se_source": "episodes_only",
            "SR": sr,
            "CR": 0.0,
            "TR": 0.0,
        }
        md["checkpoint_path"] = path
        candidate.metadata = md
        return TrainEvalBundle(metrics=metrics, checkpoint_path=path)

    def reevaluate(self, candidate, *, round_index, config, **kwargs):
        return self.train_and_eval(
            candidate, round_index=round_index, config=config
        ).metrics


def test_llm_ranker_cannot_pick_outside_fitness_top_k(tmp_path, monkeypatch):
    """Reversed LLM ranking must not beat the measured best when outside top-k."""

    def rank(_self, cards, *, fallback_order, stage, **kwargs):
        # Always reverse: LLM's favourite is the measured worst.
        ranking = list(reversed(fallback_order))
        return RankResult(
            ranking=ranking,
            scores={cid: 0.5 for cid in ranking},
            weaknesses={cid: [] for cid in ranking},
            rationale={cid: "test" for cid in ranking},
        )

    monkeypatch.setattr(LLMRanker, "rank", rank)

    call_i = {"n": 0}

    class _Ordered(_FixedFitnessStage3):
        def train_and_eval(self, candidate, *, round_index, config, **kwargs):
            # First trained gets highest fitness.
            fit = 1.0 - 0.2 * call_i["n"]
            call_i["n"] += 1
            self.fits[candidate.candidate_id] = fit
            return super().train_and_eval(
                candidate, round_index=round_index, config=config, **kwargs
            )

    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer",
        lambda pack, use_stub=False: _Ordered({}),
    )

    cfg = RaiseRunConfig(output_dir=str(tmp_path / "llm_restrict"))
    cfg.apply_fast_profile()
    cfg.selection = SelectionConfig(
        mode="llm_rank",
        top_k=1,
        n_votes=1,
        p_restart=0.0,
        seed=cfg.seed,
    )
    arts = RaisePipeline(cfg).run()
    sel = arts.manifest["stage3_selection"]
    assert sel["llm_restricted_to_top_k"] is True
    assert sel["llm_pool_k"] == 1
    assert sel["llm_restricted_to_ties"] is False
    assert sel["llm_pick_accepted"] is False
    # Measured best stays selected despite reversed LLM ranking.
    assert sel["best_id"] == arts.manifest["best_stage3_id"]
    assert sel["best_id"] == sel["fitness_best_id"]
    assert arts.manifest["stage3_status"] == "ok"


def test_llm_ranker_may_pick_within_fitness_top_k(tmp_path, monkeypatch):
    """LLM may choose the 2nd-best fitness candidate when top_k >= 2."""

    call_i = {"n": 0}

    class _Ordered(_FixedFitnessStage3):
        def train_and_eval(self, candidate, *, round_index, config, **kwargs):
            fit = 1.0 - 0.2 * call_i["n"]
            call_i["n"] += 1
            self.fits[candidate.candidate_id] = fit
            return super().train_and_eval(
                candidate, round_index=round_index, config=config, **kwargs
            )

    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer",
        lambda pack, use_stub=False: _Ordered({}),
    )

    def rank(_self, cards, *, fallback_order, stage, **kwargs):
        # Prefer the second-best by fitness (index 1 in fitness-desc order).
        ranking = list(fallback_order)
        if len(ranking) >= 2:
            ranking = [ranking[1], ranking[0], *ranking[2:]]
        return RankResult(
            ranking=ranking,
            scores={cid: 0.5 for cid in ranking},
            weaknesses={cid: [] for cid in ranking},
            rationale={cid: "test" for cid in ranking},
        )

    monkeypatch.setattr(LLMRanker, "rank", rank)

    cfg = RaiseRunConfig(output_dir=str(tmp_path / "llm_top_k"))
    cfg.apply_fast_profile()
    cfg.selection = SelectionConfig(
        mode="llm_rank",
        top_k=2,
        n_votes=1,
        p_restart=0.0,
        seed=cfg.seed,
    )
    arts = RaisePipeline(cfg).run()
    sel = arts.manifest["stage3_selection"]
    assert sel["llm_restricted_to_top_k"] is True
    assert sel["llm_pool_k"] == 2
    assert sel["llm_pick_accepted"] is True
    assert sel["best_id"] == sel["llm_ranking"][0]
    assert sel["best_id"] != sel["fitness_best_id"]
    assert arts.best_stage3.candidate_id == sel["best_id"]


def test_llm_ranker_may_pick_within_tie_set(tmp_path, monkeypatch):
    """When candidates are statistical ties, LLM may choose any of them."""

    class _TiedTrainer(_FixedFitnessStage3):
        def train_and_eval(self, candidate, *, round_index, config, **kwargs):
            # Identical fitness + SE → everyone ties with best.
            self.fits[candidate.candidate_id] = 0.5
            bundle = super().train_and_eval(
                candidate, round_index=round_index, config=config, **kwargs
            )
            md = dict(candidate.metadata or {})
            md["last_metrics"] = {
                **(md.get("last_metrics") or {}),
                "fitness": 0.5,
                "fitness_se": 0.2,
            }
            md["fitness"] = 0.5
            candidate.metadata = md
            return bundle

    monkeypatch.setattr(
        pipeline_mod,
        "make_stage3_trainer",
        lambda pack, use_stub=False: _TiedTrainer({}),
    )

    def rank(_self, cards, *, fallback_order, stage, **kwargs):
        ranking = list(reversed(fallback_order))
        return RankResult(
            ranking=ranking,
            scores={cid: 0.5 for cid in ranking},
            weaknesses={cid: [] for cid in ranking},
            rationale={cid: "test" for cid in ranking},
        )

    monkeypatch.setattr(LLMRanker, "rank", rank)

    cfg = RaiseRunConfig(output_dir=str(tmp_path / "llm_tie"))
    cfg.apply_fast_profile()
    cfg.selection = SelectionConfig(
        mode="llm_rank",
        top_k=1,
        n_votes=1,
        p_restart=0.0,
        seed=cfg.seed,
    )
    arts = RaisePipeline(cfg).run()
    sel = arts.manifest["stage3_selection"]
    assert sel["llm_restricted_to_top_k"] is True
    # With everyone tied, the reversed LLM top is allowed via tie set.
    assert sel["llm_pick_accepted"] is True
    assert sel["best_id"] == sel["llm_ranking"][0]
    assert arts.best_stage3.candidate_id == sel["best_id"]

"""Highway closed-loop evolve hooks (moved out of ClosedLoopRunner)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from domains.highway.evolve_hooks import HighwayEvolveHooks
from domains.highway.prompts import D5_SEED_FUNCTION
from domains.highway.reward_checks import make_highway_validator
from raise_core.domains import load_domain
from raise_core.explore import RewardCandidate


def _cand(cid: str, fitness: float, **md) -> RewardCandidate:
    return RewardCandidate(
        candidate_id=cid,
        code=D5_SEED_FUNCTION,
        reward_fn=make_highway_validator().validate_code(D5_SEED_FUNCTION),
        valid=True,
        origin="test",
        score=0.5,
        metadata={"fitness": fitness, **md},
    )


def test_only_highway_pack_has_evolve_hooks():
    assert load_domain("crowdnav", with_adapter=False).make_evolve_hooks is None
    hooks = load_domain("highway", with_adapter=False).make_evolve_hooks()
    assert isinstance(hooks, HighwayEvolveHooks)


def test_state_roundtrip_through_json():
    src = HighwayEvolveHooks()
    assert src.state_dict() is None
    src.pareto_elite = _cand("ini_0001", 0.2)
    src.best_ever_cand = _cand("cro_0002", 0.75)
    src.best_ever_fitness = 0.75
    saved = json.loads(json.dumps(src.state_dict()))

    dst = HighwayEvolveHooks()
    dst.load_state_dict(saved, validator=make_highway_validator())
    assert dst.pareto_elite.candidate_id == "ini_0001"
    assert dst.best_ever_cand.reward_fn is not None
    assert dst.best_ever_fitness == 0.75


def test_fitness_archive_pins_best_ever_first(tmp_path):
    hooks = HighwayEvolveHooks()
    cfg = SimpleNamespace(
        highway_calibration_mode="no_speed_floor",
        highway_elite_archive="fitness",
        keep_runtime_elite=True,
        use_stub=True,
        output_dir=str(tmp_path),
        seed=0,
    )
    inputs = hooks.ranking_inputs(cfg, "scalar")
    assert inputs["pareto_ref"] is None and inputs["ambient"] is None
    ranked = [_cand("a", 0.1), _cand("b", 0.9), _cand("c", 0.5)]
    out, elite_id = hooks.after_ranking(ranked, cfg, inputs)
    assert elite_id == "b"
    assert [c.candidate_id for c in out][0] == "b"
    assert hooks.best_ever_fitness == 0.9
    assert hooks.pareto_summary(out, "scalar", inputs) is None

    later, elite_id = hooks.after_ranking([_cand("d", 0.3)], cfg, inputs)
    assert elite_id == "b"
    assert [c.candidate_id for c in later] == ["b", "d"]

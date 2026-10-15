"""PPO discount knob, gamma sweep script and ORCA reference check."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from raise_core.refine import Stage2Config
from raise_core.validate import Stage3Config

from raise_core.tests.test_baselines import _write_minimal_run

_RAISE_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _RAISE_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ppo_gamma_defaults_to_domain_constant():
    from domains.crowdnav.adapter import _ppo_gamma
    from domains.crowdnav.objective_constants import PPO_GAMMA

    assert Stage2Config().ppo_gamma is None
    assert Stage3Config().ppo_gamma is None
    assert _ppo_gamma(Stage3Config()) == PPO_GAMMA
    assert _ppo_gamma(SimpleNamespace()) == PPO_GAMMA
    assert _ppo_gamma(Stage3Config(ppo_gamma=0.974)) == pytest.approx(0.974)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            _ppo_gamma(Stage3Config(ppo_gamma=bad))


def _fake_trainer(seen):
    """Stub train/reeval: per-case task_return grows with the training gamma."""

    def _train(self, candidate, *, round_index, config, **kw):
        gamma = config.ppo_gamma if config.ppo_gamma is not None else 0.99
        seen.append((candidate.candidate_id, gamma, config.output_root))
        n = max(2, int(config.eval_episodes))
        eps = [{"case": i, "task_return": gamma + 0.01 * i} for i in range(n)]
        md = dict(candidate.metadata or {})
        md["ppo_gamma"] = gamma
        md["checkpoint_path"] = "ckpt.zip"
        md["last_metrics"] = {
            "fitness": gamma,
            "SR": 0.5,
            "CR": 0.25,
            "TR": 0.25,
            "holdout_episodes": eps,
            "holdout": {"SR": 0.5, "CR": 0.25, "TR": 0.25},
        }
        candidate.metadata = md
        return SimpleNamespace(checkpoint_path="ckpt.zip")

    def _reeval(self, candidate, *, round_index, config):
        return None

    return _train, _reeval


def test_run_baselines_gamma_writes_separate_file(tmp_path, monkeypatch):
    bas = _load_script("run_baselines")
    run = _write_minimal_run(tmp_path)
    seen = []
    train, reeval = _fake_trainer(seen)
    monkeypatch.setattr(bas.CrowdNavPPOTrainer, "train_and_eval", train)
    monkeypatch.setattr(bas.CrowdNavPPOTrainer, "reevaluate", reeval)

    bas.run_baselines(
        str(run), eval_episodes=2, train_steps=8, skip_b=True, skip_c=True, ppo_gamma=0.974
    )
    assert not (run / "baselines.json").exists()
    data = json.loads((run / "baselines_gamma0.974.json").read_text(encoding="utf-8"))
    assert data["ppo_gamma"] == pytest.approx(0.974)
    assert data["baselines"]["A_native_ppo"]["runs"][0]["ppo_gamma"] == pytest.approx(0.974)
    assert seen and all(g == pytest.approx(0.974) for _cid, g, _root in seen)
    assert all(root.endswith("baselines_train_gamma0.974") for _cid, _g, root in seen)


def test_matched_gamma_and_parsing():
    sweep = _load_script("run_gamma_sweep")
    g = sweep.matched_gamma(None)
    assert g == pytest.approx(0.9 ** 0.25)
    assert sweep.parse_gammas("matched,0.99,0.99", matched=g) == [0.974, 0.99]
    with pytest.raises(ValueError):
        sweep.parse_gammas("1.2", matched=g)
    with pytest.raises(ValueError):
        sweep.parse_gammas(" , ", matched=g)


def test_gamma_sweep_runs_pairs_and_resumes(tmp_path, monkeypatch):
    sweep = _load_script("run_gamma_sweep")
    run = _write_minimal_run(tmp_path)
    seen = []
    train, reeval = _fake_trainer(seen)
    from domains.crowdnav.adapter import CrowdNavPPOTrainer

    monkeypatch.setattr(CrowdNavPPOTrainer, "train_and_eval", train)
    monkeypatch.setattr(CrowdNavPPOTrainer, "reevaluate", reeval)

    out = sweep.run_gamma_sweep(
        str(run), gammas="matched", rewards="native,final", seeds=2, train_steps=8, eval_episodes=2
    )
    # Reference gamma 0.99 is added automatically: 2 rewards x 2 gammas x 2 seeds.
    assert out["gammas"] == [0.974, 0.99]
    assert len(seen) == 8 and len(out["runs"]) == 8
    assert all(root.endswith("gamma_sweep_train") for _cid, _g, root in seen)
    row = out["summary"]["native"]["0.974"]
    assert row["n_seeds"] == 2
    assert row["fitness_mean"] == pytest.approx(0.974)
    pairs = row["vs_reference"]
    assert [p["seed_index"] for p in pairs] == [0, 1]
    assert all(p["delta"] == pytest.approx(0.974 - 0.99) for p in pairs)
    assert "vs_reference" not in out["summary"]["native"]["0.99"]
    on_disk = json.loads((run / "gamma_sweep.json").read_text(encoding="utf-8"))
    assert set(on_disk["runs"]) == set(out["runs"])

    seen.clear()
    sweep.run_gamma_sweep(
        str(run), gammas="matched", rewards="native,final", seeds=2, train_steps=8, eval_episodes=2
    )
    assert seen == []

    sweep.run_gamma_sweep(
        str(run), gammas="matched", rewards="native,final", seeds=2, train_steps=16, eval_episodes=2
    )
    assert len(seen) == 8


def test_gamma_sweep_rejects_unknown_reward(tmp_path):
    sweep = _load_script("run_gamma_sweep")
    run = _write_minimal_run(tmp_path)
    rb = _load_script("run_baselines")
    codes = sweep.resolve_rewards(str(run), ["native", "final", "ini_0000"], rb)
    assert set(codes) == {"native", "final", "ini_0000"}
    with pytest.raises(ValueError):
        sweep.resolve_rewards(str(run), ["nope"], rb)


def test_orca_compare_to_reference_flags_mismatch():
    orca = _load_script("check_orca_reference")
    good = {"SR": 0.426, "CR": 0.568, "nav_time": 10.86, "reward_terminal": 0.0539}
    assert orca.compare_to_reference(good)["all_ok"] is True
    bad = dict(good, SR=0.30)
    rep = orca.compare_to_reference(bad)
    assert rep["all_ok"] is False
    assert rep["metrics"]["SR"]["ok"] is False and rep["metrics"]["CR"]["ok"] is True
    assert orca.compare_to_reference(dict(good, nav_time=float("nan")))["all_ok"] is False


@pytest.mark.slow
def test_orca_matches_published_reference():
    """Full 500-case test split (~6 s): upstream config reproduces Chen et al. 2019 ORCA row."""
    pytest.importorskip("rvo2")
    orca = _load_script("check_orca_reference")
    measured = orca.run_orca_check()
    assert measured["n_cases"] == 500
    assert orca.compare_to_reference(measured)["all_ok"], measured
    assert measured["reward_full"] < measured["reward_terminal"]


@pytest.mark.slow
def test_ppo_gamma_reaches_sb3(tmp_path):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("rvo2")
    from domains.crowdnav.adapter import CrowdNavPPOTrainer
    from domains.crowdnav.prompts import D5_SEED_FUNCTION
    from domains.crowdnav.reward_checks import make_crowdnav_validator
    from raise_core.candidate import RewardCandidate

    cand = RewardCandidate(
        candidate_id="g",
        code=D5_SEED_FUNCTION,
        reward_fn=make_crowdnav_validator().validate_code(D5_SEED_FUNCTION),
        valid=True,
    )
    cfg = Stage3Config(
        train_env_steps=32, eval_episodes=1, output_root=str(tmp_path), resume=False,
        ppo_n_steps=16, ppo_gamma=0.95, eval_mode="holdout_only",
    )
    bundle = CrowdNavPPOTrainer(stage="stage3").train_and_eval(cand, round_index=0, config=cfg)
    assert cand.metadata["ppo_gamma"] == pytest.approx(0.95)
    from stable_baselines3 import PPO

    assert PPO.load(bundle.checkpoint_path, device="cpu").gamma == pytest.approx(0.95)

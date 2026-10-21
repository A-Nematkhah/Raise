"""P1-1 baselines script: selection helpers, schema, CLI hook (stub budgets)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from domains.crowdnav.prompts import D5_SEED_FUNCTION
from domains.crowdnav.reward_checks import make_crowdnav_validator
from raise_core.candidate import RewardCandidate
from raise_core.json_io import atomic_write_json

_RAISE_ROOT = Path(__file__).resolve().parents[2]


def _load_baselines_mod():
    import raise_core.post_run_baselines as mod

    return mod


def _cand(cid: str, fitness: float, cr: float) -> RewardCandidate:
    code = "def compute_reward(state, memory):\n    return 0.0, {}\n"
    fn, err = make_crowdnav_validator().try_validate(code)
    assert fn is not None, err
    return RewardCandidate(
        candidate_id=cid,
        code=code,
        reward_fn=fn,
        valid=True,
        metadata={
            "fitness": fitness,
            "last_metrics": {
                "fitness": fitness,
                "holdout": {"CR": cr, "SR": 1.0 - cr, "TR": 0.0},
                "CR": cr,
            },
        },
    )


def test_select_baseline_b_skips_collapsed_and_honours_extra_ids():
    bas = _load_baselines_mod()
    cands = {
        "good": _cand("good", 0.5, cr=0.1),
        "better_but_collapsed": _cand("better_but_collapsed", 0.9, cr=0.95),
        "mid": _cand("mid", 0.3, cr=0.2),
        "ini_0000": _cand("ini_0000", 0.1, cr=0.0),
    }
    ids = bas.select_baseline_b_ids(cands, collapse_cr=0.8, extra_ids=["ini_0000", "missing"])
    assert ids[0] == "good"
    assert "ini_0000" in ids
    assert "better_but_collapsed" not in ids
    assert "missing" not in ids


def test_vs_final_paired_delta_and_z():
    bas = _load_baselines_mod()
    eps_b = [{"case": i, "outcome": "timeout"} for i in range(10)]
    eps_f = [{"case": i, "outcome": "success"} for i in range(10)]
    row = bas._vs_final(eps_b, eps_f, baseline_fitness=-0.5, final_fitness=1.0)
    assert row["method"] == "paired_outcome_score"
    assert row["delta_fitness"] == pytest.approx(1.5)
    assert row["paired_se"] == pytest.approx(0.0)
    assert row["z"] is None or row["n"] == 10


def test_baselines_cli_flag_default_off():
    from scripts.run_raise import build_parser

    args = build_parser().parse_args([])
    assert args.baselines is False


def _write_minimal_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    cl = run / "closed_loop"
    cl.mkdir(parents=True)
    code = D5_SEED_FUNCTION
    atomic_write_json(
        str(run / "config.json"),
        {
            "seed": 7,
            "stage3_train_steps": 8,
            "final_eval_episodes": 2,
            "n_envs": 1,
            "device": "cpu",
            "action_mode": "discrete",
            "robot_kinematics": "holonomic",
            "action_speed_samples": 5,
            "action_rotation_samples": 16,
            "eval_mode": "holdout_only",
            "collapse_cr": 0.8,
        },
    )
    eps = [{"case": i, "task_return": 0.05 * i} for i in range(2)]
    atomic_write_json(
        str(run / "final_candidate.json"),
        {
            "candidate_id": "winner",
            "code": code,
            "metadata": {
                "fitness": 0.4,
                "final_eval": {
                    "status": "ok",
                    "fitness": 0.42,
                    "metrics": {
                        "fitness": 0.42,
                        "holdout_episodes": eps,
                        "task_return": 0.42,
                    },
                },
            },
        },
    )
    atomic_write_json(
        str(cl / "pop_epoch_000.json"),
        {
            "candidates": [
                {
                    "candidate_id": "ini_0000",
                    "code": code,
                    "fitness": 0.3,
                    "last_metrics": {
                        "fitness": 0.3,
                        "holdout": {"CR": 0.1, "SR": 0.8, "TR": 0.1},
                    },
                },
                {
                    "candidate_id": "col_0001",
                    "code": code,
                    "fitness": 0.9,
                    "last_metrics": {
                        "fitness": 0.9,
                        "holdout": {"CR": 0.95, "SR": 0.05, "TR": 0.0},
                    },
                },
            ]
        },
    )
    return run


def test_run_baselines_writes_expected_schema(tmp_path, monkeypatch):
    bas = _load_baselines_mod()
    run = _write_minimal_run(tmp_path)

    def _fake_train(self, candidate, *, round_index, config, **kw):
        md = dict(candidate.metadata or {})
        md["checkpoint_path"] = str(tmp_path / f"{candidate.candidate_id}.zip")
        Path(md["checkpoint_path"]).write_bytes(b"x")
        eps = [{"case": i, "task_return": 0.1 + 0.01 * i} for i in range(2)]
        md["last_metrics"] = {
            "fitness": 0.11,
            "fitness_se": 0.01,
            "SR": 0.5,
            "CR": 0.25,
            "TR": 0.25,
            "task_return": 0.11,
            "holdout_episodes": eps,
            "holdout": {"SR": 0.5, "CR": 0.25, "TR": 0.25, "task_return": 0.11},
        }
        md["fitness"] = 0.11
        candidate.metadata = md
        return SimpleNamespace(metrics=SimpleNamespace(as_dict=lambda: md["last_metrics"]),
                               checkpoint_path=md["checkpoint_path"])

    def _fake_reeval(self, candidate, *, round_index, config):
        return _fake_train(self, candidate, round_index=round_index, config=config)

    def _fake_controller(behaviour, *, cases, run_cfg):
        eps = [{"case": int(c), "task_return": 0.02} for c in cases]
        return {
            "controller": behaviour,
            "fitness": 0.02,
            "fitness_se": 0.0,
            "SR": 0.0,
            "CR": 0.0,
            "TR": 1.0,
            "holdout_episodes": eps,
        }

    monkeypatch.setattr(bas.CrowdNavPPOTrainer, "train_and_eval", _fake_train)
    monkeypatch.setattr(bas.CrowdNavPPOTrainer, "reevaluate", _fake_reeval)
    monkeypatch.setattr(bas, "_eval_controller", _fake_controller)

    out = bas.run_baselines(
        str(run),
        seeds=1,
        extra_ids=["ini_0000"],
        eval_episodes=2,
        train_steps=8,
        skip_c=False,
    )
    path = run / "baselines.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["eval_split"] == "test"
    assert data["eval_episodes"] == 2
    assert data["eval_deterministic"] is False
    assert "A_native_ppo" in data["baselines"]
    assert "B_ini_0000" in data["baselines"]
    assert "C_orca" in data["baselines"] and "C_linear" in data["baselines"]
    assert "A_native_ppo" in data["comparisons"]
    cmp_a = data["comparisons"]["A_native_ppo"]
    assert "delta_fitness" in cmp_a and "paired_se" in cmp_a and "z" in cmp_a
    assert out["final_candidate_id"] == "winner"
    # Collapsed col_0001 must not be auto-selected as B.
    assert "B_col_0001" not in data["baselines"]


def test_baseline_b_pick_equal_to_final_is_not_retrained(tmp_path, monkeypatch):
    bas = _load_baselines_mod()
    run = _write_minimal_run(tmp_path)
    final = json.loads((run / "final_candidate.json").read_text(encoding="utf-8"))
    final["candidate_id"] = "ini_0000"
    atomic_write_json(str(run / "final_candidate.json"), final)

    trained = []

    def _no_train(self, candidate, **kw):
        trained.append(candidate.candidate_id)
        raise AssertionError("must not train")

    monkeypatch.setattr(bas.CrowdNavPPOTrainer, "train_and_eval", _no_train)
    bas.run_baselines(str(run), eval_episodes=2, train_steps=8, skip_a=True, skip_c=True)
    data = json.loads((run / "baselines.json").read_text(encoding="utf-8"))
    assert data["baselines"]["B_ini_0000"]["status"] == "same_as_final"
    assert "B_ini_0000" not in data["comparisons"]
    assert trained == []


def test_stage2_loader_skips_sandbox_without_validator(tmp_path):
    bas = _load_baselines_mod()
    run = _write_minimal_run(tmp_path)
    cands = bas.load_stage2_candidates(str(run))
    assert set(cands) == {"ini_0000", "col_0001"}
    assert all(c.reward_fn is None for c in cands.values())
    assert cands["ini_0000"].metadata["fitness"] == 0.3


@pytest.mark.slow
def test_run_baselines_tiny_real_ppo_smoke(tmp_path):
    """Optional slow smoke: tiny K3 + 1 eval episode (needs CrowdNav deps)."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("rvo2")
    bas = _load_baselines_mod()
    run = _write_minimal_run(tmp_path)
    # Tiny real training; skip B/C to keep wall-clock short.
    bas.run_baselines(
        str(run),
        seeds=1,
        eval_episodes=1,
        train_steps=8,
        skip_b=True,
        skip_c=True,
    )
    data = json.loads((run / "baselines.json").read_text(encoding="utf-8"))
    assert "A_native_ppo" in data["baselines"]
    fit = data["baselines"]["A_native_ppo"]["fitness"]
    assert fit is not None and fit == fit  # not NaN

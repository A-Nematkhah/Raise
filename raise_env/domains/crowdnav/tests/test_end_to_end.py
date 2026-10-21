"""Representative CrowdNav RAISE run: real Score1 dataset, real SB3 PPO, final test eval, plots + GIF."""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys

import pytest

from domains.crowdnav.adapter import CrowdNavPPOTrainer
from domains.crowdnav.env_wrapper import load_env_config
from domains.crowdnav.tests.conftest import RAISE_ENV_ROOT
from raise_core.candidate import RewardCandidate
from raise_core.domains import load_domain, make_validator_for_domain
from raise_core.validate import Stage3Config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run_dir(dataset_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("cn_run")
    cmd = [
        sys.executable,
        os.path.join(RAISE_ENV_ROOT, "scripts", "run_raise.py"),
        "--allow-seed-llm",
        "--set",
        "ppo.device=cpu",
        "--set",
        f"budget.stage1.dataset_path={dataset_dir}",
        "--set",
        "budget.population_size=2",
        "--set",
        "budget.generations=1",
        "--set",
        "budget.stage2_train_steps=256",
        "--set",
        "budget.stage2_eval_episodes=2",
        "--set",
        "budget.stage3.train_steps=256",
        "--set",
        "budget.stage3.eval_episodes=2",
        "--set",
        "budget.final_eval_episodes=3",
        "--output-dir",
        str(out),
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, cwd=RAISE_ENV_ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return str(out)


def _load(run_dir, name):
    with open(os.path.join(run_dir, name), encoding="utf-8") as fh:
        return json.load(fh)


def test_run_writes_crowdnav_artifacts(run_dir):
    manifest = _load(run_dir, "manifest.json")
    assert manifest["domain"] == "crowdnav"
    assert manifest["config"]["robot_kinematics"] == "unicycle"
    for name in ("config.json", "seed_reward.py", "best_stage1.json", "stage2_population.json",
                 "best_stage2.json", "stage3_population.json", "best_stage3.json", "final_candidate.json"):
        assert os.path.isfile(os.path.join(run_dir, name)), name

    final = manifest["final_eval"]
    assert final is not None and final["split"] == "test"
    best = _load(run_dir, "final_candidate.json")
    md = best["metadata"]
    assert os.path.isfile(md["checkpoint_path"])
    metrics = md["last_metrics"]
    assert metrics["domain"] == "crowdnav" and math.isfinite(metrics["fitness"])
    holdout = metrics["holdout"]
    assert holdout["SR"] + holdout["CR"] + holdout["TR"] == pytest.approx(1.0)
    assert md["train_steps_per_second"] > 0.0
    ctx = metrics["eval_context"]
    env_config = load_env_config()
    assert ctx["human_num"] == env_config.sim.human_num
    assert ctx["time_limit_s"] == env_config.env.time_limit
    assert isinstance(ctx["scenario"], str) and ctx["scenario"]


def test_run_writes_plots_and_policy_animation(run_dir):
    post = _load(run_dir, "manifest.json")["post_run"]
    assert post["ok"] and post["viz_ok"], post
    assert post["n_plots"] >= 1
    viz = os.path.join(run_dir, "plots", "viz")
    assert glob.glob(os.path.join(viz, "*.gif"))
    assert glob.glob(os.path.join(viz, "*_traj.png")) and glob.glob(os.path.join(viz, "*_strip.png"))
    meta = json.load(open(os.path.join(viz, "viz_manifest.json"), encoding="utf-8"))
    assert meta["split"] == "test" and len(meta["episodes"]) == 2
    assert {e["outcome"] for e in meta["episodes"]} <= {"success", "collision", "timeout"}


def test_reevaluate_saved_policy_is_reproducible(run_dir, tmp_path):
    best = _load(run_dir, "final_candidate.json")
    cand = RewardCandidate(
        candidate_id=best["candidate_id"],
        code=best["code"],
        reward_fn=make_validator_for_domain(load_domain("crowdnav")).validate_code(best["code"]),
        valid=True,
        metadata=dict(best["metadata"]),
    )
    cfg = Stage3Config(eval_episodes=3, seed=425, eval_deterministic=True, eval_split="val",
                       output_root=str(tmp_path), device="cpu")
    trainer = CrowdNavPPOTrainer(stage="stage3")
    first = trainer.reevaluate(cand, round_index=0, config=cfg).as_dict()
    val_cases = cand.metadata["last_metrics"]["eval_context"]["cases"]
    second = trainer.reevaluate(cand, round_index=0, config=cfg).as_dict()
    assert first == second
    assert cand.metadata["eval_split"] == "val"

    cfg.eval_split = "test"
    trainer.reevaluate(cand, round_index=0, config=cfg)
    ctx = cand.metadata["last_metrics"]["eval_context"]
    assert ctx["split"] == "test" and ctx["case_set_size"] == 500
    assert len(val_cases) == 3

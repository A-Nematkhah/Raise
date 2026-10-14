"""CrowdNav fitness / metrics and the Stage I dataset + Score1 on real upstream rollouts."""

from __future__ import annotations

import json
import math

import pytest

from domains.crowdnav.metrics import (
    attach_fitness,
    crowdnav_fitness,
    discounted_task_return,
    episode_fields,
    evidence_line,
    fitness_standard_error,
)
from domains.crowdnav.prompts import D5_SEED_FUNCTION
from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.sandbox_fields import crowdnav_sandbox_config
from domains.crowdnav.score1_worker import Score1TimeoutWorker
from domains.crowdnav.settings import CrowdNavSettings
from domains.crowdnav.stage1 import (
    load_crowdnav_dataset,
    make_smoke_score_fn,
    read_dataset_manifest,
    save_crowdnav_dataset,
    score_crowdnav_dataset,
    with_settings,
)
from raise_core.sandbox.runtime import SandboxedReward, compile_compute_reward

_FN = "def compute_reward(state, memory):\n"


def _validated(code: str):
    reward_fn, err = make_crowdnav_validator().try_validate(code)
    assert reward_fn is not None, err
    return reward_fn


# ------------------------------------------------------------------ metrics


def test_discounted_task_return_matches_upstream_explorer():
    # Explorer: sum_t gamma ** (t * time_step * v_pref) * r_t
    got = discounted_task_return([0.0, -0.01, 1.0], gamma=0.9, time_step=0.25, v_pref=1.0)
    assert got == pytest.approx(-0.01 * 0.9 ** 0.25 + 0.9 ** 0.5)


def test_episode_fields_and_fitness():
    f = episode_fields(
        ["success", "collision", "timeout", "success"],
        [10.0, 3.0, 25.0, 12.0],
        [0.5, -0.25, 0.0, 0.4],
        [2, 0, 0, 1],
        [40, 12, 100, 48],
        [0.1, 0.15],
        time_limit=25.0,
    )
    assert (f["SR"], f["CR"], f["TR"]) == (0.5, 0.25, 0.25)
    assert f["NT"] == pytest.approx(11.0)
    assert f["task_return"] == pytest.approx(0.1625)
    assert f["ITR"] == pytest.approx(3 / 200)
    assert f["SD"] == pytest.approx(0.125)
    # No success: nav time is the time limit (upstream reports it the same way).
    assert episode_fields(["timeout"], [25.0], [0.0], [0], [100], [], time_limit=25.0)["NT"] == 25.0

    m = attach_fitness({"task_return": 0.1, "holdout": {"task_return": 0.3}})
    assert m["fitness"] == m["selection_scalar"] == pytest.approx(0.3) and m["domain"] == "crowdnav"
    assert crowdnav_fitness(None) == float("-inf") and crowdnav_fitness({"SR": 1.0}) == float("-inf")
    assert fitness_standard_error([1.0]) == 0.0
    assert fitness_standard_error([0.0, 1.0, 2.0]) == pytest.approx(1.0 / math.sqrt(3))


def test_evidence_line_reports_crowdnav_metrics():
    line = evidence_line({"holdout": {"SR": 0.8, "CR": 0.1, "TR": 0.1, "NT": 11.2, "PL": 9.1,
                                      "ITR": 0.05, "SD": 0.12, "n_eval_episodes": 20}})
    assert line.startswith("Holdout eval (20 episodes): SR=0.80 CR=0.10 TR=0.10 nav_time=11.20s")
    assert "path_length=9.10m" in line and "danger_freq=0.050" in line and "min_dist_in_danger=0.120m" in line


# ------------------------------------------------------------------ dataset


def test_collected_dataset_is_real_upstream_rollouts(dataset_dir):
    manifest = read_dataset_manifest(dataset_dir)
    assert manifest["domain"] == "crowdnav" and manifest["n_trajectories"] == 12
    assert manifest["upstream_commit"].startswith("20d6780")
    trajs = load_crowdnav_dataset(dataset_dir)
    labels = {t.behavior: {t2.label for t2 in trajs if t2.behavior == t.behavior} for t in trajs}
    assert labels["stop"] == {"timeout"}
    assert "success" in labels["orca"]
    for t in trajs:
        last = t.states[-1]
        assert {"success": last.reached_goal, "collision": last.collision, "timeout": last.timeout}[t.label]
        assert last.time_limit == 25.0 and last.time_step == 0.25
        # Same scenario across behaviours (train-phase case per episode).
        assert t.scenario_id.startswith("train_case_")


def test_dataset_round_trip_and_domain_check(dataset_dir, tmp_path):
    trajs = load_crowdnav_dataset(dataset_dir)
    save_crowdnav_dataset(str(tmp_path / "copy"), trajs)
    again = load_crowdnav_dataset(str(tmp_path / "copy"))
    assert [t.states for t in again] == [t.states for t in trajs]

    other = tmp_path / "other"
    save_crowdnav_dataset(str(other), trajs[:1])
    manifest = json.loads((other / "manifest.json").read_text(encoding="utf-8"))
    manifest["domain"] = "something_else"
    (other / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not a CrowdNav"):
        load_crowdnav_dataset(str(other))


def test_unicycle_view_of_holonomic_actions(dataset_dir):
    traj = next(t for t in load_crowdnav_dataset(dataset_dir) if t.behavior == "linear")
    uni = with_settings([traj], CrowdNavSettings(kinematics="unicycle", action_mode="continuous"))[0]
    for holo, st in zip(traj.states, uni.states):
        assert st.action[0] == pytest.approx(math.hypot(*holo.action))
        assert -math.pi <= st.action[1] <= math.pi


# ------------------------------------------------------------------ Score1


def test_seed_reward_scores_positive_on_real_rollouts(dataset_dir):
    result = score_crowdnav_dataset(_validated(D5_SEED_FUNCTION), load_crowdnav_dataset(dataset_dir))
    assert not result.rejected
    assert result.scenario_scores["safety"] > 0.0 and result.score > 0.0


def test_score1_rejects_failure_seeking_and_constant_rewards(dataset_dir):
    trajs = load_crowdnav_dataset(dataset_dir)
    crash = _validated(_FN + "    if state.collision:\n        return 1.0\n    return -0.01\n")
    assert score_crowdnav_dataset(crash, trajs).reject_reason == "prefers_failure"
    const = _validated(_FN + "    return 0.0\n")
    assert score_crowdnav_dataset(const, trajs).reject_reason == "no_signal"


def test_smoke_score_never_rejects():
    result = make_smoke_score_fn()(_validated(_FN + "    return 0.0\n"))
    assert not result.rejected


def test_score1_worker_matches_in_process_and_times_out(dataset_dir):
    trajs = load_crowdnav_dataset(dataset_dir)
    seed = _validated(D5_SEED_FUNCTION)
    worker = Score1TimeoutWorker(dataset_dir, trajs, timeout_seconds=30.0)
    try:
        assert worker(seed).score == pytest.approx(score_crowdnav_dataset(seed, trajs).score)
        slow_code = _FN + (
            "    total = 0.0\n"
            "    for i in range(5000000):\n"
            "        total = total + 0.0\n"
            "    return total\n"
        )
        cfg = crowdnav_sandbox_config(max_call_seconds=1e6)
        slow = SandboxedReward(compile_compute_reward(slow_code, cfg), cfg, source_code=slow_code)
        worker.timeout_seconds = 2.0
        result = worker(slow, candidate_id="slow")
        assert result.rejected and result.reject_reason == "score1_timeout"
        assert worker.restarts == 1
        worker.timeout_seconds = 30.0
        assert not worker(seed).rejected
    finally:
        worker.close()

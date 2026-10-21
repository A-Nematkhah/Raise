"""Diagnostic feedback card, lineage archive, token budget and LLM call logging.

Mocked LLM clients only: no API calls, no PPO training.
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

import domains.crowdnav.prompts as cn_prompts
from domains.crowdnav import feedback_card as fc
from domains.crowdnav.adapter import _eval_metrics
from domains.crowdnav.metrics import attach_fitness
from domains.crowdnav.objective_constants import PPO_GAMMA
from domains.crowdnav.reward_checks import make_crowdnav_validator
from domains.crowdnav.training_dynamics import TrainingDynamicsRecorder
from raise_core.explore import RewardCandidate, StageIConfig, StageIEvolver
from raise_core.feedback import (
    ArchiveRecord,
    CandidateArchive,
    FeedbackCardConfig,
    diff_code,
    summarize_code,
)
from raise_core.llm import DEFAULT_CLIENT_SYSTEM, LLMClient
from raise_core.llm_call_log import CALLS_FILE
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.raise_loop.runner import ClosedLoopRunner
from raise_core import token_budget
from raise_core.token_budget import (
    PromptBudgetError,
    fit_to_budget,
    heuristic_token_count,
    make_token_counter,
    tiktoken,
)

CLIENT_SYSTEM = DEFAULT_CLIENT_SYSTEM


# ------------------------------------------------------------------ fixtures


def _code(collision: float = -20.0, coef: float = 0.08, clearance_bonus: bool = False) -> str:
    extra_const = "    clearance_bonus = 0.2\n" if clearance_bonus else ""
    extra_term = ' "clearance_bonus": clearance_bonus,' if clearance_bonus else ""
    return (
        "def compute_reward(state, memory):\n"
        f"    collision_penalty = {collision}\n"
        f"    progress_coef = {coef}\n"
        f"{extra_const}"
        "    if state.collision:\n"
        "        total = float(collision_penalty)\n"
        '        return total, {"collision_penalty": total, "progress_term": 0.0}\n'
        "    progress_term = float(progress_coef * state.progress)\n"
        "    reach_term = float(state.reached_goal)\n"
        "    total = progress_term + reach_term\n"
        f'    return total, {{"collision_penalty": 0.0, "progress_term": progress_term,'
        f' "reach_term": reach_term,{extra_term}}}\n'
    )


def _episodes(n_success: int, n_collision: int, *, components: bool = True) -> list:
    eps = []
    for i in range(n_success):
        e = {
            "outcome": "success", "steps": 40 + i, "time": 10.0 + 0.25 * i,
            "path_length": 8.5, "min_separation": 0.15, "danger_steps": 1,
            "return": 1.6 + 0.01 * i, "disc_return": 1.2 + 0.01 * i,
            "task_return": 0.4, "selection_score": 1.0,
        }
        if components:
            e["components"] = {"progress_term": 0.6 + 0.01 * i, "reach_term": 1.0, "collision_penalty": 0.0}
        eps.append(e)
    for i in range(n_collision):
        e = {
            "outcome": "collision", "steps": 12 + i, "time": 3.0 + 0.25 * i,
            "path_length": 3.0, "min_separation": -0.05, "danger_steps": 4,
            "return": -19.8 + 0.01 * i, "disc_return": -17.0 + 0.01 * i,
            "task_return": -0.25, "selection_score": -1.0,
        }
        if components:
            e["components"] = {"progress_term": 0.2 + 0.01 * i, "reach_term": 0.0, "collision_penalty": -20.0}
        eps.append(e)
    return eps


def _metrics(sr, cr, *, tr=0.0, nt=10.0, pl=8.5, itr=0.05, sd=0.12, n=30,
             episodes=None, se=0.05):
    hold = {
        "SR": sr, "CR": cr, "TR": tr, "NT": nt, "PL": pl, "ITR": itr, "SD": sd,
        "n_eval_episodes": n, "fitness_se": se, "task_return": 0.4 * sr - 0.25 * cr,
        "selection_score": sr - cr - 0.5 * tr,
    }
    m = {**hold, "holdout": dict(hold), "eval_profile": "holdout"}
    m["eval_context"] = {
        "split": "val", "case_set_size": 100, "human_num": 5, "scenario": "circle_crossing",
        "time_limit_s": 25.0, "time_step_s": 0.25, "ideal_nav_time_s": 8.0,
        "deterministic": False,
    }
    if episodes is not None:
        m["holdout_episodes"] = episodes
    return attach_fitness(m)


def _dynamics(srs=(0.1, 0.3, 0.5, 0.6)) -> dict:
    return {
        "source": "training_rollouts", "total_timesteps": 12000, "n_episodes": 40,
        "windows": [
            {"steps": [3000 * i, 3000 * (i + 1)], "n_episodes": 10, "SR": s, "CR": 1 - s,
             "TR": 0.0, "mean_return": 0.5 + 0.1 * i, "mean_length": 40.0, "nav_time": 11.0}
            for i, s in enumerate(srs)
        ],
    }


def _cand(cid, *, origin="mutation", parents=(), code=None, metrics=None, dynamics=None):
    md = {}
    if metrics is not None:
        md.update(last_metrics=metrics, fitness=metrics["fitness"], train_steps=12000,
                  score1_scenario_scores={"safety": 0.8}, score1_n_pairs=120)
    if dynamics is not None:
        md["training_dynamics"] = dynamics
    return RewardCandidate(
        candidate_id=cid, code=code or _code(), valid=True, origin=origin,
        parent_ids=tuple(parents), metadata=md,
    )


def _lineage_archive() -> CandidateArchive:
    """ini_0000 -> mut_0004 -> mut_0009 (subject), plus a sibling and an outsider."""
    archive = CandidateArchive()
    archive.add_population(
        [
            _cand("ini_0000", origin="initial", code=_code(-10.0, 0.05),
                  metrics=_metrics(0.2, 0.8, nt=13.0, pl=10.0, itr=0.15,
                                   episodes=_episodes(6, 24))),
            _cand("ran_0002", origin="random", code=_code(-5.0, 0.2),
                  metrics=_metrics(0.9, 0.1, nt=9.0, pl=8.2)),
        ],
        generation=0,
    )
    archive.add_population(
        [
            _cand("mut_0004", parents=("ini_0000",), code=_code(-20.0, 0.05),
                  metrics=_metrics(0.4, 0.6, nt=11.0, pl=9.0,
                                   episodes=_episodes(12, 18))),
        ],
        generation=1,
    )
    archive.add_population(
        [
            _cand("mut_0009", parents=("mut_0004",), code=_code(-60.0, 0.08, clearance_bonus=True),
                  metrics=_metrics(0.63, 0.37, episodes=_episodes(19, 11)), dynamics=_dynamics()),
            _cand("mut_0010", parents=("mut_0004",), code=_code(-20.0, 0.12),
                  metrics=_metrics(0.5, 0.5, nt=10.5)),
        ],
        generation=2,
    )
    return archive


# Pin rich-card defaults so lean raise.yaml (history=1, traj=1, tight TPM)
# does not shrink integration expectations in these unit tests.
_FULL_CARD_CFG = dict(
    history_entries=3,
    max_trajectories=3,
    max_input_tokens=6000,
    target_input_tokens=5500,
    training_dynamics=True,
    component_instrumentation=True,
    random_calls=True,
)


def _card(archive=None, call_type="mutation", subjects=("mut_0009",), **cfg) -> fc.FeedbackCard:
    archive = archive or _lineage_archive()
    merged = {**_FULL_CARD_CFG, **cfg}
    return fc.FeedbackCard(
        call_type=call_type,
        subjects=[archive.get(s) for s in subjects],
        archive=archive,
        config=FeedbackCardConfig(**merged),
        population_note="Population(fitness desc): mut_0009: SR=0.63 ...",
    )


def _full(card: fc.FeedbackCard) -> fc.RenderedCard:
    return card.render(fc.truncation_ladder(card.config)[0])


class _RecordingLLM(LLMClient):
    provider = "groq"
    model = "mock-model"
    system_prompt = CLIENT_SYSTEM

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list = []

    def complete(self, prompt, *, max_tokens=None):
        self.prompts.append(prompt)
        return self.reply


_REPLY = "CR is high; raise the collision cost.\n```python\n" + _code(-80.0) + "```\n"
_BATCH_REPLY = (
    "```python\n"
    + _code(-30.0).replace("compute_reward(", "compute_reward_v1(")
    + "\n"
    + _code(-40.0).replace("compute_reward(", "compute_reward_v2(")
    + "```\n"
)


def _evolver(tmp_path, *, reply=_REPLY, archive=None, **cfg) -> StageIEvolver:
    merged = {**_FULL_CARD_CFG, **cfg}
    evo = StageIEvolver(
        _RecordingLLM(reply),
        score_fn=lambda *_a, **_k: 0.0,
        validator=make_crowdnav_validator(),
        config=StageIConfig(population_size=4),
        prompts=cn_prompts,
        feedback=FeedbackCardConfig(**merged),
        feedback_card=fc,
        llm_log_dir=str(tmp_path / "closed_loop"),
    )
    if archive is not None:
        evo.archive = archive
    return evo


def _calls(tmp_path) -> list:
    path = tmp_path / "closed_loop" / CALLS_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _requests(tmp_path) -> list:
    return [r for r in _calls(tmp_path) if r["event"] == "request"]


# ------------------------------------------------------------------ 1. full card


def test_full_card_renders_every_section_from_recorded_data():
    rendered = _full(_card())
    text = rendered.text
    assert rendered.sections == ["A", "B", "C", "D", "E", "F", "G", "H", "P"]
    assert rendered.unavailable == {}
    # A: metrics with units + evaluation context.
    assert (
        "Eval: Stage II PPO 12,000 steps; val scenarios (circle_crossing, 5 humans); "
        "30 episodes; stochastic policy." in text
    )
    assert "nav_time 10.00 s" in text and "path_length 8.50 m" in text
    assert "min_dist_in_danger 0.120 m" in text
    # B: training windows; C: per-outcome component sums with coverage.
    assert "steps 0-3000: 10 ep SR 0.10" in text and "nav_time 11.0 s" in text
    # 11 of 30 episodes collide at -20: mean -7.33, successes 0, failures -20.
    assert "collision_penalty -7.33 | +0.00 | -20.00" in text
    assert "Components sum to" in text
    # D: parent comparison incl. code change.
    assert "mut_0009 vs parent mut_0004" in text
    assert "collision_penalty -20->-60" in text and "+component clearance_bonus" in text
    # F: representative episodes; H: preference check.
    assert "typical success" in text and "typical collision" in text
    assert "Mann-Whitney AUC over 19x11 holdout episode pairs, discounted, gamma 0.99" in text
    assert "outcome-ordering AUC 0.90" in text
    assert "Observation:" in text and fc._GUIDANCE in text


# ------------------------------------------------------------------ 2. missing data


def test_missing_data_is_marked_unavailable_never_fabricated():
    archive = CandidateArchive()
    bare = RewardCandidate(
        candidate_id="ini_0001", code=_code(), valid=True, origin="initial",
        metadata={"last_metrics": {"SR": 0.5, "CR": 0.5, "TR": 0.0, "NT": 12.0, "PL": 10.0},
                  "fitness": -0.2},
    )
    archive.add_population([bare], generation=0)
    rendered = _full(_card(archive, subjects=("ini_0001",)))
    text = rendered.text
    assert set(rendered.unavailable) >= {"B", "C", "E", "F", "H"}
    assert "[B] Training dynamics - ini_0001: unavailable (no training-rollout statistics" in text
    assert "[C] Reward components - ini_0001: unavailable (no per-episode holdout records" in text
    assert "[D] Parent comparison: unavailable (ini_0001 is a fresh initial proposal" in text
    assert "[F] Representative episodes: unavailable" in text
    assert "danger_freq n/a" in text and "min_dist_in_danger n/a" in text
    assert "steps 0-" not in text and "typical success" not in text


def test_disabled_instrumentation_is_reported_as_disabled():
    text = _full(_card(training_dynamics=False, component_instrumentation=False)).text
    assert "[B] Training dynamics - mut_0009: unavailable (disabled in config)" in text
    assert "[C] Reward components - mut_0009: unavailable (disabled in config)" in text


def test_unevaluated_subject_keeps_lineage_but_no_numbers():
    archive = _lineage_archive()
    archive.add(ArchiveRecord("mut_0011", 3, "mutation", ("mut_0009",), _code(-70.0)))
    text = _full(_card(archive, subjects=("mut_0011",))).text
    assert "unavailable: no Stage II evaluation" in text
    assert "[D] mut_0011 vs parent mut_0009: unavailable (missing evaluation)" in text
    assert "Flags: insufficient evidence" in text


# ------------------------------------------------------------------ 3. history / lineage


def test_history_prefers_relatives_over_recent_outsiders():
    archive = _lineage_archive()
    picks = archive.select_history(["mut_0009"], 3, exclude=["mut_0004"])
    ids = [rec.candidate_id for rec, _ in picks]
    relations = dict((rec.candidate_id, rel) for rec, rel in picks)
    assert ids[:2] == ["ini_0000", "mut_0010"]
    assert relations["ini_0000"] == "grandparent of mut_0009"
    assert relations["mut_0010"].startswith("sibling of mut_0009")
    # The unrelated random candidate only fills the remaining slot.
    assert ids[2] == "ran_0002" and relations["ran_0002"].startswith("non-relative")


def test_history_includes_earlier_children_of_the_parent_being_mutated():
    archive = _lineage_archive()
    picks = archive.select_history(["mut_0004"], 3, exclude=["ini_0000"])
    rels = {rec.candidate_id: rel for rec, rel in picks}
    assert rels["mut_0009"] == "earlier child of mut_0004"
    assert rels["mut_0010"] == "earlier child of mut_0004"


def test_history_has_no_duplicates_and_skips_subject_and_parents():
    archive = _lineage_archive()
    # Diamond: crossover child of two relatives reaches ini_0000 twice.
    archive.add_population(
        [_cand("cro_0012", origin="crossover", parents=("mut_0009", "mut_0010"),
               metrics=_metrics(0.7, 0.3))],
        generation=3,
    )
    card = _card(archive, subjects=("cro_0012",), history_entries=5)
    rendered = _full(card)
    assert len(rendered.history_ids) == len(set(rendered.history_ids))
    assert not {"cro_0012", "mut_0009", "mut_0010"} & set(rendered.history_ids)
    assert rendered.history_ids[0] == "mut_0004"


def test_history_entries_carry_metrics_behaviour_and_change_vs_parent():
    text = _full(_card()).text
    entry = text[text.index("[E] History"):text.index("[F]")]
    assert "- ini_0000 | grandparent of mut_0009 | gen 0 initial" in entry
    assert "SR 0.20 CR 0.80" in entry and "crash-prone" in entry
    assert "vs its parent mut_0004: SR +0.10" in entry
    assert "code: 10 lines; components [collision_penalty" in entry


# ------------------------------------------------------------------ 4. deltas


def test_parent_deltas_respect_metric_direction_and_noise():
    text = _full(_card()).text
    d = text[text.index("[D] mut_0009"):text.index("[E]")]
    # 30 episodes each: SE(rate) ~ 0.09, so a 0.23 change stays inside 2 SE.
    assert "SR 0.40->0.63 (+0.23 better, within noise)" in d
    assert "CR 0.60->0.37 (-0.23 better, within noise)" in d
    assert "nav_time 11.00->10.00 (-1.00 better)" in d
    assert "path_length 9.00->8.50 (-0.50 better)" in d
    assert "TR 0.00 (same)" in d
    assert "danger_freq 0.050 (same)" in d
    assert (
        "Behaviour: rarely reaches the goal, crash-prone -> "
        "reaches the goal in most episodes, some collisions" in d
    )


def test_large_change_with_many_episodes_is_not_noise():
    archive = CandidateArchive()
    archive.add(ArchiveRecord.from_candidate(_cand("a", metrics=_metrics(0.40, 0.60, n=200, itr=0.08)), generation=0))
    archive.add(ArchiveRecord.from_candidate(
        _cand("b", parents=("a",), metrics=_metrics(0.63, 0.37, n=200, itr=0.03)), generation=1))
    text = _full(_card(archive, subjects=("b",))).text
    assert "SR 0.40->0.63 (+0.23 better)" in text
    assert "CR 0.60->0.37 (-0.23 better)" in text
    # Only fitness names its SE mode; no per-case episodes here, so independent.
    assert "fitness -0.200->0.260 (+0.460 better, independent)" in text
    # Danger frequency is not a rate with a binomial SE: no noise verdict.
    assert "danger_freq 0.080->0.030 (-0.050 better)" in text


def test_small_rate_change_is_within_noise():
    archive = CandidateArchive()
    archive.add(ArchiveRecord.from_candidate(_cand("a", metrics=_metrics(0.60, 0.40)), generation=0))
    archive.add(ArchiveRecord.from_candidate(
        _cand("b", parents=("a",), metrics=_metrics(0.65, 0.35)), generation=1))
    text = _full(_card(archive, subjects=("b",))).text
    assert "SR 0.60->0.65 (+0.05 better, within noise)" in text


def test_crossover_compares_parent_a_with_parent_b():
    text = _full(_card(call_type="crossover", subjects=("mut_0009", "mut_0010"))).text
    assert "[D] Parent A mut_0009 vs parent B mut_0010 (B - A;" in text
    assert "CR 0.37->0.50 (+0.13 worse, within noise)" in text
    assert "Code difference A->B:" in text


# ------------------------------------------------------------------ 5. diagnostics / preference


def test_preference_check_auc_and_single_class():
    rec = ArchiveRecord.from_candidate(
        _cand("x", metrics=_metrics(0.5, 0.5, episodes=_episodes(2, 2))), generation=0)
    pref, _ = fc.preference_check(rec)
    assert pref["auc"] == 1.0 and (pref["n_pos"], pref["n_neg"]) == (2, 2)
    eps = _episodes(2, 2)
    eps[2]["disc_return"] = 500.0  # one crash out-earns both successes
    rec2 = ArchiveRecord.from_candidate(_cand("y", metrics=_metrics(0.5, 0.5, episodes=eps)), generation=0)
    assert fc.preference_check(rec2)[0]["auc"] == pytest.approx(0.5)
    rec3 = ArchiveRecord.from_candidate(
        _cand("z", metrics=_metrics(1.0, 0.0, episodes=_episodes(3, 0))), generation=0)
    pref3, reason = fc.preference_check(rec3)
    assert pref3 is None and "3 successful and 0 failed" in reason


def test_diagnostics_flag_slow_freezing_and_collisions():
    archive = CandidateArchive()
    archive.add_population(
        [
            _cand("slow", metrics=_metrics(0.95, 0.05, nt=20.0, itr=0.02)),
            _cand("freeze", metrics=_metrics(0.1, 0.0, tr=0.9, itr=0.0)),
            _cand("crash", metrics=_metrics(0.4, 0.6, itr=0.2)),
        ],
        generation=0,
    )
    slow = _full(_card(archive, subjects=("slow",))).text
    assert "Flags: slow navigation" in slow
    assert "Observation: slow: nav_time 20.00 s vs 8.00 s for a straight line at v_pref" in slow
    assert "Hypothesis: slow: the reward does not pay for arriving sooner" in slow
    assert "slow arrivals" in slow
    freeze = _full(_card(archive, subjects=("freeze",))).text
    assert "Flags: freezing robot (timeouts)" in freeze
    assert "freezes until the time limit" in freeze
    crash = _full(_card(archive, subjects=("crash",))).text
    assert "Flags: collisions, discomfort-zone intrusions" in crash
    assert "Observation: crash: CR 0.60 (SR 0.40, TR 0.00)" in crash
    assert "often inside the discomfort zone" in crash


def test_unstable_training_and_reward_hacking_flags():
    archive = CandidateArchive()
    eps = _episodes(5, 5)
    for e in eps:
        if e["outcome"] == "collision":
            e["disc_return"] = 1000.0
    archive.add_population(
        [_cand("u", metrics=_metrics(0.5, 0.5, episodes=eps), dynamics=_dynamics((0.2, 0.8, 0.6, 0.3)))],
        generation=0,
    )
    text = _full(_card(archive, subjects=("u",))).text
    assert "unstable training" in text and "possible reward hacking" in text
    assert "training-rollout SR peaked at 0.80 and ended at 0.30" in text
    assert "failed episodes out-earn successful ones (preference AUC 0.00, 5x5 pairs)" in text


def test_code_summary_and_diff():
    assert "components [collision_penalty, progress_term, reach_term]" in summarize_code(_code())
    assert diff_code(_code(), _code()) == "identical code"
    assert diff_code(_code(-20.0), _code(-60.0, clearance_bonus=True)) == (
        "+component clearance_bonus; collision_penalty -20->-60; +clearance_bonus=0.2"
    )
    assert diff_code(_code(), _code() + "\n").startswith("identical")
    assert summarize_code("def f(:\n").endswith("(unparseable)")


# ------------------------------------------------------------------ 6. token counting


def test_token_count_covers_full_request():
    counter = make_token_counter("groq")
    user = "x " * 50
    with_sys = counter.count_request([CLIENT_SYSTEM, user])
    without = counter.count_request(["", user])
    assert with_sys - without == counter.count(CLIENT_SYSTEM)
    assert without > counter.count(user)  # template overhead included


def test_counter_falls_back_to_heuristic_without_tokenizer(monkeypatch):
    monkeypatch.setattr(token_budget, "_o200k_encoding", lambda: None)
    counter = make_token_counter("groq")
    assert counter.method.startswith("heuristic")
    text = cn_prompts.D1_SYSTEM_PROMPT
    assert counter.count(text) == math.ceil(heuristic_token_count(text) * token_budget.HEURISTIC_MARGIN)


@pytest.mark.skipif(tiktoken is None, reason="tiktoken not installed")
def test_heuristic_fallback_is_conservative_on_real_prompts():
    counter = make_token_counter("groq")
    card = _full(_card()).text
    for text in (
        cn_prompts.D1_SYSTEM_PROMPT,
        cn_prompts.format_d2_mutation(_code(), card),
        cn_prompts.format_d1_initial(reflection=card),
    ):
        assert heuristic_token_count(text) >= counter.count(text)


# ------------------------------------------------------------------ 7/8. budget


def _render_fn(card, user_builder, counter_system=CLIENT_SYSTEM):
    ladder = fc.truncation_ladder(card.config)

    def render(level):
        if level >= len(ladder):
            return None
        trim = ladder[level]
        rc = card.render(trim)
        user = user_builder(rc.text, trim.compact_instructions)
        return [counter_system, f"{cn_prompts.D1_SYSTEM_PROMPT}\n\n{user}"], trim.name, {
            "sections": rc.sections, "history_ids": rc.history_ids,
        }

    return render, ladder


def _mutation_user(card_text, compact):
    return cn_prompts.format_d2_mutation(_code(-60.0, clearance_bonus=True), card_text, compact=compact)


def test_truncation_follows_priority_order():
    card = _card()
    counter = make_token_counter("groq")
    render, ladder = _render_fn(card, _mutation_user)
    full_tokens = counter.count_request(render(0)[0])
    names = [t.name for t in ladder]
    assert names[1:4] == ["trajectories->2", "trajectories->1", "trajectories->0"]
    assert names[4:6] == ["history code summaries dropped", "population context dropped"]
    assert names.index("instructions compacted") > names.index("history entries->0")
    assert names[-1] == "diagnostic summary dropped"

    fit = fit_to_budget(render, counter, max_input_tokens=6000, target_input_tokens=full_tokens - 1)
    assert fit.steps == ["trajectories->2"]
    previous = 0
    for target in range(full_tokens - 1, 1500, -150):
        try:
            fit = fit_to_budget(render, counter, max_input_tokens=target, target_input_tokens=target)
        except PromptBudgetError:
            break
        # Applied steps keep ladder order (no-op levels are not logged as cuts).
        positions = [names.index(s) for s in fit.steps]
        assert positions == sorted(positions)
        assert len(fit.steps) >= previous
        previous = len(fit.steps)
        user = fit.messages[1]
        assert "[A] Current performance" in user and "[D] mut_0009 vs parent" in user
        assert "def compute_reward(state, memory):" in user  # current code kept
        assert "Action space (this run)" in user or "Sandbox rules" in user


def test_no_op_levels_are_not_logged_as_cuts():
    archive = CandidateArchive()
    archive.add_population([_cand("a", metrics=_metrics(0.5, 0.5))], generation=0)  # no episodes
    render, _ = _render_fn(_card(archive, subjects=("a",)), _mutation_user)
    counter = make_token_counter("groq")
    fit = fit_to_budget(render, counter, max_input_tokens=6000,
                        target_input_tokens=counter.count_request(render(0)[0]) - 1)
    # Trajectory levels 3->2->1 render the same "unavailable" line; only ->0 removes it.
    assert fit.steps == ["trajectories->0"]


def test_hard_limit_never_exceeded():
    card = _card()
    counter = make_token_counter("groq")
    render, _ = _render_fn(card, _mutation_user)
    sent = refused = 0
    for limit in range(800, 7001, 200):
        target = max(1, limit - 300)
        try:
            fit = fit_to_budget(render, counter, max_input_tokens=limit, target_input_tokens=target)
        except PromptBudgetError as exc:
            refused += 1
            assert exc.fit.tokens > limit
            continue
        sent += 1
        assert fit.tokens <= limit
        assert counter.count_request(fit.messages) == fit.tokens
    assert sent and refused


def test_config_validation():
    with pytest.raises(ValueError):
        FeedbackCardConfig(max_input_tokens=5000, target_input_tokens=5500)
    with pytest.raises(ValueError):
        FeedbackCardConfig(history_entries=-1)


# ------------------------------------------------------------------ 9. integration per call type


def _pop(archive):
    return {cid: _cand(cid, parents=archive.get(cid).parent_ids, code=archive.get(cid).code,
                       metrics=archive.get(cid).metrics or None)
            for cid in ("mut_0009", "mut_0010", "mut_0004")}


def test_every_call_type_gets_the_card_within_budget(tmp_path):
    archive = _lineage_archive()
    evo = _evolver(tmp_path, archive=archive)
    pop = _pop(archive)
    llm = evo.llm

    mut = evo._try_mutate(pop["mut_0009"], "", phase="mutation", attempt=1)
    evo.llm.reply = _REPLY.replace("-80.0", "-81.0")  # identical children are rejected
    cro = evo._try_crossover(pop["mut_0009"], pop["mut_0010"], "", phase="crossover", attempt=1)
    evo.llm.reply = _REPLY.replace("-80.0", "-82.0")
    ran = evo._try_random_restart(phase="random", attempt=1)
    evo.llm.reply = _BATCH_REPLY
    batch = evo._fill_random_batch(2)

    assert mut.valid and cro.valid and ran.valid and len(batch) == 2
    assert "DIAGNOSTIC FEEDBACK CARD (mutation)" in llm.prompts[0]
    assert "DIAGNOSTIC FEEDBACK CARD (crossover)" in llm.prompts[1]
    assert "DIAGNOSTIC FEEDBACK CARD (reference)" in llm.prompts[2]
    assert "DIAGNOSTIC FEEDBACK CARD (reference)" in llm.prompts[3]
    # Output schema and sandbox constraints stay in every prompt.
    for prompt in llm.prompts:
        assert prompt.startswith(cn_prompts.D1_SYSTEM_PROMPT)
        assert "(total_reward, reward_components)" in prompt or "dict[str, float]" in prompt

    requests = _requests(tmp_path)
    assert [r["call_type"] for r in requests] == ["mutation", "crossover", "random", "random_batch"]
    counter = make_token_counter("groq")
    for req, prompt in zip(requests, llm.prompts):
        assert req["status"] == "sent" and req["feedback_card"] is True
        assert req["input_tokens"] <= 6000
        assert req["input_tokens"] == counter.count_request([CLIENT_SYSTEM, prompt])
        assert req["provider"] == "groq" and req["model"] == "mock-model"
        assert req["token_method"] == "tiktoken:o200k_harmony" or tiktoken is None
        assert "A" in req["sections"] and "D" in req["sections"]
        snap = tmp_path / "closed_loop" / req["prompt_snapshot"]
        assert prompt in snap.read_text(encoding="utf-8")
    assert requests[0]["subject_ids"] == ["mut_0009"]
    assert requests[0]["history_ids"][:2] == ["ini_0000", "mut_0010"]
    assert requests[1]["subject_ids"] == ["mut_0009", "mut_0010"]

    validations = [r for r in _calls(tmp_path) if r["event"] == "validation"]
    assert {v["candidate_id"] for v in validations} >= {mut.candidate_id, cro.candidate_id}
    assert all(v["validator_passed"] for v in validations)
    assert mut.metadata["llm_call_id"] == requests[0]["call_id"]
    assert "weakness" not in mut.metadata


def test_gen0_without_evidence_sends_plain_prompt(tmp_path):
    evo = _evolver(tmp_path, reply=_BATCH_REPLY)
    out = evo.initialize_population()
    assert len(out) == 4
    first = evo.llm.prompts[0]
    assert "DIAGNOSTIC FEEDBACK CARD" not in first
    req = _requests(tmp_path)[0]
    assert req["call_type"] == "initial_batch" and req["feedback_card"] is False
    assert req["sections"] == [] and req["truncation_steps"] == []


def test_tight_budget_truncates_and_logs_cuts(tmp_path):
    archive = _lineage_archive()
    evo = _evolver(tmp_path, archive=archive)
    counter = make_token_counter("groq")
    render, _ = _render_fn(_card(archive), _mutation_user)
    limit = counter.count_request(render(0)[0]) - 400
    evo.feedback = FeedbackCardConfig(
        **{**_FULL_CARD_CFG, "max_input_tokens": limit, "target_input_tokens": limit}
    )
    evo._try_mutate(_pop(archive)["mut_0009"], "", phase="mutation", attempt=1)
    req = _requests(tmp_path)[0]
    assert req["input_tokens"] <= limit
    assert req["truncation_steps"][:3] == ["trajectories->2", "trajectories->1", "trajectories->0"]
    assert "A" in req["sections"] and "D" in req["sections"] and "F" not in req["sections"]


def test_over_budget_request_is_never_sent_and_falls_back(tmp_path):
    archive = _lineage_archive()
    evo = _evolver(tmp_path, archive=archive, max_input_tokens=900, target_input_tokens=900)
    parent = _pop(archive)["mut_0009"]
    out = evo._fill_valid(
        1,
        lambda n: evo._try_mutate(parent, "", phase="mutation", attempt=n),
        max_extra=3,
        phase="mutation",
        fallback=lambda: evo._clone_valid(parent.code, origin="mutation_fallback",
                                          parent_ids=(parent.candidate_id,)),
    )
    assert evo.llm.prompts == []
    assert out[0].origin == "mutation_fallback"
    reqs = _requests(tmp_path)
    assert len(reqs) == 1 and reqs[0]["status"] == "refused_over_budget"
    assert reqs[0]["input_tokens"] > 900


# ------------------------------------------------------------------ 10. disabled card


def test_disabled_card_sends_the_legacy_prompt_exactly(tmp_path):
    archive = _lineage_archive()
    evo = _evolver(tmp_path, archive=archive, enabled=False)
    parent = _pop(archive)["mut_0009"]
    parent.metadata["proxy_feedback"] = "Holdout eval (30 episodes): SR=0.63"
    evo._try_mutate(parent, "note", phase="mutation", attempt=1)
    weakness = "Parent mut_0009.\nHoldout eval (30 episodes): SR=0.63\nGlobal reflection: note"
    expected = f"{cn_prompts.D1_SYSTEM_PROMPT}\n\n" + cn_prompts.format_d2_mutation(parent.code, weakness)
    assert evo.llm.prompts == [expected]
    req = _requests(tmp_path)[0]
    assert req["feedback_card"] is False and req["status"] == "sent"


def test_runner_fast_loop_logs_cards_and_persists_archive(tmp_path):
    out = str(tmp_path / "run")
    cfg = ClosedLoopConfig(population_size=4, generations=2, stage2_train_steps=32,
                           use_stub=True, llm_provider="seed", output_dir=out, device="cpu")
    ClosedLoopRunner(cfg).run()
    cl = os.path.join(out, "closed_loop")
    with open(os.path.join(cl, CALLS_FILE), encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    requests = [r for r in records if r["event"] == "request"]
    assert requests[0]["feedback_card"] is False  # Gen0: no evidence yet
    later = [r for r in requests if r["call_type"] in ("mutation", "crossover")]
    assert later and all(r["feedback_card"] for r in later)
    assert all(r["input_tokens"] <= 6000 and r["status"] == "sent" for r in requests)
    # Stub trainer records no episodes/dynamics: those sections say so.
    assert all("B" in r["unavailable_sections"] for r in later)
    with open(os.path.join(cl, "pop_epoch_000.json"), encoding="utf-8") as fh:
        row = json.load(fh)["candidates"][0]
    assert row["code"].startswith("def compute_reward") and "generation" in row
    rebuilt = CandidateArchive.from_epoch_files(cl)
    assert len(rebuilt) >= 4 and all(r.code for r in rebuilt.records())


# ------------------------------------------------------------------ instrumentation


class _FakeEnv:
    """Minimal stand-in for RewardInjectedCrowdNavEnv (scripted steps, robot moves +1 m in x)."""

    time_step = 0.25
    v_pref = 1.0
    time_limit = 25.0

    def __init__(self, plans):
        self.plans = plans
        self.ep = -1
        self.sim = SimpleNamespace(robot=SimpleNamespace(px=0.0, py=0.0))
        self.last_state = None

    def reset(self, *, seed=None, options=None):
        self.ep += 1
        self.t = 0
        self.sim.robot.px = 0.0
        return np.zeros(3), {}

    def step(self, action):
        reward, info = self.plans[self.ep][self.t]
        self.t += 1
        self.sim.robot.px += 1.0
        self.last_state = SimpleNamespace(robot=SimpleNamespace(px=self.sim.robot.px, py=0.0))
        terminated = info["raise_outcome"] in ("success", "collision")
        truncated = info["raise_outcome"] == "timeout"
        return np.zeros(3), reward, terminated, truncated, {**info, "raise_global_time": self.t * self.time_step}


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return 0, None


def _info(outcome=None, native=0.0, danger=None, comps=None):
    return {
        "raise_outcome": outcome, "raise_native_reward": native,
        "raise_min_separation": 0.1 if danger is None else danger,
        "raise_danger": danger is not None, "raise_danger_min_dist": danger,
        "raise_reward_components": comps or {},
    }


@pytest.mark.parametrize("record_components", [True, False])
def test_eval_metrics_keeps_per_episode_records(record_components):
    plans = [
        [(1.0, _info(comps={"a": 0.5, "b": 0.5})), (1.0, _info(danger=0.05, comps={"a": 0.5, "b": 0.5})),
         (1.0, _info("success", native=1.0, comps={"a": 0.5, "b": 0.5}))],
        [(1.0, _info(comps={"a": 1.0})), (-5.0, _info("collision", native=-0.25, comps={"crash": -5.0}))],
    ]
    metrics = _eval_metrics(_FakeEnv(plans), _FakeModel(), cases=[0, 1],
                            record_components=record_components)
    ok, crash = metrics._crowdnav_episodes
    assert ok["outcome"] == "success" and crash["outcome"] == "collision"
    assert ok["return"] == pytest.approx(3.0)
    assert ok["disc_return"] == pytest.approx(1 + PPO_GAMMA + PPO_GAMMA ** 2, abs=1e-4)
    # Native task return stays diagnostic; selection uses terminal outcomes.
    assert ok["task_return"] == pytest.approx(1.0)
    assert crash["task_return"] == pytest.approx(-0.25)
    assert ok["selection_score"] == 1.0 and crash["selection_score"] == -1.0
    assert crash["steps"] == 2 and ok["path_length"] == pytest.approx(3.0)
    assert ok["danger_steps"] == 1 and ok["time"] == pytest.approx(0.75)
    if record_components:
        assert ok["components"] == {"a": 1.5, "b": 1.5}
        assert crash["components"] == {"a": 1.0, "crash": -5.0}
    else:
        assert "components" not in ok
    assert metrics.sr == 0.5 and metrics.cr == 0.5
    assert metrics.nt == pytest.approx(0.75) and metrics.pl == pytest.approx(3.0)
    assert metrics.itr == pytest.approx(1 / 5) and metrics.sd == pytest.approx(0.05)


def test_training_dynamics_recorder_windows():
    rec = TrainingDynamicsRecorder(100, n_windows=4)
    outcomes = {10: "collision", 30: "success", 90: "timeout"}
    for t in range(1, 101):
        done = t in outcomes
        info = {"raise_outcome": outcomes.get(t), "raise_global_time": 2.5}
        if done:
            info["episode"] = {"r": float(t), "l": 10}
        rec.observe([info], [done], t)
    summary = rec.summary()
    w = summary["windows"]
    assert summary["n_episodes"] == 3
    assert w[0]["n_episodes"] == 1 and w[0]["CR"] == 1.0
    assert w[1]["n_episodes"] == 1 and w[1]["SR"] == 1.0 and w[1]["nav_time"] == 2.5
    assert w[2]["n_episodes"] == 0
    assert w[3]["TR"] == 1.0 and w[3]["mean_return"] == 90.0 and w[3]["nav_time"] is None

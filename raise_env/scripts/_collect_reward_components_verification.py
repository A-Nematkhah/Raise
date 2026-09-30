"""Collect evidence for REWARD_COMPONENTS_VERIFICATION.md. Run from raise_env/."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "results" / "_reward_components_verification"
OUT.mkdir(parents=True, exist_ok=True)


def w(name: str, text: str) -> None:
    path = OUT / name
    path.write_text(text, encoding="utf-8")
    print(f"WROTE {path} ({len(text)} chars)")


def section_adversarial() -> None:
    from domains.highway.reward_checks import make_highway_validator
    from domains.highway.state import default_smoke_states
    from raise_core.sandbox.errors import RewardSandboxError
    from raise_core.sandbox.runtime import MAX_REWARD_COMPONENT_KEYS

    v = make_highway_validator(smoke_states=default_smoke_states())
    cases = []

    def try_code(label: str, code: str) -> None:
        try:
            rw = v.validate_code(code)
            st = default_smoke_states()[0]
            total = rw.compute(st)
            comps = rw.last_reward_components()
            cases.append(
                f"=== {label} ===\nPASS total={total!r} comps={comps!r}\n"
            )
        except RewardSandboxError as exc:
            cases.append(f"=== {label} ===\nFAIL: {exc}\n")
        except Exception as exc:  # noqa: BLE001
            cases.append(f"=== {label} ===\nFAIL({type(exc).__name__}): {exc}\n")

    keys15 = ", ".join(f'"k{i}": 1.0' for i in range(15))
    try_code(
        "1_over_cap_15_keys",
        f"def compute_reward(state, memory):\n    return 1.0, {{{keys15}}}\n",
    )
    try_code(
        "2_non_string_key",
        "def compute_reward(state, memory):\n    return 1.0, {1: 2.0}\n",
    )
    try_code(
        "3_nested_dict_value",
        'def compute_reward(state, memory):\n    return 1.0, {"a": {"b": 1.0}}\n',
    )
    try_code(
        "4_dict_comprehension",
        "def compute_reward(state, memory):\n"
        '    return 1.0, {str(i): float(i) for i in range(3)}\n',
    )
    try_code(
        "5_non_finite_value",
        'def compute_reward(state, memory):\n    return 1.0, {"a": float("nan")}\n',
    )
    try_code(
        "5b_inf_value",
        'def compute_reward(state, memory):\n    return 1.0, {"a": float("inf")}\n',
    )
    try_code(
        "6_non_numeric_total",
        'def compute_reward(state, memory):\n    return "x", {"a": 1.0}\n',
    )
    try_code(
        "7_bare_float_shim",
        "def compute_reward(state, memory):\n    return 2.5\n",
    )
    try_code(
        "build_locals_then_return_eureka_style",
        """def compute_reward(state, memory):
    progress_term = float(state.progress)
    speed_term = float(0.08 * state.speed)
    collision_penalty = -20.0 if state.collision else 0.0
    total_reward = progress_term + speed_term + collision_penalty
    reward_components = {
        "progress_term": progress_term,
        "speed_term": speed_term,
        "collision_penalty": collision_penalty,
    }
    return total_reward, reward_components
""",
    )
    cases.append(f"MAX_REWARD_COMPONENT_KEYS={MAX_REWARD_COMPONENT_KEYS}\n")
    w("section2_adversarial.txt", "\n".join(cases))


def section_prompts_and_evidence() -> None:
    from domains.highway.prompts import (
        D1_USER_PROMPT,
        D5_SEED_FUNCTION,
        _DIAGNOSIS_BEFORE_CODE,
        format_d1_initial,
        format_d2_mutation,
    )
    from raise_core.raise_loop.proxy_feedback import evidence_block

    rendered_d1 = format_d1_initial(
        func_name="compute_reward",
        include_seed=True,
        include_external_knowledge=True,
        reflection="",
    )
    # Extract example from D1 template + seed
    w("section3_d1_user_template.txt", D1_USER_PROMPT)
    w("section3_d1_rendered.txt", rendered_d1)
    w("section3_d5_seed.txt", D5_SEED_FUNCTION)
    w("section3_diagnosis_principles.txt", _DIAGNOSIS_BEFORE_CODE)

    metrics = {
        "SR": 0.45,
        "CR": 0.35,
        "TR": 0.05,
        "mean_speed": 23.4,
        "mean_progress": 95.0,
        "lane_change_rate": 0.12,
        "overtakes_per_km": 0.8,
        "overtake_episode_frac": 0.25,
        "n_eval_episodes": 20,
        "speed_p10": 19.0,
    }
    md = {
        "pareto_rank": 2,
        "pareto_n": 8,
        "pareto_feasible": True,
        "pareto_front": 0,
        "pareto_require_survival": True,
        "pareto_calibration_source": "no_speed_floor",
        "pareto_objectives": [
            "-CR",
            "-TR",
            "progress",
            "mean_speed",
            "lane_change_rate",
            "overtakes_per_km",
        ],
        "reward_component_trends": {
            "speed_term": {
                "values": [0.12, 0.31, 0.30, 0.34, 0.28],
                "max": 0.34,
                "mean": 0.27,
                "min": 0.12,
            },
            "collision_penalty": {
                "values": [0.0, 0.0, -2.5, 0.0, 0.0],
                "max": 0.0,
                "mean": -0.5,
                "min": -2.5,
            },
            "progress_term": {
                "values": [0.40, 0.55, 0.52, 0.61, 0.58],
                "max": 0.61,
                "mean": 0.532,
                "min": 0.40,
            },
        },
    }
    block = evidence_block(metrics, score1=0.612, metadata=md)
    w("section3_evidence_block.txt", block)

    mut = format_d2_mutation(
        """def compute_reward(state, memory):
    progress_term = float(state.progress)
    speed_term = float(0.08 * state.speed)
    total = progress_term + speed_term
    return total, {"progress_term": progress_term, "speed_term": speed_term}
""",
        "Parent r00 Score1=0.4\n" + block + "\nGlobal reflection: Gen1 trends",
    )
    w("section3_d2_mutation_full.txt", mut)

    # Grep principles for highway-specific nouns in the principle lines only
    principles = []
    for line in _DIAGNOSIS_BEFORE_CODE.splitlines():
        if line.strip().startswith("- If "):
            principles.append(line)
    w("section3_principle_lines_only.txt", "\n".join(principles))


def section_shim_and_aggregate_tests() -> None:
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "domains/highway/tests/test_eureka_components.py::test_highway_accepts_tuple_dict_and_shim_float",
        "domains/highway/tests/test_eureka_components.py::test_components_not_aggregated_outside_rollout_callback",
        "domains/highway/tests/test_eureka_components.py::test_constant_component_flat_in_trend_summary",
        "domains/highway/tests/test_eureka_components.py::test_evidence_block_and_mutation_include_component_trends",
        "raise_core/tests/test_sandbox_comprehension_ban.py",
        "-v",
        "--tb=short",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        cmd, cwd=str(ROOT), env=env, capture_output=True, text=True
    )
    w(
        "section1_pytest_key_tests.txt",
        proc.stdout + "\n" + proc.stderr + f"\nexit={proc.returncode}\n",
    )


def section_plateau_with_evidence() -> None:
    """Synthetic constant component → trends + fabricated varying SR in same block."""
    from domains.highway.diagnostics_callback import RolloutDiagnosticsCallback
    from raise_core.raise_loop.proxy_feedback import evidence_block

    src = '''def compute_reward(state, memory):
    """One component fixed; one varies with speed."""
    fixed_term = 0.42
    speed_term = float(0.05 * state.speed)
    collision_penalty = -20.0 if state.collision else 0.0
    total = fixed_term + speed_term + collision_penalty
    return total, {
        "fixed_term": fixed_term,
        "speed_term": speed_term,
        "collision_penalty": collision_penalty,
    }
'''
    cb = RolloutDiagnosticsCallback(candidate_id="plateau_demo", log_path=str(OUT / "plateau_rollout.jsonl"))
    # Simulate 5 rollouts with varying speed_term means, flat fixed_term
    for rollout in range(5):
        for step in range(10):
            speed = 15.0 + rollout * 2.0 + 0.3 * step
            cb._ingest_step_components(
                {
                    "fixed_term": 0.42,
                    "speed_term": 0.05 * speed,
                    "collision_penalty": 0.0 if step % 7 else -20.0,
                }
            )
        cb._flush_rollout_component_means()
    trends = cb.component_trend_summary()
    # Varying SR/CR across "checkpoints" shown as holdout metrics (final)
    metrics = {
        "SR": 0.55,
        "CR": 0.30,
        "TR": 0.05,
        "mean_speed": 24.0,
        "mean_progress": 110.0,
        "lane_change_rate": 0.1,
        "overtakes_per_km": 0.5,
        "overtake_episode_frac": 0.2,
        "n_eval_episodes": 20,
        "speed_p10": 18.0,
    }
    md = {
        "pareto_rank": 1,
        "pareto_n": 6,
        "pareto_feasible": True,
        "pareto_front": 1,
        "pareto_require_survival": True,
        "pareto_calibration_source": "no_speed_floor",
        "reward_component_trends": trends,
    }
    block = evidence_block(metrics, score1=0.7, metadata=md)
    w(
        "section5_plateau.txt",
        "SOURCE:\n"
        + src
        + "\nTRENDS_JSON:\n"
        + json.dumps(trends, indent=2)
        + "\nEVIDENCE_BLOCK:\n"
        + block
        + "\n",
    )


def section_throughput_ppo() -> None:
    """Short real PPO train with callback; ON (tuple comps) vs OFF (bare float)."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        w("section4_throughput.txt", f"SKIP: {exc}")
        return

    from domains.highway.adapter import _make_train_vec_env
    from domains.highway.diagnostics_callback import (
        RolloutDiagnosticsCallback,
        combine_train_callbacks,
    )
    from domains.highway.env_wrapper import training_env_config
    from domains.highway.reward_checks import make_highway_validator
    from domains.highway.state import default_smoke_states

    v = make_highway_validator(smoke_states=default_smoke_states())
    code_on = """def compute_reward(state, memory):
    if state.collision:
        t = -20.0
        return t, {"collision_penalty": t, "progress_term": 0.0, "speed_term": 0.0}
    if state.off_road:
        t = -10.0
        return t, {"collision_penalty": 0.0, "progress_term": 0.0, "speed_term": t}
    p = float(state.progress)
    s = float(0.08 * state.speed)
    return p + s, {"collision_penalty": 0.0, "progress_term": p, "speed_term": s}
"""
    code_off = """def compute_reward(state, memory):
    if state.collision:
        return -20.0
    if state.off_road:
        return -10.0
    return float(state.progress + 0.08 * state.speed)
"""
    steps = 3000
    lines = []
    for label, code in (("ON_tuple_components", code_on), ("OFF_bare_float_shim", code_off)):
        rw = v.validate_code(code)
        env_cfg = training_env_config()
        train_env = _make_train_vec_env(rw, seed=7, n_envs=1, env_config=env_cfg)
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            seed=7,
            device="cpu",
            n_steps=64,
            batch_size=64,
            learning_rate=3e-4,
        )
        log_path = str(OUT / f"thru_{label}.jsonl")
        if os.path.isfile(log_path):
            os.remove(log_path)
        cb = RolloutDiagnosticsCallback(
            candidate_id=label, log_path=log_path, continuous_actions=False
        )
        t0 = time.perf_counter()
        model.learn(total_timesteps=steps, progress_bar=False, callback=cb)
        wall = time.perf_counter() - t0
        sps = steps / max(wall, 1e-9)
        n_rollouts = len(cb.component_history.get("progress_term", [])) or len(
            cb.component_history.get("speed_term", [])
        )
        # bare float: empty history expected
        n_hist_keys = len(cb.component_history)
        lines.append(
            f"{label}: steps={steps} wall={wall:.3f}s sps={sps:.1f} "
            f"component_history_keys={n_hist_keys} "
            f"progress_term_checkpoints={len(cb.component_history.get('progress_term', []))} "
            f"summary={cb.component_trend_summary()}\n"
        )
        try:
            train_env.close()
        except Exception:  # noqa: BLE001
            pass
    w("section4_throughput.txt", "".join(lines))


def section_comprehension_rg() -> None:
    import subprocess

    proc = subprocess.run(
        [
            "rg",
            "-n",
            "visit_(List|Set|Dict)Comp|visit_GeneratorExp",
            "raise_core/sandbox/ast_policy.py",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    w(
        "section1_comprehension_rg.txt",
        (proc.stdout or "") + (proc.stderr or "") + f"\nexit={proc.returncode}\n",
    )


if __name__ == "__main__":
    try:
        section_adversarial()
        section_prompts_and_evidence()
        section_shim_and_aggregate_tests()
        section_plateau_with_evidence()
        section_comprehension_rg()
        section_throughput_ppo()
        print("OK evidence collection done")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

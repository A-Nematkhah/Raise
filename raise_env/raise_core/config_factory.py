"""Build runtime dataclasses from the single editable ``raise_config.CONFIG``.

Internal objects (``RaiseRunConfig``, ``ClosedLoopConfig``, ``Stage2Config``, …)
carry no budget/env literal defaults of their own — factories populate them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from typing import Any, Dict, Optional

from raise_config import (
    CONFIG,
    RaiseConfig,
    build_upstream_config,
    config_to_dict,
    validate,
)
from raise_core import console
from raise_core.evo_selection.config import SelectionConfig
from raise_core.feedback import FeedbackCardConfig
from raise_core.pipeline import RaiseRunConfig
from raise_core.raise_loop.config import ClosedLoopConfig
from raise_core.refine import Stage2Config
from raise_core.sandbox.config import SandboxConfig
from raise_core.validate import Stage3Config


def make_feedback(cfg: RaiseConfig) -> FeedbackCardConfig:
    s = cfg.selection
    return FeedbackCardConfig(
        enabled=bool(s.feedback_card),
        max_input_tokens=int(s.max_input_tokens),
        target_input_tokens=int(s.target_input_tokens),
        history_entries=int(s.feedback_history),
        max_trajectories=int(s.feedback_trajectories),
        random_calls=bool(s.feedback_random_calls),
        training_dynamics=bool(s.feedback_training_dynamics),
        component_instrumentation=bool(s.feedback_components),
    )


def make_selection(cfg: RaiseConfig, *, seed: Optional[int] = None) -> SelectionConfig:
    s = cfg.selection
    return SelectionConfig(
        mode=str(s.mode),
        top_k=int(s.top_k),
        n_votes=int(s.n_votes),
        temperature=float(s.temperature),
        p_restart=float(s.p_restart),
        guards=bool(s.guards),
        guard_success_threshold=float(s.guard_success_threshold),
        guard_min_path_length=float(s.guard_min_path_length),
        reflection_note_cap=int(s.reflection_note_cap),
        seed=int(seed if seed is not None else cfg.seed),
    )


def make_sandbox_config(cfg: RaiseConfig) -> SandboxConfig:
    """Base sandbox limits; domain packs still apply attribute allowlists."""
    s = cfg.safety
    return SandboxConfig(
        timeout_seconds=float(s.timeout_seconds),
        startup_timeout_seconds=float(s.startup_timeout_seconds),
        max_code_length=int(s.max_code_length),
        max_call_seconds=float(s.max_call_seconds),
        max_mean_call_seconds=float(s.max_mean_call_seconds),
        mean_call_window=int(s.mean_call_window),
        smoke_max_call_seconds=float(s.smoke_max_call_seconds),
        max_abs_reward=float(s.max_abs_reward),
        max_range_length=int(s.max_range_length),
        max_memory_keys=int(s.max_memory_keys),
        max_memory_value_len=int(s.max_memory_value_len),
        smoke_rlimit_as_bytes=int(s.smoke_rlimit_as_bytes),
        max_pow_exponent=int(s.max_pow_exponent),
        max_pow_bits=int(s.max_pow_bits),
        max_shift_bits=int(s.max_shift_bits),
        max_sequence_repeat=int(s.max_sequence_repeat),
        max_factorial_arg=int(s.max_factorial_arg),
    )


def make_run_config(
    cfg: Optional[RaiseConfig] = None,
    *,
    output_dir: Optional[str] = None,
    resume: bool = True,
    allow_config_drift: bool = False,
    use_stub: bool = False,
    fast: bool = False,
    score1_mode: Optional[str] = None,
) -> RaiseRunConfig:
    """Populate ``RaiseRunConfig`` from ``RaiseConfig`` (no literal budget defaults)."""
    cfg = validate(cfg or CONFIG)
    bud = cfg.budget
    env = cfg.env
    ppo = cfg.ppo
    llm = cfg.llm
    sel = cfg.selection
    upstream = build_upstream_config(env, config=cfg)

    run = RaiseRunConfig(
        output_dir=str(output_dir or "results/raise_run"),
        seed=int(cfg.seed),
        domain="crowdnav",
        llm_provider=str(llm.provider),
        llm_model=llm.model,
        llm_max_tokens=llm.max_tokens,
        llm_reasoning_effort=llm.reasoning_effort,
        llm_seed=llm.seed,
        score1_mode=str(score1_mode or "dataset"),
        stage1_dataset_path=str(bud.stage1.dataset_path),
        population=int(bud.population_size),
        generations=int(bud.generations),
        stage2_train_steps=int(bud.stage2_train_steps),
        stage2_eval_episodes=int(bud.stage2_eval_episodes),
        stage2_eval_split=str(bud.stage2_eval_split),
        stage2_paired_seeds=bool(bud.stage2_paired_seeds),
        stage2_seeds=int(bud.stage2_seeds),
        stage3_train_steps=int(bud.stage3.train_steps),
        stage3_eval_episodes=int(bud.stage3.eval_episodes),
        stage3_warm_start=bool(bud.stage3.warm_start),
        stage3_skip_collapsed=bool(bud.stage3.skip_collapsed),
        collapse_cr=float(bud.collapse_cr),
        stage3_seeds=int(bud.stage3.n_seeds),
        stage3_top_k=bud.stage3.top_k,
        final_eval_episodes=int(bud.final_eval_episodes),
        eval_tied_finalists=bool(bud.eval_tied_finalists),
        use_stub=bool(use_stub),
        elitism=bool(bud.elitism),
        n_crossover=bud.n_crossover,
        n_mutation=bud.n_mutation,
        n_random=bud.n_random,
        mutation_parents=str(sel.mutation_parents),
        mutation_exclude_collapsed=bool(bud.mutation_exclude_collapsed),
        stage2_successive_halving=bool(bud.stage2_successive_halving),
        resume=bool(resume),
        allow_config_drift=bool(allow_config_drift),
        infra_error_policy=str(sel.infra_error_policy),
        max_fallback_fraction=float(sel.max_fallback_fraction),
        strict_fallback=bool(sel.strict_fallback),
        ppo_normalize_reward=bool(ppo.normalize_reward),
        n_envs=int(bud.n_envs),
        label_workers=int(bud.label_workers),
        warm_start=bool(bud.warm_start),
        eval_mode=str(bud.eval_mode),
        action_mode=str(env.action_mode),
        robot_kinematics=str(env.kinematics),
        action_speed_samples=int(env.speed_samples),
        action_rotation_samples=int(env.rotation_samples),
        env_config_path=None,
        env_config_obj=upstream,
        feedback=make_feedback(cfg),
        selection=make_selection(cfg),
        device=str(ppo.device),
        torch_threads=int(ppo.torch_threads),
        fast=bool(fast),
    )
    if fast or use_stub:
        run.apply_fast_profile()
    return run


def make_closed_loop_config(run: RaiseRunConfig) -> ClosedLoopConfig:
    """Map ``RaiseRunConfig`` → ``ClosedLoopConfig`` (pipeline helper)."""
    return ClosedLoopConfig(
        population_size=int(run.population),
        generations=int(run.generations),
        stage2_train_steps=int(run.stage2_train_steps),
        stage1_dataset_path=str(run.stage1_dataset_path),
        seed=int(run.seed),
        device=str(run.device),
        use_stub=bool(run.use_stub),
        llm_provider=str(run.llm_provider),
        output_dir=str(run.output_dir),
        keep_runtime_elite=bool(run.elitism),
        n_crossover=run.n_crossover,
        n_mutation=run.n_mutation,
        n_random=run.n_random,
        mutation_parents=str(run.mutation_parents),
        mutation_exclude_collapsed=bool(run.mutation_exclude_collapsed),
        collapse_cr=float(run.collapse_cr),
        resume=bool(run.resume),
        domain=str(run.domain),
        eval_episodes=int(run.stage2_eval_episodes),
        n_envs=int(run.n_envs),
        label_workers=int(run.label_workers),
        warm_start=bool(run.warm_start),
        stage2_successive_halving=bool(run.stage2_successive_halving),
        stage2_paired_seeds=bool(run.stage2_paired_seeds),
        stage2_seeds=int(run.stage2_seeds),
        eval_mode=str(run.eval_mode),
        stage2_eval_split=str(run.stage2_eval_split),
        action_mode=str(run.action_mode),
        robot_kinematics=str(run.robot_kinematics),
        action_speed_samples=int(run.action_speed_samples),
        action_rotation_samples=int(run.action_rotation_samples),
        env_config_path=run.env_config_path,
        env_config_obj=run.env_config_obj,
        infra_error_policy=str(run.infra_error_policy),
        allow_config_drift=bool(run.allow_config_drift),
        max_fallback_fraction=float(run.max_fallback_fraction),
        strict_fallback=bool(run.strict_fallback),
        ppo_normalize_reward=bool(run.ppo_normalize_reward),
        feedback=run.feedback,
        selection=run.selection,
    )


def write_config_snapshot(output_dir: str, cfg: RaiseConfig) -> Dict[str, Any]:
    """Write ``config_snapshot.json`` (+ sha256) under the run directory."""
    os.makedirs(output_dir, exist_ok=True)
    payload = config_to_dict(cfg)
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = os.path.join(output_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"sha256": digest, "config": payload}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return {"path": path, "sha256": digest, "config": payload}


def load_config_snapshot(output_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(output_dir, "config_snapshot.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def snapshot_digest(cfg: RaiseConfig) -> str:
    text = json.dumps(config_to_dict(validate(cfg)), sort_keys=True, indent=2) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_config_snapshot(
    output_dir: str,
    cfg: RaiseConfig,
    *,
    resume: bool,
    allow_config_drift: bool,
) -> None:
    """Write or verify ``config_snapshot.json`` for crash-safe resume."""
    os.makedirs(output_dir, exist_ok=True)
    existing = load_config_snapshot(output_dir)
    current_digest = snapshot_digest(cfg)
    if existing is not None:
        saved_digest = str(existing.get("sha256") or "")
        if saved_digest and saved_digest != current_digest and not allow_config_drift:
            raise RuntimeError(
                "config drift: effective raise_config differs from "
                f"{output_dir}/config_snapshot.json (use --allow-config-drift to override)"
            )
        return
    if resume and os.path.isdir(output_dir) and _run_has_artifacts(output_dir):
        console.warn(
            "resume: no config_snapshot.json in run dir; skipping raise_config drift check "
            "(legacy run)"
        )
    write_config_snapshot(output_dir, cfg)


def _run_has_artifacts(output_dir: str) -> bool:
    for name in ("config.json", "closed_loop", "manifest.json"):
        if os.path.exists(os.path.join(output_dir, name)):
            return True
    return False


__all__ = [
    "ensure_config_snapshot",
    "ensure_config_snapshot",
    "load_config_snapshot",
    "snapshot_digest",
    "make_closed_loop_config",
    "make_feedback",
    "make_run_config",
    "make_sandbox_config",
    "make_selection",
    "write_config_snapshot",
]

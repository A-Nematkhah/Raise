"""Dataclass ``default_factory`` targets backed by ``raise_config.CONFIG``."""

from __future__ import annotations

from typing import Any

from raise_config import CONFIG, build_upstream_config


def upstream_env_config() -> Any:
    """Isolated upstream CrowdNav++ ``Config`` for the active ``CONFIG.env``."""
    return build_upstream_config(CONFIG.env, config=CONFIG)


def run_seed() -> int:
    return int(CONFIG.seed)


def budget_population() -> int:
    return int(CONFIG.budget.population_size)


def budget_generations() -> int:
    return int(CONFIG.budget.generations)


def budget_stage2_train_steps() -> int:
    return int(CONFIG.budget.stage2_train_steps)


def budget_stage2_eval_episodes() -> int:
    return int(CONFIG.budget.stage2_eval_episodes)


def budget_stage2_seeds() -> int:
    return int(CONFIG.budget.stage2_seeds)


def budget_stage2_paired_seeds() -> bool:
    return bool(CONFIG.budget.stage2_paired_seeds)


def budget_stage3_train_steps() -> int:
    return int(CONFIG.budget.stage3.train_steps)


def budget_stage3_eval_episodes() -> int:
    return int(CONFIG.budget.stage3.eval_episodes)


def budget_final_eval_episodes() -> int:
    return int(CONFIG.budget.final_eval_episodes)


def budget_stage1_dataset_path() -> str:
    return str(CONFIG.budget.stage1.dataset_path)


def budget_stage1_max_invalid() -> int:
    return int(CONFIG.budget.stage1.max_invalid_replacements)


def budget_stage1_behavioral_dedup() -> bool:
    return bool(CONFIG.budget.stage1.behavioral_dedup)


def budget_allow_config_drift() -> bool:
    return False  # resume aborts on drift unless explicitly enabled on the run


def selection_max_fallback_fraction() -> float:
    return float(CONFIG.selection.max_fallback_fraction)


def selection_strict_fallback() -> bool:
    return bool(CONFIG.selection.strict_fallback)


def selection_infra_error_policy() -> str:
    return str(CONFIG.selection.infra_error_policy)


def selection_mutation_parents() -> str:
    return str(CONFIG.selection.mutation_parents)


def selection_mode() -> str:
    return str(CONFIG.selection.mode)


def selection_top_k() -> int:
    return int(CONFIG.selection.top_k)


def selection_n_votes() -> int:
    return int(CONFIG.selection.n_votes)


def selection_temperature() -> float:
    return float(CONFIG.selection.temperature)


def selection_p_restart() -> float:
    return float(CONFIG.selection.p_restart)


def selection_guards() -> bool:
    return bool(CONFIG.selection.guards)


def selection_guard_success_threshold() -> float:
    return float(CONFIG.selection.guard_success_threshold)


def selection_guard_min_path_length() -> float:
    return float(CONFIG.selection.guard_min_path_length)


def selection_reflection_note_cap() -> int:
    return int(CONFIG.selection.reflection_note_cap)


def ppo_n_steps() -> int:
    return int(CONFIG.ppo.n_steps)


def ppo_n_epochs() -> int:
    return int(CONFIG.ppo.n_epochs)


def ppo_normalize_reward() -> bool:
    return bool(CONFIG.ppo.normalize_reward)


def ppo_device() -> str:
    return str(CONFIG.ppo.device)


def ppo_torch_threads() -> int:
    return int(CONFIG.ppo.torch_threads)


def env_kinematics() -> str:
    return str(CONFIG.env.kinematics)


def env_action_mode() -> str:
    return str(CONFIG.env.action_mode)


def env_speed_samples() -> int:
    return int(CONFIG.env.speed_samples)


def env_rotation_samples() -> int:
    return int(CONFIG.env.rotation_samples)


def llm_provider() -> str:
    return str(CONFIG.llm.provider)


def feedback_enabled() -> bool:
    return bool(CONFIG.selection.feedback_card)


def feedback_max_input_tokens() -> int:
    return int(CONFIG.selection.max_input_tokens)


def feedback_target_input_tokens() -> int:
    return int(CONFIG.selection.target_input_tokens)


def feedback_history_entries() -> int:
    return int(CONFIG.selection.feedback_history)


def feedback_max_trajectories() -> int:
    return int(CONFIG.selection.feedback_trajectories)


def feedback_random_calls() -> bool:
    return bool(CONFIG.selection.feedback_random_calls)


def feedback_training_dynamics() -> bool:
    return bool(CONFIG.selection.feedback_training_dynamics)


def feedback_component_instrumentation() -> bool:
    return bool(CONFIG.selection.feedback_components)


def safety_timeout_seconds() -> float:
    return float(CONFIG.safety.timeout_seconds)


def safety_startup_timeout_seconds() -> float:
    return float(CONFIG.safety.startup_timeout_seconds)


def safety_max_code_length() -> int:
    return int(CONFIG.safety.max_code_length)


def safety_max_call_seconds() -> float:
    return float(CONFIG.safety.max_call_seconds)


def safety_max_mean_call_seconds() -> float:
    return float(CONFIG.safety.max_mean_call_seconds)


def safety_mean_call_window() -> int:
    return int(CONFIG.safety.mean_call_window)


def safety_smoke_max_call_seconds() -> float:
    return float(CONFIG.safety.smoke_max_call_seconds)


def safety_max_abs_reward() -> float:
    return float(CONFIG.safety.max_abs_reward)


def safety_max_range_length() -> int:
    return int(CONFIG.safety.max_range_length)


def safety_max_memory_keys() -> int:
    return int(CONFIG.safety.max_memory_keys)


def safety_max_memory_value_len() -> int:
    return int(CONFIG.safety.max_memory_value_len)


def safety_smoke_rlimit_as_bytes() -> int:
    return int(CONFIG.safety.smoke_rlimit_as_bytes)


def safety_max_pow_exponent() -> int:
    return int(CONFIG.safety.max_pow_exponent)


def safety_max_pow_bits() -> int:
    return int(CONFIG.safety.max_pow_bits)


def safety_max_shift_bits() -> int:
    return int(CONFIG.safety.max_shift_bits)


def safety_max_sequence_repeat() -> int:
    return int(CONFIG.safety.max_sequence_repeat)


def safety_max_factorial_arg() -> int:
    return int(CONFIG.safety.max_factorial_arg)

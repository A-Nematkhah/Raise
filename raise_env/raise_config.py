"""RAISE config schema + loader.

**Edit experiment values in** ``configs/raise.yaml`` (not this file).
This module defines dataclasses, validation, and ``CONFIG`` loaded from that YAML.
Override at runtime with ``--config other.yaml`` or ``--set section.key=value``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

_RAISE_ENV_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _RAISE_ENV_ROOT / "configs" / "raise.yaml"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    """CrowdNav++ simulator parameters (formerly ``configs/crowdnav_hard.py``)."""

    time_limit: float = 30.0  # seconds; episode horizon
    time_step: float = 0.25  # seconds per sim step
    val_size: int = 100  # held-out val scenarios
    test_size: int = 500  # blind test scenarios
    success_reward: float = 1.0  # upstream native success bonus
    collision_penalty: float = -0.25  # upstream native collision penalty
    discomfort_dist: float = 0.2  # metres; comfort radius
    discomfort_penalty_factor: float = 0.5  # upstream discomfort scale
    circle_radius: float = 6.0  # metres; spawn circle
    human_num: int = 10  # pedestrians in the circle
    human_num_range: int = 0  # random ± range (must be < human_num)
    predict_method: str = "const_vel"  # none | const_vel
    predict_steps: int = 5  # const-vel prediction horizon (steps)
    pred_timestep: float = 0.25  # seconds; prediction interval base
    robot_visible: bool = True  # humans react to the robot
    # Train / Stage II / Stage III robot kinematics (effective default today).
    kinematics: str = "unicycle"  # holonomic | unicycle
    action_mode: str = "continuous"  # discrete | continuous
    speed_samples: int = 5  # discrete action table speeds
    rotation_samples: int = 16  # discrete action table headings


@dataclass(frozen=True)
class Stage1Budget:
    """Stage I Score1 dataset collection / scoring."""

    dataset_path: str = "domains/crowdnav/data/stage1_dataset"
    # Collect kinematics historically defaulted to holonomic (independent of env.kinematics).
    collect_kinematics: str = "holonomic"
    collect_episodes: int = 40  # episodes per behaviour in the collector CLI default
    max_invalid_replacements: int = 16  # extra LLM draws for invalid proposals
    behavioral_dedup: bool = True


@dataclass(frozen=True)
class Stage3Budget:
    """Stage III long PPO + blind eval."""

    top_k: Optional[int] = None  # None = all Stage II survivors after collapse filter
    train_steps: int = 70_000  # K3 PPO env steps per finalist
    eval_episodes: int = 30
    n_seeds: int = 1  # independent PPO train seeds per finalist
    warm_start: bool = False
    skip_collapsed: bool = True


@dataclass(frozen=True)
class BudgetConfig:
    """Closed-loop population, Stage II/III budgets, workers."""

    population_size: int = 6
    generations: int = 4
    stage2_train_steps: int = 12_000  # K2 PPO env steps per candidate
    stage2_eval_episodes: int = 30
    stage2_seeds: int = 1
    stage2_paired_seeds: bool = True
    stage2_successive_halving: bool = False
    n_envs: int = 1  # DummyVecEnv count
    label_workers: int = 1  # parallel Stage II label threads
    final_eval_episodes: int = 100  # blind test re-eval of the winner (0 = skip)
    eval_tied_finalists: bool = True
    stall_seconds: float = 0.0  # reserved; 0 disables
    elitism: bool = True
    n_crossover: Optional[int] = None  # None → operator_counts default split
    n_mutation: Optional[int] = None
    n_random: Optional[int] = None
    mutation_exclude_collapsed: bool = False
    collapse_cr: float = 0.8
    stage2_eval_split: str = "val"
    eval_mode: str = "holdout_only"  # both | holdout_only
    warm_start: bool = False  # Stage II parent warm-start
    stage1: Stage1Budget = field(default_factory=Stage1Budget)
    stage3: Stage3Budget = field(default_factory=Stage3Budget)


@dataclass(frozen=True)
class PpoConfig:
    """Stable-Baselines3 PPO knobs."""

    n_steps: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    normalize_reward: bool = False  # VecNormalize on training VecEnv only
    device: str = "cuda"  # cpu | cuda
    torch_threads: int = 1  # 0 = library default


@dataclass(frozen=True)
class LlmConfig:
    """LLM proposal / ranking provider."""

    provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    model: Optional[str] = None
    temperature: float = 0.1  # used by LLM ranker
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None  # Groq gpt-oss: none|low|medium|high
    seed: Optional[int] = None  # None → run seed


@dataclass(frozen=True)
class SelectionSection:
    """Evolution parent selection + fallback / infra policy."""

    mode: str = "legacy"  # legacy | llm_rank
    top_k: int = 2
    n_votes: int = 3
    temperature: float = 0.1
    p_restart: float = 0.1
    guards: bool = False
    guard_success_threshold: float = 0.9
    guard_min_path_length: float = 0.1
    reflection_note_cap: int = 3
    mutation_parents: str = "lower"  # lower | rank_weighted | tournament
    keep_runtime_elite: bool = True
    behavioral_dedup: bool = True
    max_fallback_fraction: float = 0.5
    strict_fallback: bool = False
    infra_error_policy: str = "retry_then_abort"  # retry_then_abort|abort|penalize
    # Feedback card (diagnostic prompts).
    feedback_card: bool = True
    max_input_tokens: int = 6000
    target_input_tokens: int = 5500
    feedback_history: int = 3
    feedback_trajectories: int = 3
    feedback_random_calls: bool = True
    feedback_training_dynamics: bool = True
    feedback_components: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    """Reward sandbox limits."""

    timeout_seconds: float = 5.0
    startup_timeout_seconds: float = 120.0
    max_code_length: int = 20_000
    max_call_seconds: float = 1.0
    max_mean_call_seconds: float = 0.02
    mean_call_window: int = 100
    smoke_max_call_seconds: float = 0.1
    max_abs_reward: float = 1e4
    max_range_length: int = 10_000
    max_memory_keys: int = 64
    max_memory_value_len: int = 10_000
    smoke_rlimit_as_bytes: int = 2 * 1024 ** 3  # POSIX smoke child VM budget
    max_pow_exponent: int = 64
    max_pow_bits: int = 64
    max_shift_bits: int = 64
    max_sequence_repeat: int = 10_000
    max_factorial_arg: int = 1000


@dataclass(frozen=True)
class RaiseConfig:
    """Top-level editable config."""

    seed: int = 425
    env: EnvConfig = field(default_factory=EnvConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    selection: SelectionSection = field(default_factory=SelectionSection)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    # Named env overrides (replaces configs/crowdnav_generalization_17.py).
    evaluation_variants: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "generalization_17": {"env": {"human_num": 17, "human_num_range": 0}},
        }
    )


# ---------------------------------------------------------------------------
# Validation / upstream builder / overlays
# ---------------------------------------------------------------------------


def validate(config: RaiseConfig) -> RaiseConfig:
    """Fail fast with clear messages; return ``config`` unchanged on success."""
    env = config.env
    bud = config.budget
    ppo = config.ppo
    sel = config.selection

    if env.kinematics not in ("holonomic", "unicycle"):
        raise ValueError(f"env.kinematics must be holonomic|unicycle, got {env.kinematics!r}")
    if env.action_mode not in ("discrete", "continuous"):
        raise ValueError(f"env.action_mode must be discrete|continuous, got {env.action_mode!r}")
    if env.human_num < 1 or env.human_num_range < 0 or env.human_num_range >= env.human_num:
        raise ValueError("env requires human_num >= 1 and 0 <= human_num_range < human_num")
    if env.predict_method not in ("none", "const_vel"):
        raise ValueError(f"env.predict_method must be none|const_vel, got {env.predict_method!r}")
    if env.time_limit <= 0 or env.time_step <= 0:
        raise ValueError("env.time_limit and env.time_step must be > 0")
    if bud.stage1.collect_kinematics not in ("holonomic", "unicycle"):
        raise ValueError(
            "budget.stage1.collect_kinematics must be holonomic|unicycle, "
            f"got {bud.stage1.collect_kinematics!r}"
        )
    if bud.population_size < 2:
        raise ValueError("budget.population_size must be >= 2")
    if bud.generations < 1:
        raise ValueError("budget.generations must be >= 1")
    if bud.stage2_train_steps < 1 or bud.stage3.train_steps < 1:
        raise ValueError("K2/K3 train_steps must be >= 1")
    if bud.stage2_seeds < 1 or bud.stage3.n_seeds < 1:
        raise ValueError("stage2_seeds and stage3.n_seeds must be >= 1")
    if bud.stage2_successive_halving and bud.stage2_seeds > 1:
        raise ValueError("stage2_successive_halving cannot combine with stage2_seeds > 1")
    if bud.n_envs < 1 or bud.label_workers < 1:
        raise ValueError("n_envs and label_workers must be >= 1")
    if ppo.n_steps < 1 or ppo.n_epochs < 1:
        raise ValueError("ppo.n_steps and ppo.n_epochs must be >= 1")
    if ppo.device not in ("cpu", "cuda"):
        raise ValueError(f"ppo.device must be cpu|cuda, got {ppo.device!r}")
    if not (0.0 < ppo.gamma <= 1.0):
        raise ValueError("ppo.gamma must be in (0, 1]")
    if sel.infra_error_policy not in ("retry_then_abort", "abort", "penalize"):
        raise ValueError(
            "selection.infra_error_policy must be retry_then_abort|abort|penalize, "
            f"got {sel.infra_error_policy!r}"
        )
    if sel.mode not in ("legacy", "llm_rank"):
        raise ValueError("selection.mode must be legacy|llm_rank")
    if sel.mutation_parents not in ("lower", "rank_weighted", "tournament"):
        raise ValueError(
            "selection.mutation_parents must be lower|rank_weighted|tournament"
        )
    if not (0.0 <= sel.max_fallback_fraction <= 1.0):
        raise ValueError("selection.max_fallback_fraction must be in [0, 1]")
    # Operator counts must sum when all three are set.
    ops = (bud.n_crossover, bud.n_mutation, bud.n_random)
    if all(x is not None for x in ops):
        elite = 1 if bud.elitism else 0
        total = elite + int(bud.n_crossover) + int(bud.n_mutation) + int(bud.n_random)
        if total != bud.population_size:
            raise ValueError(
                f"elitism({elite})+crossover+mutation+random={total} "
                f"must equal population_size={bud.population_size}"
            )
    # Soft warning for undersized K2 relative to rollout length.
    min_k2 = bud.n_envs * ppo.n_steps
    if bud.stage2_train_steps < min_k2:
        import warnings

        warnings.warn(
            f"budget.stage2_train_steps={bud.stage2_train_steps} < "
            f"n_envs*ppo.n_steps={min_k2}; PPO may not complete a rollout",
            UserWarning,
            stacklevel=2,
        )
    return config


def build_upstream_config(env: EnvConfig, *, variant: Optional[str] = None, config: Optional[RaiseConfig] = None) -> Any:
    """Build an isolated upstream ``crowd_nav.configs.config.Config`` from ``env``.

    Replaces ``configs/crowdnav_hard.py`` / ``env_config_path`` / ``human_num_override``.
    """
    import sys
    from copy import deepcopy as _deepcopy

    # Upstream Config calls ``get_args()`` at import time; isolate from pytest argv.
    old_argv = sys.argv
    sys.argv = [old_argv[0] if old_argv else "raise_config"]
    try:
        from crowd_nav.configs.config import Config as _UpstreamConfig
    finally:
        sys.argv = old_argv

    cfg = _UpstreamConfig()
    for section in (
        "training",
        "env",
        "reward",
        "sim",
        "humans",
        "robot",
        "action_space",
        "orca",
        "sf",
        "data",
    ):
        setattr(cfg, section, _deepcopy(getattr(_UpstreamConfig, section)))

    active = env
    if variant:
        root = config or CONFIG
        overrides = (root.evaluation_variants or {}).get(variant)
        if overrides is None:
            raise KeyError(
                f"unknown evaluation variant {variant!r}; "
                f"known={sorted((root.evaluation_variants or {}).keys())}"
            )
        env_over = dict(overrides.get("env") or {})
        active = replace(env, **{k: env_over[k] for k in env_over if hasattr(env, k)})

    cfg.env.time_limit = float(active.time_limit)
    cfg.env.time_step = float(active.time_step)
    cfg.env.val_size = int(active.val_size)
    cfg.env.test_size = int(active.test_size)
    cfg.reward.success_reward = float(active.success_reward)
    cfg.reward.collision_penalty = float(active.collision_penalty)
    cfg.reward.discomfort_dist = float(active.discomfort_dist)
    cfg.reward.discomfort_penalty_factor = float(active.discomfort_penalty_factor)
    cfg.sim.circle_radius = float(active.circle_radius)
    cfg.sim.human_num = int(active.human_num)
    cfg.sim.human_num_range = int(active.human_num_range)
    cfg.sim.predict_method = str(active.predict_method)
    cfg.sim.predict_steps = int(active.predict_steps)
    cfg.data.pred_timestep = float(active.pred_timestep)
    cfg.action_space.kinematics = str(active.kinematics)
    cfg.robot.visible = bool(active.robot_visible)
    return cfg


def config_to_dict(config: RaiseConfig) -> Dict[str, Any]:
    """JSON-serialisable nested dict of the effective config."""
    return asdict(config)


def apply_set_overrides(config: RaiseConfig, assignments: Mapping[str, str]) -> RaiseConfig:
    """Apply ``section.key=value`` (or nested ``budget.stage3.train_steps=…``) strings."""
    data = asdict(config)
    for raw_key, raw_val in assignments.items():
        key = str(raw_key).strip()
        if not key or "=" in key and raw_val == "":
            # allow "section.key=value" packed in the key
            if "=" in key and raw_val == "":
                key, raw_val = key.split("=", 1)
            else:
                raise ValueError(f"invalid --set {raw_key!r}")
        parts = key.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"--set expects section.key[=value], got {key!r}; "
                "see configs/raise.yaml header and the README migration table"
            )
        cursor: Any = data
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"unknown config path {key!r} (at {part!r})")
            cursor = cursor[part]
        leaf = parts[-1]
        if leaf not in cursor:
            raise ValueError(f"unknown config field {key!r}")
        cursor[leaf] = _coerce_value(cursor[leaf], raw_val, path=key)
    return _dict_to_raise_config(data)


def _coerce_value(current: Any, raw: str, *, path: str) -> Any:
    text = str(raw).strip()
    if current is None:
        if text.lower() in ("none", "null", ""):
            return None
        # infer
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            return text
    if isinstance(current, bool):
        if text.lower() in ("true", "1", "yes", "on"):
            return True
        if text.lower() in ("false", "0", "no", "off"):
            return False
        raise TypeError(f"{path}: expected bool, got {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(text, 10)
        except ValueError as exc:
            raise TypeError(f"{path}: expected int, got {raw!r}") from exc
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError as exc:
            raise TypeError(f"{path}: expected float, got {raw!r}") from exc
    if isinstance(current, str):
        return text
    if isinstance(current, dict):
        raise TypeError(f"{path}: cannot --set a mapping; set nested keys")
    return text


def _dict_to_raise_config(data: Dict[str, Any]) -> RaiseConfig:
    env = EnvConfig(**data["env"])
    s1 = Stage1Budget(**data["budget"]["stage1"])
    s3 = Stage3Budget(**data["budget"]["stage3"])
    bud_raw = {k: v for k, v in data["budget"].items() if k not in ("stage1", "stage3")}
    budget = BudgetConfig(**bud_raw, stage1=s1, stage3=s3)
    return RaiseConfig(
        seed=int(data["seed"]),
        env=env,
        budget=budget,
        ppo=PpoConfig(**data["ppo"]),
        llm=LlmConfig(**data["llm"]),
        selection=SelectionSection(**data["selection"]),
        safety=SafetyConfig(**data["safety"]),
        evaluation_variants=dict(data.get("evaluation_variants") or {}),
    )


def load_config_yaml(path: str | os.PathLike[str]) -> RaiseConfig:
    """Load a YAML file with the same sections as ``configs/raise.yaml``."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load configs/*.yaml; "
            "pip install pyyaml"
        ) from exc
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(data)!r}")
    return validate(_dict_to_raise_config(data))


def load_config_module(path: str) -> RaiseConfig:
    """Load a Python file that exports ``CONFIG`` (a ``RaiseConfig``)."""
    import importlib.util

    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("_raise_user_config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot load config module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "CONFIG", None)
    if not isinstance(cfg, RaiseConfig):
        raise TypeError(f"{path} must export CONFIG: RaiseConfig, got {type(cfg)!r}")
    return validate(cfg)


def load_config_file(path: str | os.PathLike[str]) -> RaiseConfig:
    """Load ``.yaml`` / ``.yml`` or a ``.py`` module exporting ``CONFIG``."""
    path_str = os.fspath(path)
    lower = path_str.lower()
    if lower.endswith((".yaml", ".yml")):
        return load_config_yaml(path_str)
    if lower.endswith(".py"):
        return load_config_module(path_str)
    raise ValueError(
        f"unsupported config type {path_str!r}; use .yaml / .yml or .py"
    )


def resolve_config(
    *,
    config_path: Optional[str] = None,
    set_overrides: Optional[Tuple[str, ...]] = None,
) -> RaiseConfig:
    """Precedence: ``configs/raise.yaml`` < ``--config`` file < ``--set``."""
    cfg = CONFIG
    if config_path:
        cfg = load_config_file(config_path)
    if set_overrides:
        parsed: Dict[str, str] = {}
        for item in set_overrides:
            if "=" not in item:
                raise ValueError(
                    f"--set requires section.key=value, got {item!r}. "
                    "Removed CLI flags map to --set keys; see configs/raise.yaml "
                    "header and README Configuration."
                )
            k, v = item.split("=", 1)
            parsed[k.strip()] = v.strip()
        cfg = apply_set_overrides(cfg, parsed)
    return validate(cfg)


# Loaded from configs/raise.yaml at import (the file users edit).
CONFIG = load_config_yaml(DEFAULT_CONFIG_PATH)


__all__ = [
    "BudgetConfig",
    "CONFIG",
    "DEFAULT_CONFIG_PATH",
    "EnvConfig",
    "LlmConfig",
    "PpoConfig",
    "RaiseConfig",
    "SafetyConfig",
    "SelectionSection",
    "Stage1Budget",
    "Stage3Budget",
    "apply_set_overrides",
    "build_upstream_config",
    "config_to_dict",
    "load_config_file",
    "load_config_module",
    "load_config_yaml",
    "resolve_config",
    "validate",
]

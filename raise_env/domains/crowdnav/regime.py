"""
Evolution randomization regime + GST checkpoint consistency.

See AUDIT.md §8 (Decisions). Default first-pass regime is ``without_random``.
Observation-space parity choice (a): Stage II/III / final eval use
``predict_method="inferred"`` with the GST path matched to the regime.
"""

from __future__ import annotations

import logging
from typing import Any, FrozenSet, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

EvolutionRandomizationRegime = Literal["without_random", "with_random", "both"]

REGIME_WITHOUT_RANDOM: EvolutionRandomizationRegime = "without_random"
REGIME_WITH_RANDOM: EvolutionRandomizationRegime = "with_random"
REGIME_BOTH: EvolutionRandomizationRegime = "both"

ALLOWED_REGIMES: FrozenSet[str] = frozenset(
    {REGIME_WITHOUT_RANDOM, REGIME_WITH_RANDOM, REGIME_BOTH}
)

# Default for the first validation pass (AUDIT.md §8.1).
EVOLUTION_RANDOMIZATION_REGIME: EvolutionRandomizationRegime = REGIME_WITHOUT_RANDOM

# Choice (a): Stage II/III + final eval match CrowdNav++ obs space.
EVOLUTION_PREDICT_METHOD: str = "inferred"
EVOLUTION_ENV_NAME_INFERRED: str = "CrowdSimPredRealGST-v0"
EVOLUTION_ENV_NAME_NONE: str = "CrowdSimVarNum-v0"

# Upstream CrowdNav++ GST predictor trees (see config.py comments / README).
GST_MODEL_DIR_WITHOUT_RANDOM = (
    "gst_updated/results/"
    "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-"
    "edge_head_0-ebd_64-snl_1-snh_8-seed_1000/sj"
)
GST_MODEL_DIR_WITH_RANDOM = (
    "gst_updated/results/"
    "100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-"
    "edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj"
)


def parse_regime(value: Optional[str]) -> EvolutionRandomizationRegime:
    if value is None or not str(value).strip():
        return EVOLUTION_RANDOMIZATION_REGIME
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "without_random": REGIME_WITHOUT_RANDOM,
        "no_rand": REGIME_WITHOUT_RANDOM,
        "norand": REGIME_WITHOUT_RANDOM,
        "with_random": REGIME_WITH_RANDOM,
        "rand": REGIME_WITH_RANDOM,
        "both": REGIME_BOTH,
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown EVOLUTION_RANDOMIZATION_REGIME={value!r}; "
            f"expected one of {sorted(ALLOWED_REGIMES)}"
        )
    return aliases[key]  # type: ignore[return-value]


def randomization_flags(regime: str) -> Tuple[bool, bool]:
    """
    Return ``(randomize_attributes, random_goal_changing)``.

    ``both`` is not a simultaneous mixed flag — callers must run two passes.
    """
    r = parse_regime(regime)
    if r == REGIME_BOTH:
        raise ValueError(
            "regime='both' requires two separate full passes (doubled budget), "
            "not a single mixed env. Set without_random or with_random."
        )
    on = r == REGIME_WITH_RANDOM
    return on, on


def gst_model_dir_for_regime(regime: str) -> str:
    r = parse_regime(regime)
    if r == REGIME_BOTH:
        raise ValueError(
            "Cannot pick a single GST model_dir for regime='both'; "
            "run without_random and with_random as separate passes."
        )
    if r == REGIME_WITH_RANDOM:
        return GST_MODEL_DIR_WITH_RANDOM
    return GST_MODEL_DIR_WITHOUT_RANDOM


def gst_path_looks_randomized(model_dir: str) -> bool:
    """True if the GST tree name embeds the upstream ``_rand`` training tag."""
    norm = model_dir.replace("\\", "/").rstrip("/") + "/"
    return "seed_1000_rand/" in norm or norm.rstrip("/").endswith("_rand")


def assert_gst_matches_regime(
    model_dir: str,
    regime: str,
    *,
    predict_method: str,
    entry_point: str = "unknown",
) -> None:
    """
    If ``predict_method == 'inferred'``, require GST ``_rand`` suffix ↔ regime.

    If ``predict_method == 'none'``, log and return (non-predictive env; no GST).
    """
    method = (predict_method or "").strip().lower()
    if method == "none":
        logger.info(
            "[%s] predict_method=none — GST-regime consistency N/A "
            "(non-predictive env for this call; GST checkpoint not loaded).",
            entry_point,
        )
        return
    if method != "inferred":
        raise ValueError(
            f"[{entry_point}] Unsupported predict_method={predict_method!r}; "
            "expected 'inferred' or 'none'."
        )

    r = parse_regime(regime)
    if r == REGIME_BOTH:
        raise ValueError(
            f"[{entry_point}] regime='both' cannot bind one GST checkpoint; "
            "use without_random or with_random."
        )

    expected_dir = gst_model_dir_for_regime(r)
    looks_rand = gst_path_looks_randomized(model_dir)
    expect_rand = r == REGIME_WITH_RANDOM
    norm_loaded = model_dir.replace("\\", "/").rstrip("/")
    norm_expected = expected_dir.replace("\\", "/").rstrip("/")
    suffix_ok = norm_loaded == norm_expected or norm_loaded.endswith(
        "/" + norm_expected
    )

    if looks_rand != expect_rand or not suffix_ok:
        raise AssertionError(
            f"[{entry_point}] GST pred.model_dir does not match "
            f"EVOLUTION_RANDOMIZATION_REGIME={r!r}.\n"
            f"  loaded model_dir   = {model_dir}\n"
            f"  expected model_dir = {expected_dir}\n"
            f"  path looks _rand = {looks_rand}, regime expects _rand = {expect_rand}"
        )


def detach_config_namespaces(cfg: Any) -> Any:
    """
    Shallow-copy class-level CrowdNav Config namespaces onto the instance.

    ``Config.env`` / ``Config.sim`` / … are shared class attributes. Mutating
    them in-place poisons other stages in-process, and on Windows ``spawn`` the
    child re-imports ``config.py`` (defaults like ``human_num=20``) while the
    parent may have sized ShmemVecEnv buffers from a mutated value (e.g. 5 for
    ``--easy``). Detaching before mutation keeps pickle round-trips consistent.
    """
    from copy import copy

    for name in ("env", "sim", "humans", "robot", "pred", "reward", "action_space"):
        if name in getattr(cfg, "__dict__", {}):
            continue
        if not hasattr(cfg, name):
            continue
        setattr(cfg, name, copy(getattr(cfg, name)))
    return cfg


def apply_regime_to_config(
    cfg: Any,
    regime: str,
    *,
    predict_method: str,
    entry_point: str = "unknown",
) -> Any:
    """
    Set randomization flags, predict_method / wrapper, and pred.model_dir on a
    CrowdNav ``Config`` instance (class-attr style). Assert GST when inferred.
    """
    detach_config_namespaces(cfg)
    attrs, goal_changing = randomization_flags(regime)
    cfg.env.randomize_attributes = bool(attrs)
    cfg.humans.random_goal_changing = bool(goal_changing)

    method = (predict_method or "none").strip().lower()
    cfg.sim.predict_method = method
    if method == "inferred":
        cfg.env.use_wrapper = True
        cfg.pred.model_dir = gst_model_dir_for_regime(regime)
        assert_gst_matches_regime(
            cfg.pred.model_dir,
            regime,
            predict_method=method,
            entry_point=entry_point,
        )
    else:
        cfg.env.use_wrapper = False
        assert_gst_matches_regime(
            getattr(getattr(cfg, "pred", None), "model_dir", "") or "",
            regime,
            predict_method=method,
            entry_point=entry_point,
        )
    return cfg


def env_name_for_predict_method(predict_method: str) -> str:
    if (predict_method or "").strip().lower() == "inferred":
        return EVOLUTION_ENV_NAME_INFERRED
    return EVOLUTION_ENV_NAME_NONE

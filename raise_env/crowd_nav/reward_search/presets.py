"""
Named run presets for RAISE.

``fast`` — stub trainers, tiny budgets (pytest / smoke).
``paper`` — Tables 3–6 budgets (K2=8000, G2=16, K3=1e7, …). Only apply via
an explicit human-triggered paper-scale entry point — never as pytest default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from crowd_nav.reward_search.validate import STAGE3_PAPER_STEPS

# Paper Tables 3–6 (authoritative constants; mirrored in configs/paper_scale.yaml).
PAPER_N = 8
PAPER_G1 = 10
PAPER_M = 100
PAPER_N_TRAJ = 10
PAPER_K2 = 8000
PAPER_G2 = 16
PAPER_E2 = 50
PAPER_T_SHORT = 100
PAPER_K3 = STAGE3_PAPER_STEPS  # 1e7
PAPER_G3 = 3
PAPER_E3 = 500
PAPER_HUMAN_COUNTS = (5, 10, 15, 20)

# Paper does not state how many seeds Table 1 mean±std used.
# We default to 5 independent Algorithm-1 seeds and document that choice.
PAPER_DEFAULT_N_SEEDS = 5
PAPER_DEFAULT_SEEDS = (425, 426, 427, 428, 429)

PAPER_SCALE_YAML = os.path.join("configs", "paper_scale.yaml")

METHODOLOGY_SEED_NOTE = (
    "The RAISE paper does not state how many random seeds Table 1's "
    "mean±std bars used. This report uses {n_seeds} independent full "
    "Algorithm-1 runs (seeds={seeds}) and aggregates final-policy metrics "
    "as mean±std across seeds — not across evaluation episodes within a "
    "single seed, which is a smaller variance source than the paper's "
    "reported bars."
)


@dataclass(frozen=True)
class PaperScaleSpec:
    """Immutable paper-scale hyper-parameters (Tables 3–6)."""

    N: int = PAPER_N
    G1: int = PAPER_G1
    M: int = PAPER_M
    N_traj: int = PAPER_N_TRAJ
    K2: int = PAPER_K2
    G2: int = PAPER_G2
    E2: int = PAPER_E2
    T_short: int = PAPER_T_SHORT
    K3: int = PAPER_K3
    G3: int = PAPER_G3
    E3: int = PAPER_E3
    human_counts: tuple = PAPER_HUMAN_COUNTS
    seeds: tuple = PAPER_DEFAULT_SEEDS
    score1_mode: str = "dataset"
    stage1_dataset_path: str = "data/stage1_dataset"
    device: str = "cuda"
    llm_provider: str = "seed"
    output_dir: str = "results/paper_scale"
    cost_log: str = "results/paper_scale/cost_log.json"
    # AUDIT.md §8
    randomization_regime: str = "without_random"
    predict_method: str = "inferred"

    def methodology_note(self) -> str:
        return METHODOLOGY_SEED_NOTE.format(
            n_seeds=len(self.seeds),
            seeds=list(self.seeds),
        )


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Minimal YAML subset reader (no PyYAML dependency).

    Supports ``key: value``, lists like ``[a, b]``, ints/floats/bools/strings,
    and `#` comments. Enough for ``configs/paper_scale.yaml``.
    """
    out: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        out[key] = _parse_yaml_scalar(val)
    return out


def _parse_yaml_scalar(val: str) -> Any:
    if val == "":
        return None
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_yaml_scalar(x.strip()) for x in inner.split(",")]
    low = val.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        if "." in val or "e" in low:
            return float(val)
        return int(val)
    except ValueError:
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            return val[1:-1]
        return val


def load_paper_scale_yaml(path: Optional[str] = None) -> PaperScaleSpec:
    """Load ``configs/paper_scale.yaml`` (falls back to baked-in paper constants)."""
    path = path or PAPER_SCALE_YAML
    data: Dict[str, Any] = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = _parse_simple_yaml(f.read())
    seeds = data.get("seeds")
    if seeds is None and data.get("n_seeds"):
        n = int(data["n_seeds"])
        seeds = list(PAPER_DEFAULT_SEEDS[:n])
        if len(seeds) < n:
            base = int(seeds[-1]) if seeds else 425
            seeds = list(range(base, base + n))
    return PaperScaleSpec(
        N=int(data.get("N", PAPER_N)),
        G1=int(data.get("G1", PAPER_G1)),
        M=int(data.get("M", PAPER_M)),
        N_traj=int(data.get("N_traj", PAPER_N_TRAJ)),
        K2=int(data.get("K2", PAPER_K2)),
        G2=int(data.get("G2", PAPER_G2)),
        E2=int(data.get("E2", PAPER_E2)),
        T_short=int(data.get("T_short", PAPER_T_SHORT)),
        K3=int(data.get("K3", PAPER_K3)),
        G3=int(data.get("G3", PAPER_G3)),
        E3=int(data.get("E3", PAPER_E3)),
        human_counts=tuple(int(x) for x in (data.get("human_counts") or PAPER_HUMAN_COUNTS)),
        seeds=tuple(int(x) for x in (seeds or PAPER_DEFAULT_SEEDS)),
        score1_mode=str(data.get("score1_mode", "dataset")),
        stage1_dataset_path=str(data.get("stage1_dataset_path", "data/stage1_dataset")),
        device=str(data.get("device", "cuda")),
        llm_provider=str(data.get("llm_provider", "seed")),
        output_dir=str(data.get("output_dir", "results/paper_scale")),
        cost_log=str(data.get("cost_log", "results/paper_scale/cost_log.json")),
        randomization_regime=str(
            data.get(
                "EVOLUTION_RANDOMIZATION_REGIME",
                data.get("randomization_regime", "without_random"),
            )
        ),
        predict_method=str(data.get("predict_method", "inferred")),
    )


def apply_paper_scale(config: Any, spec: Optional[PaperScaleSpec] = None) -> Any:
    """
    Mutate an ``RaiseRunConfig`` to paper Tables 3–6 budgets.

    Does **not** enable stubs. Distinct from ``apply_fast_profile``.
    """
    spec = spec or load_paper_scale_yaml()
    config.fast = False
    config.score1_mode = spec.score1_mode
    config.stage1_dataset_path = spec.stage1_dataset_path
    config.stage1_population = spec.N
    config.stage1_generations = spec.G1
    config.stage2_rounds = spec.G2
    config.stage2_train_steps = spec.K2
    config.stage2_k2_unit = "gradient_steps"
    config.stage2_eval_episodes = spec.E2
    config.stage2_horizon = spec.T_short
    config.stage2_use_stub = False
    config.stage3_rounds = spec.G3
    config.stage3_train_steps = spec.K3
    config.stage3_eval_episodes = spec.E3
    config.stage3_use_stub = False
    config.stage3_run_h_sweep = True
    config.elitism = False
    config.final_rank = "llm"
    config.device = spec.device
    config.llm_provider = spec.llm_provider
    config.randomization_regime = spec.randomization_regime
    config.predict_method = spec.predict_method
    # Stash dataset collect sizes for the paper-scale runner / report.
    if not hasattr(config, "metadata") or config.metadata is None:
        try:
            config.metadata = {}
        except Exception:  # noqa: BLE001
            pass
    meta = getattr(config, "metadata", None)
    if isinstance(meta, dict):
        meta["paper_scale"] = {
            "M": spec.M,
            "N_traj": spec.N_traj,
            "K3": spec.K3,
            "methodology": spec.methodology_note(),
            "EVOLUTION_RANDOMIZATION_REGIME": spec.randomization_regime,
            "predict_method": spec.predict_method,
        }
    return config


def parse_seeds_arg(
    seeds: Optional[Sequence[int]] = None,
    *,
    default: Sequence[int] = PAPER_DEFAULT_SEEDS,
) -> List[int]:
    if seeds is None or len(list(seeds)) == 0:
        return list(default)
    return [int(s) for s in seeds]

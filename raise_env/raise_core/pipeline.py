"""
RAISE orchestrator (highway).

Closed-loop evolution (Score1 → Stage II PPO train+eval → fitness rank → LLM
breeding) for G generations, then Stage III long PPO on the final population.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from domains.highway.action_config import configure_action_mode_from_config
from domains.highway.post_run import write_highway_run_artifacts
from raise_core import console
from raise_core.domains import (
    DEFAULT_DOMAIN,
    DomainPack,
    load_domain,
    make_score_fn_for_domain,
    make_stage2_trainer_for_domain,
    make_stage3_trainer_for_domain,
    make_validator_for_domain,
)
from raise_core.candidate import RewardCandidate
from raise_core.explore import OperatorCounts, operator_counts
from raise_core.feedback import FeedbackCardConfig
from raise_core.llm import LLMClient, make_llm_client
from raise_core.population_io import candidate_to_dict
from raise_core.raise_loop import ClosedLoopConfig, ClosedLoopRunner
from raise_core.raise_loop.evolve_rank import parse_evolve_rank
from raise_core.raise_loop.logging_io import write_json
from raise_core.selection import candidate_fitness, pick_best_trained
from raise_core.validate import STAGE3_STEPS, Stage3Config, Stage3Runner


@dataclass
class RaiseRunConfig:
    """End-to-end run settings (highway defaults)."""

    output_dir: str = "results/raise_run"
    seed: int = 425
    domain: str = DEFAULT_DOMAIN
    llm_provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    llm_model: Optional[str] = None
    # Stage I Score1: "dataset" (default) | "smoke" (fast tests only)
    score1_mode: str = "dataset"
    stage1_dataset_path: str = "domains/highway/data/stage1_dataset"

    # Closed loop: population N, generations G, Stage II budget per candidate.
    population: int = 6
    generations: int = 4
    stage2_train_steps: int = 12_000
    stage2_eval_episodes: int = 30

    # Stage III: long PPO on the final population.
    stage3_train_steps: int = STAGE3_STEPS
    stage3_eval_episodes: int = 30

    # Stub trainers (pytest / --fast).
    use_stub: bool = False
    # Carry the generation's best genome into the next generation.
    elitism: bool = True
    # Next-generation children per operator; None -> default split
    # (N=6 with elitism: 1 elite + 2 crossover + 2 mutation + 1 random).
    n_crossover: Optional[int] = None
    n_mutation: Optional[int] = None
    n_random: Optional[int] = None
    # Next-generation parent order: scalar (fitness) | score1 | hybrid.
    evolve_rank: str = "scalar"
    evolve_rank_score1_weight: float = 0.4
    # Crash-safe resume of closed loop + Stage III from output_dir.
    resume: bool = True

    # Highway trainer knobs.
    highway_n_envs: int = 1
    highway_label_workers: int = 1
    # Off: a warm-started child carries its parent's PPO steps, so its fitness
    # would be compared against siblings trained on a smaller budget.
    highway_warm_start: bool = False
    highway_eval_mode: str = "holdout_only"
    highway_action_mode: str = "meta_default"
    highway_action_continuous_lateral: bool = True
    highway_meta_fine_low: float = 15.0
    highway_meta_fine_high: float = 35.0
    highway_meta_fine_n: int = 21
    # Diagnostic feedback card + LLM input-token budget.
    feedback: FeedbackCardConfig = field(default_factory=FeedbackCardConfig)

    device: str = "cuda"
    fast: bool = False

    def apply_fast_profile(self) -> None:
        """Seconds-scale walk with stub trainers + tiny budgets."""
        self.fast = True
        self.score1_mode = "smoke"
        self.population = 2
        self.generations = 2
        self.stage2_train_steps = 8
        self.stage2_eval_episodes = 2
        self.stage3_train_steps = 8
        self.stage3_eval_episodes = 2
        self.use_stub = True
        self.llm_provider = "seed"
        self.n_crossover = self.n_mutation = self.n_random = None

    def operators(self) -> OperatorCounts:
        """Next-generation operator counts; ``ValueError`` if they do not add up."""
        return operator_counts(
            int(self.population),
            elitism=bool(self.elitism),
            n_crossover=self.n_crossover,
            n_mutation=self.n_mutation,
            n_random=self.n_random,
        )


@dataclass
class RaiseArtifacts:
    """Populations / bests produced by one run."""

    output_dir: str
    seed_code: str
    stage2_population: List[RewardCandidate] = field(default_factory=list)
    stage3_population: List[RewardCandidate] = field(default_factory=list)
    best_stage1: Optional[RewardCandidate] = None
    best_stage2: Optional[RewardCandidate] = None
    best_stage3: Optional[RewardCandidate] = None
    manifest: Dict[str, Any] = field(default_factory=dict)


def _stage2_history(epochs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-candidate Stage II rows (``round_index`` = generation) for plots."""
    rows: List[Dict[str, Any]] = []
    for rec in epochs:
        for cid, metrics in (rec.get("label_metrics") or {}).items():
            rows.append(
                {
                    "round_index": int(rec.get("epoch", 0)),
                    "candidate_id": str(cid),
                    "metrics": dict(metrics or {}),
                }
            )
    return rows


class RaisePipeline:
    """Run closed-loop evolution + Stage III and persist JSON artifacts."""

    def __init__(
        self,
        config: Optional[RaiseRunConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        domain_pack: Optional[DomainPack] = None,
    ) -> None:
        self.config = config or RaiseRunConfig()
        self.llm = llm
        self.domain_pack = domain_pack or load_domain(self.config.domain)
        # Smoke states read the active action mode (int vs tuple action).
        configure_action_mode_from_config(self.config)
        self.validator = make_validator_for_domain(self.domain_pack)

    def _build_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        seed = self.domain_pack.seed_reward_source
        if self.config.llm_model is None:
            return make_llm_client(self.config.llm_provider, base_code=seed)
        return make_llm_client(
            self.config.llm_provider,
            model=self.config.llm_model,
            base_code=seed,
        )

    def _closed_loop_config(self) -> ClosedLoopConfig:
        cfg = self.config
        return ClosedLoopConfig(
            population_size=int(cfg.population),
            generations=int(cfg.generations),
            stage2_train_steps=int(cfg.stage2_train_steps),
            stage1_dataset_path=str(cfg.stage1_dataset_path),
            seed=int(cfg.seed),
            device=str(cfg.device),
            use_stub=bool(cfg.use_stub),
            llm_provider=str(cfg.llm_provider),
            output_dir=str(cfg.output_dir),
            keep_runtime_elite=bool(cfg.elitism),
            n_crossover=cfg.n_crossover,
            n_mutation=cfg.n_mutation,
            n_random=cfg.n_random,
            resume=bool(cfg.resume),
            domain=str(self.domain_pack.name),
            eval_episodes=int(cfg.stage2_eval_episodes),
            evolve_rank=parse_evolve_rank(cfg.evolve_rank),
            evolve_rank_score1_weight=float(cfg.evolve_rank_score1_weight),
            highway_n_envs=max(1, int(cfg.highway_n_envs or 1)),
            highway_label_workers=max(1, int(cfg.highway_label_workers or 1)),
            highway_warm_start=bool(cfg.highway_warm_start),
            highway_eval_mode=str(cfg.highway_eval_mode or "both"),
            highway_action_mode=str(cfg.highway_action_mode or "meta_default"),
            highway_action_continuous_lateral=bool(cfg.highway_action_continuous_lateral),
            highway_meta_fine_low=float(cfg.highway_meta_fine_low),
            highway_meta_fine_high=float(cfg.highway_meta_fine_high),
            highway_meta_fine_n=int(cfg.highway_meta_fine_n),
            feedback=cfg.feedback,
        )

    def _stage3_config(self) -> Stage3Config:
        cfg = self.config
        return Stage3Config(
            train_env_steps=int(cfg.stage3_train_steps),
            eval_episodes=int(cfg.stage3_eval_episodes),
            seed=int(cfg.seed),
            output_root=os.path.join(cfg.output_dir, "stage3_train"),
            device=str(cfg.device),
            resume=bool(cfg.resume),
            highway_action_mode=str(cfg.highway_action_mode or "meta_default"),
            highway_action_continuous_lateral=bool(cfg.highway_action_continuous_lateral),
            highway_meta_fine_low=float(cfg.highway_meta_fine_low),
            highway_meta_fine_high=float(cfg.highway_meta_fine_high),
            highway_meta_fine_n=int(cfg.highway_meta_fine_n),
            highway_eval_mode=str(cfg.highway_eval_mode or "both"),
            highway_warm_start=bool(cfg.highway_warm_start),
            highway_n_envs=max(1, int(cfg.highway_n_envs or 1)),
        )

    def run(self) -> RaiseArtifacts:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        wall_t0 = time.perf_counter()
        pack = self.domain_pack
        if pack.name != str(cfg.domain).strip().lower():
            raise ValueError(
                f"Domain pack mismatch: config.domain={cfg.domain!r}, "
                f"pack.name={pack.name!r}"
            )
        console.banner("RAISE")
        console.status(
            f"output={cfg.output_dir} seed={cfg.seed} llm={cfg.llm_provider} "
            f"fast={cfg.fast} device={cfg.device} domain={pack.name}"
        )
        llm = self._build_llm()

        seed_code = pack.seed_reward_source.strip() + "\n"
        manifest: Dict[str, Any] = {
            "algorithm": "RAISE",
            "domain": pack.name,
            "domain_display_name": pack.display_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
        }
        write_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
        with open(os.path.join(cfg.output_dir, "seed_reward.py"), "w", encoding="utf-8") as f:
            f.write(seed_code)

        # ----- Closed loop (Stage I proposal + Stage II evaluation per generation) -----
        console.banner("Closed-loop reward evolution")
        cl_cfg = self._closed_loop_config()
        if cfg.fast:
            cl_cfg.apply_fast_profile()
        cl_result = ClosedLoopRunner(
            cl_cfg,
            llm=llm,
            score_fn=make_score_fn_for_domain(
                pack, mode=cfg.score1_mode, dataset_path=cfg.stage1_dataset_path
            ),
            trainer=make_stage2_trainer_for_domain(pack, use_stub=bool(cfg.use_stub)),
            validator=self.validator,
        ).run()

        stage2_pop = list(cl_result.population)
        best_s1 = max(
            stage2_pop,
            key=lambda c: float("-inf") if c.score is None else float(c.score),
        )
        best_s2 = pick_best_trained(stage2_pop) or stage2_pop[0]
        write_json(
            os.path.join(cfg.output_dir, "best_stage1.json"),
            candidate_to_dict(best_s1),
        )
        write_json(
            os.path.join(cfg.output_dir, "stage2_population.json"),
            {
                "population": [candidate_to_dict(c) for c in stage2_pop],
                "best_trained_id": best_s2.candidate_id,
                "history": _stage2_history(cl_result.history),
            },
        )
        write_json(
            os.path.join(cfg.output_dir, "best_stage2.json"),
            candidate_to_dict(best_s2),
        )
        console.status(
            f"Closed loop complete - best={best_s2.candidate_id} "
            f"fitness={candidate_fitness(best_s2):.3f} "
            f"epochs={cl_result.manifest.get('n_epochs')} "
            f"trained={cl_result.n_labeled_total}",
            stage="pipeline",
        )

        # ----- Stage III (long PPO on the final population) -----
        s3_runner = Stage3Runner(
            make_stage3_trainer_for_domain(pack, use_stub=bool(cfg.use_stub)),
            validator=self.validator,
            config=self._stage3_config(),
        )
        stage3_pop = s3_runner.run(stage2_pop)
        best_s3 = s3_runner.best_trained or best_s2
        write_json(
            os.path.join(cfg.output_dir, "stage3_population.json"),
            {
                "population": [candidate_to_dict(c) for c in stage3_pop],
                "best_trained_id": best_s3.candidate_id,
                "history": [r.as_dict() for r in s3_runner.history],
            },
        )
        write_json(
            os.path.join(cfg.output_dir, "best_stage3.json"),
            candidate_to_dict(best_s3),
        )
        write_json(
            os.path.join(cfg.output_dir, "final_candidate.json"),
            candidate_to_dict(best_s3),
        )
        console.status(
            f"Stage III complete - best={best_s3.candidate_id} "
            f"fitness={candidate_fitness(best_s3):.3f}",
            stage="pipeline",
        )

        manifest["best_stage1_id"] = best_s1.candidate_id
        manifest["best_stage2_id"] = best_s2.candidate_id
        manifest["best_stage3_id"] = best_s3.candidate_id
        manifest["closed_loop_summary"] = dict(cl_result.manifest)
        write_json(os.path.join(cfg.output_dir, "manifest.json"), manifest)

        console.final_run_summary(
            output_dir=cfg.output_dir,
            wall_seconds=time.perf_counter() - wall_t0,
            best_stage1=best_s1,
            best_stage2=best_s2,
            best_stage3=best_s3,
        )

        if not bool(cfg.fast):
            console.banner("Highway plots + animation")
            art = write_highway_run_artifacts(
                cfg.output_dir,
                episodes=2,
                seed=int(cfg.seed),
            )
            n_plots = len(art.get("plots") or [])
            viz_ok = bool((art.get("viz") or {}).get("ok"))
            console.status(
                f"plots={n_plots} viz={'ok' if viz_ok else 'skipped'} "
                f"→ {os.path.join(cfg.output_dir, 'plots')}",
                stage="pipeline",
            )

        return RaiseArtifacts(
            output_dir=cfg.output_dir,
            seed_code=seed_code,
            stage2_population=stage2_pop,
            stage3_population=stage3_pop,
            best_stage1=best_s1,
            best_stage2=best_s2,
            best_stage3=best_s3,
            manifest=manifest,
        )

"""
RAISE orchestrator (faithful replication baseline).

seed → Stage I → Stage II → Stage III. No AMFRS mechanisms.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from raise_core.explore import (
    RewardCandidate,
    StageIConfig,
    StageIEvolver,
)
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
from raise_core.llm import LLMClient, make_llm_client
from domains.crowdnav.reporting import candidate_to_dict, write_json
from raise_core.sandbox import RewardValidator
from raise_core.selection import (
    candidate_nav_scalar,
    navigation_scalar_from_dict,
    pick_best_trained,
)
from raise_core.ranking import (
    pick_candidate_by_ranking,
    produce_final_ranking,
)
from raise_core.proxy_consistency import compute_proxy_consistency
from raise_core.surrogate.gate import (
    gate_population,
    predict_population,
    surrogate_model_ready,
)
from raise_core.refine import (
    Stage2Config,
    Stage2Runner,
)
from raise_core.validate import (
    STAGE3_PAPER_STEPS,
    STAGE3_STEPS,
    Stage3Config,
    Stage3Runner,
)

logger = logging.getLogger(__name__)


@dataclass
class RaiseRunConfig:
    """End-to-end Algorithm 1 settings (paper defaults + practical overrides)."""

    output_dir: str = "results/raise_run"
    seed: int = 425
    # Domain pack under raise_core.domains (default preserves Algorithm 1 baseline).
    domain: str = DEFAULT_DOMAIN
    llm_provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    llm_model: Optional[str] = None
    # Stage I Score1: "dataset" (default, paper) | "smoke" (opt-in fast tests only)
    score1_mode: str = "dataset"
    stage1_dataset_path: str = "domains/crowdnav/data/stage1_dataset"

    # Stage I (Table 5 / §5.1)
    stage1_population: int = 8
    stage1_generations: int = 10

    # Stage II — paper Table 5: K2=8000 **gradient** steps (§4.3.2).
    stage2_rounds: int = 16
    stage2_train_steps: int = 8_000
    stage2_k2_unit: str = "gradient_steps"
    stage2_eval_episodes: int = 50
    stage2_horizon: int = 100
    stage2_use_stub: bool = False

    # Stage III
    stage3_rounds: int = 3
    stage3_train_steps: int = STAGE3_STEPS  # paper: 1e7 — see STAGE3_PAPER_STEPS
    stage3_eval_episodes: int = 500
    stage3_use_stub: bool = False
    stage3_run_h_sweep: bool = True

    # Paper-faithful defaults (Alg. 1 has no elitism; R2/R3 via LLM eval).
    elitism: bool = False
    final_rank: str = "llm"

    # Surrogate / Active Learning (opt-in; off unless model dir set / flags).
    surrogate_model_dir: Optional[str] = None
    surrogate_dataset: str = "domains/crowdnav/data/surrogate_dataset"
    surrogate_gate_stage3: bool = True
    surrogate_drop_fraction: float = 0.25
    surrogate_max_uncertainty_to_drop: float = 0.15
    surrogate_min_keep: int = 2
    active_learning: bool = False
    active_learning_queue: str = "domains/crowdnav/data/active_learning"
    active_learning_max_queries: int = 3
    active_learning_refit_every: int = 20

    # Closed-loop multi-fidelity (innovation; opt-in --closed-loop).
    closed_loop: bool = False
    closed_loop_no_al: bool = False
    closed_loop_al_max_per_epoch: int = 4
    closed_loop_min_labels_for_gate: int = 24
    closed_loop_k2: int = 8_000
    closed_loop_final_stage2_rounds: int = 0
    closed_loop_refit_every_new_labels: int = 8
    closed_loop_min_stage2_per_gen: int = 4
    # Crash-safe resume of the closed loop from output_dir/closed_loop/checkpoint.json.
    closed_loop_resume: bool = True
    # Keep surrogate model / dataset / AL queue inside output_dir (self-contained run).
    closed_loop_isolate_artifacts: bool = True
    # Feed Stage-II SR/CR/TR into D.2 mutation prompts (selective).
    closed_loop_proxy_feedback: bool = False
    closed_loop_proxy_feedback_min_labels: int = 16
    # Max in-loop D.3 rewrites per epoch (0 = off; needs proxy_feedback).
    closed_loop_proxy_feedback_d3_per_epoch: int = 0
    closed_loop_enable_refine: bool = False  # legacy → d3_per_epoch=1 when set
    # Phase 4: allow hard gate early when mean val MAE ≤ this (None = labels only).
    closed_loop_max_val_mae_for_gate: Optional[float] = None
    # Closed-loop next-gen parent order: score1 | scalar | hybrid.
    # Empty → domain default (highway=hybrid, else score1).
    closed_loop_evolve_rank: str = ""
    closed_loop_evolve_rank_score1_weight: float = 0.4
    # Highway wall-clock speed (same K2/K3 budget).
    highway_n_envs: int = 1
    highway_label_workers: int = 1
    highway_warm_start: bool = True
    highway_eval_mode: str = "both"
    # Crash-safe Stage III resume from output_dir/stage3/checkpoint.json (+ mid-PPO).
    stage3_resume: bool = True
    stage3_save_interval_updates: int = 50

    device: str = "cuda"
    # None → auto (min(16, cpu-1)); set low on 4GB GPUs to avoid OOM.
    num_processes: Optional[int] = None
    # AUDIT.md §8.1 — first validation pass defaults to without_random.
    randomization_regime: str = "without_random"
    # AUDIT.md §8.2 choice (a): Stage II/III use GST-inferred obs.
    predict_method: str = "inferred"
    # Crowd size for Stage II/III train+eval (paper=20). Lower for easier debugging.
    human_num: int = 20
    # Fast dry-run profile (tests / laptop)
    fast: bool = False

    def apply_fast_profile(self) -> None:
        """Seconds-scale Algorithm 1 walk with stub trainers + tiny budgets."""
        self.fast = True
        self.score1_mode = "smoke"
        self.stage1_population = 2
        self.stage1_generations = 1
        self.stage2_rounds = 1
        self.stage2_train_steps = 8
        # Tiny absolute env-step budget for stub dry-runs (not paper unit).
        self.stage2_k2_unit = "env_steps"
        self.stage2_eval_episodes = 2
        self.stage2_horizon = 5
        self.stage2_use_stub = True
        self.stage3_rounds = 1
        self.stage3_train_steps = 8
        self.stage3_eval_episodes = 2
        self.stage3_use_stub = True
        self.stage3_run_h_sweep = True
        self.llm_provider = "seed"
        # Stubs never load GST; keep flags consistent for config builders.
        self.predict_method = "none"

    def apply_easy_profile(self) -> None:
        """
        Easier simulator for pipeline result-getting (not paper claims).

        - No GST prediction (faster, simpler obs)
        - Fewer humans (5 vs 20)
        - Longer Stage II horizon (~50s episodes)
        """
        self.predict_method = "none"
        self.human_num = 5
        self.stage2_horizon = 200  # 200 * 0.25s = 50s
        self.stage3_run_h_sweep = False


@dataclass
class RaiseArtifacts:
    """Paths / populations produced by one Algorithm 1 run."""

    output_dir: str
    seed_code: str
    stage1_population: List[RewardCandidate] = field(default_factory=list)
    stage2_population: List[RewardCandidate] = field(default_factory=list)
    stage3_population: List[RewardCandidate] = field(default_factory=list)
    best_stage1: Optional[RewardCandidate] = None
    best_stage2: Optional[RewardCandidate] = None
    best_stage3: Optional[RewardCandidate] = None
    manifest: Dict[str, Any] = field(default_factory=dict)


class RaisePipeline:
    """Reproduce Algorithm 1 end-to-end and persist JSON artifacts."""

    def __init__(
        self,
        config: Optional[RaiseRunConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        checkpoint_store: Optional[Any] = None,
        domain_pack: Optional[DomainPack] = None,
    ) -> None:
        self.config = config or RaiseRunConfig()
        self.llm = llm
        # Optional paper-scale resume store (seed/stage/round/candidate).
        self.checkpoint_store = checkpoint_store
        self.domain_pack = domain_pack or load_domain(self.config.domain)
        self.validator = make_validator_for_domain(self.domain_pack)

    def _build_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        seed = self.domain_pack.seed_reward_source if self.domain_pack else None
        if self.config.llm_model is None:
            return make_llm_client(
                self.config.llm_provider, base_code=seed
            )
        return make_llm_client(
            self.config.llm_provider,
            model=self.config.llm_model,
            base_code=seed,
        )

    def _score_fn(self):
        return make_score_fn_for_domain(
            self.domain_pack,
            mode=self.config.score1_mode,
            dataset_path=self.config.stage1_dataset_path,
        )

    @staticmethod
    def _include_global_best(
        population: List[RewardCandidate], global_best: RewardCandidate
    ) -> List[RewardCandidate]:
        """Keep the best Stage-I candidate in the population sent to Stage II."""
        if any(candidate.candidate_id == global_best.candidate_id for candidate in population):
            return population
        worst_index = min(
            range(len(population)),
            key=lambda index: population[index].score if population[index].score is not None else float("-inf"),
        )
        population[worst_index] = global_best
        return population

    def _best(self, population: List[RewardCandidate]) -> RewardCandidate:
        ranked = sorted(
            population,
            key=lambda c: (
                float(c.score) if c.score is not None else float("-inf"),
                float((c.metadata or {}).get("last_metrics", {}).get("SR", 0.0)),
            ),
            reverse=True,
        )
        return ranked[0]

    def _run_closed_loop_branch(
        self,
        cfg: RaiseRunConfig,
        *,
        llm: Any,
        pack: Any,
        regime: str,
        predict_method: str,
    ):
        """Innovation path: interleaved Score1 ↔ Stage II short ↔ Surrogate+AL."""
        from raise_core.raise_loop import ClosedLoopConfig, ClosedLoopRunner

        console.banner("Closed-loop multi-fidelity (innovation)")
        from raise_core.raise_loop.evolve_rank import parse_evolve_rank

        cl_cfg = ClosedLoopConfig(
            population_size=int(cfg.stage1_population),
            generations=int(cfg.stage1_generations),
            min_labels_for_gate=int(cfg.closed_loop_min_labels_for_gate),
            stage2_train_steps=int(cfg.closed_loop_k2),
            k2_unit=str(cfg.stage2_k2_unit),
            drop_fraction=float(cfg.surrogate_drop_fraction),
            max_uncertainty_to_drop=float(cfg.surrogate_max_uncertainty_to_drop),
            min_keep=int(cfg.surrogate_min_keep),
            al_enabled=not bool(cfg.closed_loop_no_al),
            al_max_per_epoch=int(cfg.closed_loop_al_max_per_epoch),
            final_stage2_rounds=int(cfg.closed_loop_final_stage2_rounds),
            refit_every_new_labels=int(cfg.closed_loop_refit_every_new_labels),
            min_stage2_per_gen=int(cfg.closed_loop_min_stage2_per_gen),
            surrogate_model_dir=str(cfg.surrogate_model_dir or "artifacts/surrogate"),
            surrogate_dataset=str(cfg.surrogate_dataset),
            al_root=str(cfg.active_learning_queue),
            stage1_dataset_path=str(cfg.stage1_dataset_path),
            seed=int(cfg.seed),
            device=str(cfg.device),
            num_processes=1 if cfg.num_processes is None else max(1, int(cfg.num_processes)),
            use_stub=bool(cfg.stage2_use_stub or cfg.fast),
            llm_provider=str(cfg.llm_provider),
            output_dir=str(cfg.output_dir),
            keep_runtime_elite=bool(cfg.elitism),
            human_num=max(1, int(cfg.human_num)),
            predict_method=str(predict_method),
            randomization_regime=str(regime),
            horizon_steps=max(1, int(cfg.stage2_horizon)),
            resume=bool(cfg.closed_loop_resume),
            isolate_run_artifacts=bool(cfg.closed_loop_isolate_artifacts),
            proxy_feedback=bool(cfg.closed_loop_proxy_feedback),
            proxy_feedback_min_labels=int(cfg.closed_loop_proxy_feedback_min_labels),
            proxy_feedback_d3_per_epoch=int(
                cfg.closed_loop_proxy_feedback_d3_per_epoch
            ),
            enable_refine=bool(cfg.closed_loop_enable_refine),
            domain=str(pack.name),
            eval_episodes=(
                int(cfg.stage2_eval_episodes)
                if getattr(cfg, "stage2_eval_episodes", None) is not None
                else None
            ),
            max_val_mae_for_gate=(
                float(cfg.closed_loop_max_val_mae_for_gate)
                if getattr(cfg, "closed_loop_max_val_mae_for_gate", None) is not None
                else None
            ),
            evolve_rank=parse_evolve_rank(
                getattr(cfg, "closed_loop_evolve_rank", None),
                domain=str(pack.name),
            ),
            evolve_rank_score1_weight=float(
                getattr(cfg, "closed_loop_evolve_rank_score1_weight", 0.4) or 0.4
            ),
            highway_n_envs=max(
                1, int(getattr(cfg, "highway_n_envs", None) or 1)
            ),
            highway_label_workers=max(
                1, int(getattr(cfg, "highway_label_workers", None) or 1)
            ),
            highway_warm_start=bool(getattr(cfg, "highway_warm_start", True)),
            highway_eval_mode=str(
                getattr(cfg, "highway_eval_mode", None) or "both"
            ),
        )
        if cfg.fast:
            cl_cfg.apply_fast_profile()
            cl_cfg.output_dir = str(cfg.output_dir)
            cl_cfg.surrogate_model_dir = str(
                cfg.surrogate_model_dir or "artifacts/surrogate"
            )
            cl_cfg.surrogate_dataset = str(cfg.surrogate_dataset)

        cl_result = ClosedLoopRunner(
            cl_cfg,
            llm=llm,
            score_fn=self._score_fn(),
            trainer=make_stage2_trainer_for_domain(
                pack, use_stub=bool(cfg.stage2_use_stub or cfg.fast)
            ),
            validator=self.validator,
        ).run()

        # The runner may have nested surrogate artifacts under output_dir; point the
        # rest of the pipeline (Stage III gate, manifest) at what it actually wrote.
        cfg.surrogate_model_dir = str(cl_result.model_dir)
        cfg.surrogate_dataset = str(cl_result.dataset_dir)

        stage1_pop = list(cl_result.population)
        best_s1 = max(
            stage1_pop,
            key=lambda c: float("-inf") if c.score is None else float(c.score),
        )
        write_json(
            os.path.join(cfg.output_dir, "stage1_population.json"),
            {
                "mode": "closed_loop_v1",
                "ranking": [c.candidate_id for c in stage1_pop],
                "population": [candidate_to_dict(c) for c in stage1_pop],
            },
        )
        write_json(
            os.path.join(cfg.output_dir, "best_stage1.json"),
            candidate_to_dict(best_s1),
        )

        stage2_pop = list(stage1_pop)
        s2_runner = None
        if int(cfg.closed_loop_final_stage2_rounds) > 0:
            s2_cfg = Stage2Config(
                population_size=len(stage2_pop),
                rounds=int(cfg.closed_loop_final_stage2_rounds),
                train_env_steps=cfg.stage2_train_steps,
                k2_unit=str(cfg.stage2_k2_unit),
                eval_episodes=cfg.stage2_eval_episodes,
                horizon_steps=cfg.stage2_horizon,
                seed=cfg.seed,
                device=cfg.device,
                num_processes=cfg.num_processes,
                human_num=int(cfg.human_num),
                output_root=os.path.join(cfg.output_dir, "stage2_train"),
                randomization_regime=regime,
                predict_method=predict_method,
                env_name=env_name_for_predict_method(predict_method),
                protect_elite_refine=bool(cfg.elitism),
                inject_elite=bool(cfg.elitism),
            )
            s2_trainer = make_stage2_trainer_for_domain(
                pack, use_stub=cfg.stage2_use_stub
            )
            s2_runner = Stage2Runner(
                llm,
                s2_trainer,
                validator=self.validator,
                config=s2_cfg,
                prompts=pack.prompts,
            )
            stage2_pop = s2_runner.run(stage2_pop)

        rank_pool_s2 = (
            list(s2_runner.trained_snapshots)
            if s2_runner is not None and s2_runner.trained_snapshots
            else list(stage2_pop)
        )
        r2_rank = produce_final_ranking(rank_pool_s2, mode=cfg.final_rank, llm=llm)
        best_s2 = pick_candidate_by_ranking(r2_rank, rank_pool_s2)
        if best_s2 is None:
            best_s2 = stage2_pop[0] if stage2_pop else best_s1
        write_json(
            os.path.join(cfg.output_dir, "stage2_population.json"),
            {
                "mode": "closed_loop_v1",
                "population": [candidate_to_dict(c) for c in stage2_pop],
                "best_trained_id": best_s2.candidate_id if best_s2 else None,
                "final_ranking_R2": r2_rank,
                "closed_loop": cl_result.manifest,
            },
        )
        write_json(
            os.path.join(cfg.output_dir, "best_stage2.json"),
            candidate_to_dict(best_s2),
        )
        console.status(
            f"Closed-loop complete - best_s2={best_s2.candidate_id} "
            f"epochs={cl_result.manifest.get('n_epochs')} "
            f"labels={cl_result.n_labeled_total}",
            stage="pipeline",
        )
        return (
            stage1_pop,
            best_s1,
            stage2_pop,
            best_s2,
            r2_rank,
            dict(cl_result.manifest),
        )

    def run(self) -> RaiseArtifacts:
        import time

        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        wall_t0 = time.perf_counter()
        from domains.crowdnav.regime import (
            EVOLUTION_PREDICT_METHOD,
            assert_gst_matches_regime,
            env_name_for_predict_method,
            gst_model_dir_for_regime,
            parse_regime,
            randomization_flags,
        )

        regime = parse_regime(cfg.randomization_regime)
        cfg.randomization_regime = regime
        predict_method = (cfg.predict_method or EVOLUTION_PREDICT_METHOD).strip().lower()
        cfg.predict_method = predict_method
        attrs, goals = randomization_flags(regime)
        console.banner("RAISE")
        console.status(
            f"output={cfg.output_dir} seed={cfg.seed} llm={cfg.llm_provider} "
            f"fast={cfg.fast} device={cfg.device}"
        )
        llm = self._build_llm()
        pack = self.domain_pack
        if pack.name != str(cfg.domain).strip().lower():
            raise ValueError(
                f"Domain pack mismatch: config.domain={cfg.domain!r}, "
                f"pack.name={pack.name!r}"
            )
        is_crowdnav = pack.name == "crowdnav"
        pipeline_profile = str(
            (pack.metadata or {}).get("pipeline_profile") or pack.name
        )
        if is_crowdnav:
            console.status(
                f"regime={regime} (randomize_attributes={attrs}, "
                f"random_goal_changing={goals}); predict_method={predict_method}"
            )
            if predict_method == "inferred":
                assert_gst_matches_regime(
                    gst_model_dir_for_regime(regime),
                    regime,
                    predict_method=predict_method,
                    entry_point="pipeline_startup",
                )
            else:
                assert_gst_matches_regime(
                    "",
                    regime,
                    predict_method=predict_method,
                    entry_point="pipeline_startup",
                )
        else:
            console.status(
                f"domain_profile={pipeline_profile} (CrowdNav GST/human_num path skipped)"
            )

        seed_code = pack.seed_reward_source.strip() + "\n"

        manifest: Dict[str, Any] = {
            "algorithm": "RAISE",
            "domain": pack.name,
            "domain_display_name": pack.display_name,
            "pipeline_profile": pipeline_profile,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "stage3_paper_steps": STAGE3_PAPER_STEPS,
            "evolution_randomization_regime": regime if is_crowdnav else None,
            "predict_method": predict_method if is_crowdnav else None,
            "notes": (
                "Faithful replication baseline. No AMFRS novelty / archive / "
                "Pareto / adaptive controller. "
                + (
                    "Obs-space choice (a): Stage II/III use predict_method=inferred "
                    "when not --fast (AUDIT.md §8)."
                    if is_crowdnav
                    else f"Domain {pack.name!r} uses pack-local trainers/metrics."
                )
            ),
        }
        console.status(f"domain={pack.name} ({pack.display_name})")
        write_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
        with open(os.path.join(cfg.output_dir, "seed_reward.py"), "w", encoding="utf-8") as f:
            f.write(seed_code)

        closed_loop_summary: Optional[Dict[str, Any]] = None
        surrogate_preds_s1: Optional[List[Dict[str, Any]]] = None
        al_summary: Optional[Dict[str, Any]] = None
        s2_runner = None

        if bool(cfg.closed_loop):
            stage1_pop, best_s1, stage2_pop, best_s2, r2_rank, closed_loop_summary = (
                self._run_closed_loop_branch(cfg, llm=llm, pack=pack, regime=regime, predict_method=predict_method)
            )
        else:
            # ----- Stage I -----
            n = cfg.stage1_population
            n_crossover = min(2, n)
            n_mutation = min(4, max(0, n - n_crossover))
            n_random = n - n_crossover - n_mutation
            s1_cfg = StageIConfig(
                population_size=n,
                generations=cfg.stage1_generations,
                n_crossover=n_crossover,
                n_mutation=n_mutation,
                n_random=n_random,
                keep_runtime_elite=bool(cfg.elitism),
            )
            evolver = StageIEvolver(
                llm,
                score_fn=self._score_fn(),
                validator=self.validator,
                config=s1_cfg,
                prompts=pack.prompts,
                rejection_log_path=os.path.join(cfg.output_dir, "stage1_rejections.jsonl"),
            )
            stage1_pop = evolver.run()
            if evolver.global_best is None:
                raise RuntimeError("Stage I did not produce a global best candidate.")
            best_s1 = evolver.global_best
            if bool(cfg.elitism):
                stage1_pop = self._include_global_best(stage1_pop, best_s1)
            write_json(
                os.path.join(cfg.output_dir, "stage1_population.json"),
                {
                    "ranking": [c.candidate_id for c in stage1_pop],
                    "population": [candidate_to_dict(c) for c in stage1_pop],
                    "history": [
                        {
                            "generation": h.generation,
                            "ranking": h.ranking,
                            "best_id": h.best_id,
                            "best_score": h.best_score,
                            "reflection": h.reflection,
                        }
                        for h in evolver.history
                    ],
                },
            )
            write_json(
                os.path.join(cfg.output_dir, "best_stage1.json"),
                candidate_to_dict(best_s1),
            )
            console.status(
                f"Stage I complete - best={best_s1.candidate_id} score={best_s1.score}",
                stage="pipeline",
            )

            if surrogate_model_ready(cfg.surrogate_model_dir):
                console.status(
                    f"Surrogate predict after Stage I ({cfg.surrogate_model_dir})",
                    stage="pipeline",
                )
                try:
                    surrogate_preds_s1, _model = predict_population(
                        stage1_pop,
                        str(cfg.surrogate_model_dir),
                        score_fn=self._score_fn(),
                    )
                    write_json(
                        os.path.join(cfg.output_dir, "surrogate_preds_stage1.json"),
                        {"predictions": surrogate_preds_s1},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Surrogate Stage I predict failed: %s", exc)
                    console.warn(f"Surrogate predict skipped: {exc}", stage="pipeline")
                    surrogate_preds_s1 = None

                if bool(cfg.active_learning):
                    from raise_core.active_learning.loop import (
                        run_active_learning_step,
                    )

                    console.status("Active learning step after Stage I", stage="pipeline")
                    al_summary = run_active_learning_step(
                        surrogate_model_dir=str(cfg.surrogate_model_dir),
                        queue_root=str(cfg.active_learning_queue),
                        max_queries=int(cfg.active_learning_max_queries),
                        refit_every=int(cfg.active_learning_refit_every),
                        candidates=list(stage1_pop),
                        surrogate_dataset=str(cfg.surrogate_dataset),
                        use_stub=bool(cfg.stage2_use_stub or cfg.fast),
                        seed=int(cfg.seed),
                        stage1_dataset_path=str(cfg.stage1_dataset_path),
                    )
                    write_json(
                        os.path.join(cfg.output_dir, "active_learning_step.json"),
                        {k: v for k, v in al_summary.items() if k != "executed"},
                    )
                    if al_summary.get("status") == "error":
                        console.warn(
                            f"Active learning: {al_summary.get('message')}",
                            stage="pipeline",
                        )
                    else:
                        console.status(
                            f"AL enqueued={al_summary.get('n_enqueued')} "
                            f"executed={al_summary.get('n_executed')} "
                            f"refit={al_summary.get('refit')}",
                            stage="pipeline",
                        )
            elif cfg.surrogate_model_dir:
                console.warn(
                    f"Surrogate dir set but no model.joblib at {cfg.surrogate_model_dir}; "
                    "skipping surrogate/AL",
                    stage="pipeline",
                )
            elif cfg.active_learning:
                console.warn(
                    "active_learning=True but surrogate_model_dir unset/missing; skipping AL",
                    stage="pipeline",
                )

            # ----- Stage II -----
            s2_cfg = Stage2Config(
                population_size=len(stage1_pop),
                rounds=cfg.stage2_rounds,
                train_env_steps=cfg.stage2_train_steps,
                k2_unit=str(cfg.stage2_k2_unit),
                eval_episodes=cfg.stage2_eval_episodes,
                horizon_steps=cfg.stage2_horizon,
                seed=cfg.seed,
                device=cfg.device,
                num_processes=cfg.num_processes,
                human_num=int(cfg.human_num),
                output_root=os.path.join(cfg.output_dir, "stage2_train"),
                randomization_regime=regime,
                predict_method=predict_method,
                env_name=env_name_for_predict_method(predict_method),
                protect_elite_refine=bool(cfg.elitism),
                inject_elite=bool(cfg.elitism),
            )
            if cfg.stage2_use_stub:
                console.status(
                    "Stage II using domain StubPolicyTrainer (via adapter)",
                    stage="pipeline",
                )
            s2_trainer = make_stage2_trainer_for_domain(
                pack, use_stub=cfg.stage2_use_stub
            )
            s2_runner = Stage2Runner(
                llm,
                s2_trainer,
                validator=self.validator,
                config=s2_cfg,
                prompts=pack.prompts,
            )
            if self.checkpoint_store is not None:
                s2_runner.checkpoint_store = self.checkpoint_store
                s2_runner.checkpoint_seed = int(cfg.seed)
            stage2_pop = s2_runner.run(stage1_pop)
            rank_pool_s2 = list(s2_runner.trained_snapshots) or list(stage2_pop)
            r2_rank = produce_final_ranking(
                rank_pool_s2, mode=cfg.final_rank, llm=llm
            )
            best_s2 = pick_candidate_by_ranking(r2_rank, rank_pool_s2)
            if best_s2 is None:
                best_s2 = s2_runner.best_trained or self._best_by_ever_metrics(
                    stage2_pop, s2_runner.history, s2_runner.trained_snapshots
                )
            write_json(
                os.path.join(cfg.output_dir, "stage2_population.json"),
                {
                    "population": [candidate_to_dict(c) for c in stage2_pop],
                    "best_trained_id": (
                        best_s2.candidate_id if best_s2 is not None else None
                    ),
                    "final_ranking_R2": r2_rank,
                    "history": [
                        {
                            "round_index": r.round_index,
                            "candidate_id": r.candidate_id,
                            "metrics": r.metrics.as_dict(),
                            "refined": r.refined,
                            "kept_previous": r.kept_previous,
                        }
                        for r in s2_runner.history
                    ],
                },
            )
            write_json(
                os.path.join(cfg.output_dir, "best_stage2.json"),
                candidate_to_dict(best_s2),
            )
            console.status(
                f"Stage II complete - best={best_s2.candidate_id} "
                f"fitness={candidate_nav_scalar(best_s2):.3f} "
                f"R2_mode={r2_rank.get('mode')}",
                stage="pipeline",
            )

        # ----- Surrogate gate before Stage III -----
        stage3_input_pop = list(stage2_pop)
        surrogate_gate_report: Optional[Dict[str, Any]] = None
        surrogate_preds_s2: Optional[List[Dict[str, Any]]] = None
        stage3_elite_report: Optional[Dict[str, Any]] = None
        if (
            surrogate_model_ready(cfg.surrogate_model_dir)
            and bool(cfg.surrogate_gate_stage3)
        ):
            from raise_core.raise_loop.gate_policy import surrogate_hard_gate_ready

            n_lab = 0
            try:
                from raise_core.surrogate.dataset_io import existing_example_ids

                n_lab = len(existing_example_ids(str(cfg.surrogate_dataset)))
            except Exception:  # noqa: BLE001
                n_lab = 0
            fit_m = None
            mpath = os.path.join(str(cfg.surrogate_model_dir or ""), "metrics.json")
            if os.path.isfile(mpath):
                try:
                    with open(mpath, encoding="utf-8") as fh:
                        fit_m = json.load(fh)
                except Exception:  # noqa: BLE001
                    fit_m = None
            min_lab = int(cfg.closed_loop_min_labels_for_gate)
            max_mae = getattr(cfg, "closed_loop_max_val_mae_for_gate", None)
            ready, ready_reason = surrogate_hard_gate_ready(
                n_labeled=n_lab,
                min_labels=min_lab,
                fit_metrics=fit_m if isinstance(fit_m, dict) else None,
                max_val_mae=float(max_mae) if max_mae is not None else None,
            )
            if not ready:
                surrogate_gate_report = {
                    "enabled": False,
                    "soft": True,
                    "reason": ready_reason,
                    "n_kept": len(stage3_input_pop),
                    "n_dropped": 0,
                }
                console.status(
                    f"Surrogate Stage III gate soft ({ready_reason})",
                    stage="pipeline",
                )
            else:
                try:
                    surrogate_preds_s2, _m2 = predict_population(
                        stage3_input_pop,
                        str(cfg.surrogate_model_dir),
                        score_fn=self._score_fn(),
                    )
                    write_json(
                        os.path.join(cfg.output_dir, "surrogate_preds_stage2.json"),
                        {"predictions": surrogate_preds_s2},
                    )
                    stage3_input_pop, surrogate_gate_report = gate_population(
                        stage3_input_pop,
                        surrogate_preds_s2,
                        drop_fraction=float(cfg.surrogate_drop_fraction),
                        max_uncertainty_to_drop=float(
                            cfg.surrogate_max_uncertainty_to_drop
                        ),
                        min_keep=int(cfg.surrogate_min_keep),
                    )
                    write_json(
                        os.path.join(cfg.output_dir, "surrogate_gate_stage3.json"),
                        surrogate_gate_report,
                    )
                    console.status(
                        f"Surrogate Stage III gate: kept={surrogate_gate_report['n_kept']} "
                        f"dropped={surrogate_gate_report['n_dropped']}",
                        stage="pipeline",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Surrogate Stage III gate failed: %s", exc)
                    console.warn(f"Surrogate gate skipped: {exc}", stage="pipeline")
                    stage3_input_pop = list(stage2_pop)
                    surrogate_gate_report = {"enabled": False, "error": str(exc)}

        # Force elites: kept ∪ best_s2 ∪ best_scalar; Score1-best if strong (highway).
        try:
            from raise_core.raise_loop.gate_policy import assemble_stage3_population

            stage3_input_pop, stage3_elite_report = assemble_stage3_population(
                stage3_input_pop,
                full_pool=list(stage2_pop),
                best_s2=best_s2,
                best_s1=best_s1,
                domain=str(pack.name),
            )
            write_json(
                os.path.join(cfg.output_dir, "stage3_elite_assembly.json"),
                stage3_elite_report,
            )
            if stage3_elite_report.get("forced_elites") or stage3_elite_report.get(
                "score1_best_added"
            ):
                console.status(
                    f"Stage III elites: {stage3_elite_report.get('forced_elites')} "
                    f"score1_added={stage3_elite_report.get('score1_best_added')}",
                    stage="pipeline",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage III elite assembly failed: %s", exc)

        # ----- Stage III -----
        # H-sweep only for CrowdNav (human counts). Other domains skip it.
        h_train = max(1, int(cfg.human_num))
        run_h_sweep = bool(cfg.stage3_run_h_sweep) and is_crowdnav
        if not is_crowdnav and bool(cfg.stage3_run_h_sweep):
            console.status(
                f"H-sweep disabled for domain={pack.name}",
                stage="pipeline",
            )
        if run_h_sweep:
            swept = [h for h in (5, 10, 15, 20) if h <= h_train]
            if h_train not in swept:
                swept.append(h_train)
            human_counts = tuple(sorted(set(swept)))
            if len(human_counts) <= 1:
                console.status(
                    f"H-sweep enabled but only H={human_counts} "
                    f"(train human_num={h_train}); raise --human-num "
                    f"for multi-H Table-6 generalization",
                    stage="pipeline",
                )
        else:
            human_counts = (h_train,)
        # Closed-loop / Windows: unset nproc must not auto-scale to cpu-1 (OOM).
        stage3_nproc = cfg.num_processes
        if stage3_nproc is None and bool(cfg.closed_loop):
            stage3_nproc = 1
        s3_cfg = Stage3Config(
            population_size=len(stage3_input_pop),
            rounds=cfg.stage3_rounds,
            train_env_steps=cfg.stage3_train_steps,
            eval_episodes=cfg.stage3_eval_episodes,
            seed=cfg.seed,
            device=cfg.device,
            num_processes=stage3_nproc,
            train_human_num=h_train,
            output_root=os.path.join(cfg.output_dir, "stage3_train"),
            human_counts=human_counts,
            randomization_regime=regime,
            predict_method=predict_method,
            env_name=env_name_for_predict_method(predict_method),
            protect_elite_refine=bool(cfg.elitism),
            inject_elite=bool(cfg.elitism),
            resume=bool(getattr(cfg, "stage3_resume", True)),
            save_interval_updates=int(
                getattr(cfg, "stage3_save_interval_updates", 50) or 0
            ),
        )
        if cfg.stage3_use_stub:
            console.status(
                "Stage III using domain StubPolicyTrainer (via adapter)",
                stage="pipeline",
            )
        s3_trainer = make_stage3_trainer_for_domain(
            pack, use_stub=cfg.stage3_use_stub
        )
        s3_runner = Stage3Runner(
            llm,
            s3_trainer,
            validator=self.validator,
            config=s3_cfg,
            prompts=pack.prompts,
        )
        if self.checkpoint_store is not None:
            s3_runner.checkpoint_store = self.checkpoint_store
            s3_runner.checkpoint_seed = int(cfg.seed)
        stage3_pop = s3_runner.run(stage3_input_pop, run_h_sweep=run_h_sweep)
        rank_pool_s3 = list(s3_runner.trained_snapshots) or list(stage3_pop)
        r3_rank = produce_final_ranking(
            rank_pool_s3, mode=cfg.final_rank, llm=llm
        )
        best_s3 = pick_candidate_by_ranking(r3_rank, rank_pool_s3)
        if best_s3 is None:
            best_s3 = s3_runner.best_trained or self._best_by_ever_metrics(
                stage3_pop, s3_runner.history, s3_runner.trained_snapshots
            )
        write_json(
            os.path.join(cfg.output_dir, "stage3_population.json"),
            {
                "population": [candidate_to_dict(c) for c in stage3_pop],
                "best_trained_id": (
                    best_s3.candidate_id if best_s3 is not None else None
                ),
                "final_ranking_R3": r3_rank,
                "history": [
                    {
                        "round_index": r.round_index,
                        "candidate_id": r.candidate_id,
                        "metrics": r.metrics.as_dict(),
                        "refined": r.refined,
                        "kept_previous": r.kept_previous,
                        "checkpoint_path": r.checkpoint_path,
                    }
                    for r in s3_runner.history
                ],
                "h_sweep": [
                    {
                        "candidate_id": rep.candidate_id,
                        "by_human_count": {
                            str(h): m.as_dict()
                            for h, m in rep.by_human_count.items()
                        },
                        "summary_table": rep.summary_table(),
                    }
                    for rep in s3_runner.sweep_reports
                ],
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
            f"fitness={candidate_nav_scalar(best_s3):.3f} "
            f"R3_mode={r3_rank.get('mode')}",
            stage="pipeline",
        )

        manifest["best_stage1_id"] = best_s1.candidate_id
        manifest["best_stage2_id"] = best_s2.candidate_id
        manifest["best_stage3_id"] = best_s3.candidate_id
        manifest["final_ranking_R2"] = r2_rank
        manifest["final_ranking_R3"] = r3_rank
        manifest["elitism"] = bool(cfg.elitism)
        manifest["stage2_k2_unit"] = str(cfg.stage2_k2_unit)
        manifest["final_rank"] = str(cfg.final_rank)
        manifest["surrogate"] = {
            "model_dir": cfg.surrogate_model_dir,
            "enabled": surrogate_model_ready(cfg.surrogate_model_dir),
            "gate_stage3": bool(cfg.surrogate_gate_stage3),
            "n_preds_stage1": (
                len(surrogate_preds_s1) if surrogate_preds_s1 is not None else 0
            ),
            "n_preds_stage2": (
                len(surrogate_preds_s2) if surrogate_preds_s2 is not None else 0
            ),
            "gate": surrogate_gate_report,
            "active_learning": bool(cfg.active_learning),
            "active_learning_summary": (
                None
                if al_summary is None
                else {k: v for k, v in al_summary.items() if k != "executed"}
            ),
            "closed_loop": bool(cfg.closed_loop),
            "closed_loop_summary": closed_loop_summary,
        }

        # §4.3.4 proxy consistency (Spearman + top-k on genome fingerprints).
        # Closed-loop with final_stage2_rounds=0 leaves s2_runner=None; use pop.
        stage2_for_consistency = (
            list(s2_runner.trained_snapshots)
            if s2_runner is not None and s2_runner.trained_snapshots
            else list(stage2_pop)
        )
        stage3_for_consistency = (
            list(s3_runner.trained_snapshots)
            if s3_runner is not None and s3_runner.trained_snapshots
            else list(stage3_pop)
        )
        consistency = compute_proxy_consistency(
            stage1=stage1_pop,
            stage2=stage2_for_consistency,
            stage3=stage3_for_consistency,
            top_k=min(3, max(1, len(stage1_pop))),
        )
        write_json(
            os.path.join(cfg.output_dir, "proxy_consistency.json"),
            consistency,
        )
        manifest["proxy_consistency"] = {
            "stage1_vs_stage2_rho": (
                consistency.get("stage1_vs_stage2", {})
                .get("spearman", {})
                .get("rho")
            ),
            "stage2_vs_stage3_rho": (
                consistency.get("stage2_vs_stage3", {})
                .get("spearman", {})
                .get("rho")
            ),
            "stage2_vs_stage3_lineage_rho": (
                consistency.get("stage2_vs_stage3_lineage", {})
                .get("spearman", {})
                .get("rho")
            ),
            "stage2_vs_stage3_preferred": consistency.get(
                "stage2_vs_stage3_preferred"
            ),
            "stage1_vs_stage2_top_k": (
                consistency.get("stage1_vs_stage2", {})
                .get("top_k", {})
                .get("preservation")
            ),
            "stage2_vs_stage3_top_k": (
                consistency.get("stage2_vs_stage3", {})
                .get("top_k", {})
                .get("preservation")
            ),
        }
        s12 = consistency.get("stage1_vs_stage2", {}).get("spearman", {})
        s23 = consistency.get("stage2_vs_stage3", {}).get("spearman", {})
        s23l = consistency.get("stage2_vs_stage3_lineage", {}).get("spearman", {})
        console.status(
            f"proxy consistency ρ12={s12.get('rho')} "
            f"(n={s12.get('n_overlap')}) ρ23={s23.get('rho')} "
            f"(n={s23.get('n_overlap')}) ρ23_lineage={s23l.get('rho')} "
            f"(n={s23l.get('n_overlap')})",
            stage="pipeline",
        )

        write_json(os.path.join(cfg.output_dir, "manifest.json"), manifest)

        wall = time.perf_counter() - wall_t0
        console.final_run_summary(
            output_dir=cfg.output_dir,
            wall_seconds=wall,
            best_stage1=best_s1,
            best_stage2=best_s2,
            best_stage3=best_s3,
            closed_loop=bool(cfg.closed_loop),
        )

        if pack.name == "highway" and not bool(getattr(cfg, "fast", False)):
            try:
                from domains.highway.post_run import write_highway_run_artifacts

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
            except Exception as exc:  # noqa: BLE001
                console.status(
                    f"post-run plots/viz failed (non-fatal): {exc}",
                    stage="pipeline",
                )

        return RaiseArtifacts(
            output_dir=cfg.output_dir,
            seed_code=seed_code,
            stage1_population=stage1_pop,
            stage2_population=stage2_pop,
            stage3_population=stage3_pop,
            best_stage1=best_s1,
            best_stage2=best_s2,
            best_stage3=best_s3,
            manifest=manifest,
        )

    @staticmethod
    def _best_by_ever_metrics(
        population,
        history,
        trained_snapshots=None,
    ) -> RewardCandidate:
        """
        Pick the best-ever trained genome by SR - CR - 0.5*TR across all rounds.

        Prefers explicit trained snapshots (correct code+metrics+checkpoint).
        Falls back to history scan + final population mapping.
        """
        if trained_snapshots:
            best = pick_best_trained(trained_snapshots)
            if best is not None:
                return best

        best_hist = None
        best_score = float("-inf")
        if history:
            for r in history:
                score = float(r.metrics.scalar_score())
                if score > best_score:
                    best_score = score
                    best_hist = r

        def _key(c: RewardCandidate):
            m = (c.metadata or {}).get("last_metrics")
            if m:
                return navigation_scalar_from_dict(m)
            if best_hist is not None and (
                c.candidate_id == best_hist.candidate_id
                or best_hist.candidate_id in (c.parent_ids or ())
            ):
                return best_score
            return float(c.score or float("-inf"))

        return max(population, key=_key)

    # Backward-compatible alias (older call sites / notebooks).
    _best_by_last_metrics = _best_by_ever_metrics

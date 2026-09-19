"""
EvoNav Algorithm 1 orchestrator (faithful replication baseline).

seed → Stage I → Stage II → Stage III. No AMFRS mechanisms.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from crowd_nav.reward_search.evolver import (
    RewardCandidate,
    StageIConfig,
    StageIEvolver,
)
from crowd_nav.reward_search import console
from crowd_nav.domains import (
    DEFAULT_DOMAIN,
    DomainPack,
    load_domain,
    make_score_fn_for_domain,
    make_stage2_trainer_for_domain,
    make_stage3_trainer_for_domain,
)
from crowd_nav.reward_search.llm import LLMClient, make_llm_client
from crowd_nav.reward_search.reporting import candidate_to_dict, write_json
from crowd_nav.reward_search.sandbox import RewardValidator
from crowd_nav.reward_search.selection import (
    candidate_nav_scalar,
    navigation_scalar_from_dict,
    pick_best_trained,
)
from crowd_nav.reward_search.ranking import (
    pick_candidate_by_ranking,
    produce_final_ranking,
)
from crowd_nav.reward_search.proxy_consistency import compute_proxy_consistency
from crowd_nav.reward_search.stage2 import (
    Stage2Config,
    Stage2Runner,
)
from crowd_nav.reward_search.stage3 import (
    STAGE3_PAPER_STEPS,
    STAGE3_STEPS,
    Stage3Config,
    Stage3Runner,
)

logger = logging.getLogger(__name__)


@dataclass
class EvoNavRunConfig:
    """End-to-end Algorithm 1 settings (paper defaults + practical overrides)."""

    output_dir: str = "results/evonav_run"
    seed: int = 425
    # Domain pack under crowd_nav.domains (default preserves Algorithm 1 baseline).
    domain: str = DEFAULT_DOMAIN
    llm_provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    llm_model: Optional[str] = None
    # Stage I Score1: "dataset" (default, paper) | "smoke" (opt-in fast tests only)
    score1_mode: str = "dataset"
    stage1_dataset_path: str = "data/stage1_dataset"

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
class EvoNavArtifacts:
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


class EvoNavPipeline:
    """Reproduce Algorithm 1 end-to-end and persist JSON artifacts."""

    def __init__(
        self,
        config: Optional[EvoNavRunConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        checkpoint_store: Optional[Any] = None,
        domain_pack: Optional[DomainPack] = None,
    ) -> None:
        self.config = config or EvoNavRunConfig()
        self.llm = llm
        self.validator = RewardValidator()
        # Optional paper-scale resume store (seed/stage/round/candidate).
        self.checkpoint_store = checkpoint_store
        self.domain_pack = domain_pack or load_domain(self.config.domain)

    def _build_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        if self.config.llm_model is None:
            return make_llm_client(self.config.llm_provider)
        return make_llm_client(self.config.llm_provider, model=self.config.llm_model)

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

    def run(self) -> EvoNavArtifacts:
        import time

        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        wall_t0 = time.perf_counter()
        from crowd_nav.reward_search.regime import (
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
        console.banner("EvoNav Algorithm 1")
        console.status(
            f"output={cfg.output_dir} seed={cfg.seed} llm={cfg.llm_provider} "
            f"fast={cfg.fast} device={cfg.device}"
        )
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

        llm = self._build_llm()
        pack = self.domain_pack
        if pack.name != str(cfg.domain).strip().lower():
            raise ValueError(
                f"Domain pack mismatch: config.domain={cfg.domain!r}, "
                f"pack.name={pack.name!r}"
            )
        seed_code = pack.seed_reward_source.strip() + "\n"

        manifest: Dict[str, Any] = {
            "algorithm": "EvoNav Algorithm 1",
            "domain": pack.name,
            "domain_display_name": pack.display_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(cfg),
            "stage3_paper_steps": STAGE3_PAPER_STEPS,
            "evolution_randomization_regime": regime,
            "predict_method": predict_method,
            "notes": (
                "Faithful replication baseline. No AMFRS novelty / archive / "
                "Pareto / adaptive controller. Obs-space choice (a): "
                "Stage II/III use predict_method=inferred when not --fast "
                "(AUDIT.md §8)."
            ),
        }
        console.status(f"domain={pack.name} ({pack.display_name})")
        write_json(os.path.join(cfg.output_dir, "config.json"), asdict(cfg))
        with open(os.path.join(cfg.output_dir, "seed_reward.py"), "w", encoding="utf-8") as f:
            f.write(seed_code)

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
        )
        evolver = StageIEvolver(
            llm,
            score_fn=self._score_fn(),
            validator=self.validator,
            config=s1_cfg,
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
            llm, s2_trainer, validator=self.validator, config=s2_cfg
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
            f"scalar={candidate_nav_scalar(best_s2):.3f} "
            f"R2_mode={r2_rank.get('mode')}",
            stage="pipeline",
        )

        # ----- Stage III -----
        # H-sweep only at H <= training crowd size (obs / Policy width).
        # Paper set {5,10,15,20} when human_num=20; with --human-num 5 → {5}.
        h_train = max(1, int(cfg.human_num))
        if cfg.stage3_run_h_sweep:
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
        s3_cfg = Stage3Config(
            population_size=len(stage2_pop),
            rounds=cfg.stage3_rounds,
            train_env_steps=cfg.stage3_train_steps,
            eval_episodes=cfg.stage3_eval_episodes,
            seed=cfg.seed,
            device=cfg.device,
            num_processes=cfg.num_processes,
            train_human_num=h_train,
            output_root=os.path.join(cfg.output_dir, "stage3_train"),
            human_counts=human_counts,
            randomization_regime=regime,
            predict_method=predict_method,
            env_name=env_name_for_predict_method(predict_method),
            protect_elite_refine=bool(cfg.elitism),
            inject_elite=bool(cfg.elitism),
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
            llm, s3_trainer, validator=self.validator, config=s3_cfg
        )
        if self.checkpoint_store is not None:
            s3_runner.checkpoint_store = self.checkpoint_store
            s3_runner.checkpoint_seed = int(cfg.seed)
        stage3_pop = s3_runner.run(stage2_pop, run_h_sweep=cfg.stage3_run_h_sweep)
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
            f"scalar={candidate_nav_scalar(best_s3):.3f} "
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

        # §4.3.4 proxy consistency (Spearman + top-k on genome fingerprints).
        consistency = compute_proxy_consistency(
            stage1=stage1_pop,
            stage2=list(s2_runner.trained_snapshots) or stage2_pop,
            stage3=list(s3_runner.trained_snapshots) or stage3_pop,
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
        console.status(
            f"proxy consistency ρ12={s12.get('rho')} "
            f"(n={s12.get('n_overlap')}) ρ23={s23.get('rho')} "
            f"(n={s23.get('n_overlap')})",
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
        )

        return EvoNavArtifacts(
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

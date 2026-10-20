"""
RAISE orchestrator (CrowdNav).

Closed-loop evolution (Score1 → Stage II PPO train+eval → fitness rank → LLM
breeding) for G generations, then Stage III long PPO on the final population.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
from raise_core.evo_selection import LLMRanker, SelectionConfig
from raise_core.evo_selection.consistency import append_consistency_csv
from raise_core.evo_selection.feedback import render_candidate_cards
from raise_core.evo_selection.guards import candidate_guard_flags
from raise_core.explore import OperatorCounts, operator_counts
from raise_core.feedback import CandidateArchive, FeedbackCardConfig
from raise_core.llm import LLMClient, make_llm_client
from raise_core.population_io import candidate_to_dict
from raise_core.raise_loop import ClosedLoopConfig, ClosedLoopRunner
from raise_core.json_io import read_json_dict, utc_now_z
from raise_core.raise_loop.logging_io import write_json
from raise_core.selection import (
    candidate_fitness,
    candidate_fitness_se,
    pick_best_trained,
    statistical_ties,
)
from raise_core.validate import (
    FINAL_EVAL_SEED_OFFSET,
    STAGE3_EVAL_SEED_OFFSET,
    STAGE3_STEPS,
    Stage3Config,
    Stage3Runner,
)


def _persist_run_config(output_dir: str, cfg: RaiseRunConfig) -> Dict[str, Any]:
    """Write ``config.json``; on resume keep the original and add a stamp file.

    Returns the on-disk config dict that should be treated as the run's
    canonical first-process config (used for ``manifest.config``).
    """
    path = os.path.join(output_dir, "config.json")
    payload = asdict(cfg)
    existing = read_json_dict(path) if bool(cfg.resume) else None
    if isinstance(existing, dict) and existing:
        stamp = utc_now_z().replace(":", "").replace("-", "")
        write_json(os.path.join(output_dir, f"config.resume_{stamp}.json"), payload)
        return existing
    write_json(path, payload)
    return payload


def _manifest_start(
    output_dir: str,
    *,
    pack: DomainPack,
    cfg: RaiseRunConfig,
    config_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the run manifest, preserving ``created_utc`` across resumes."""
    now = datetime.now(timezone.utc).isoformat()
    prior = read_json_dict(os.path.join(output_dir, "manifest.json"))
    resumes: List[Dict[str, Any]] = []
    created = now
    if bool(cfg.resume) and isinstance(prior, dict) and prior.get("created_utc"):
        created = str(prior["created_utc"])
        prior_resumes = prior.get("resumes")
        if isinstance(prior_resumes, list):
            resumes = [dict(r) for r in prior_resumes if isinstance(r, dict)]
        resumes.append({"utc": now, "argv": list(sys.argv)})
    return {
        "algorithm": "RAISE",
        "domain": pack.name,
        "domain_display_name": pack.display_name,
        "created_utc": created,
        "resumes": resumes,
        "config": config_payload,
    }


@dataclass
class RaiseRunConfig:
    """End-to-end run settings (CrowdNav defaults)."""

    output_dir: str = "results/raise_run"
    seed: int = 425
    domain: str = DEFAULT_DOMAIN
    llm_provider: str = "seed"  # seed | groq | vllm | ollama | scripted
    llm_model: Optional[str] = None
    # None → provider defaults (Groq: 3500; vLLM: 1500; Ollama: 4096).
    llm_max_tokens: Optional[int] = None
    # Groq gpt-oss only when set (e.g. low | medium | high); omitted otherwise.
    llm_reasoning_effort: Optional[str] = None
    # Stage I Score1: "dataset" (default) | "smoke" (fast tests only)
    score1_mode: str = "dataset"
    stage1_dataset_path: str = "domains/crowdnav/data/stage1_dataset"

    # Closed loop: population N, generations G, Stage II budget per candidate.
    population: int = 6
    generations: int = 4
    stage2_train_steps: int = 12_000
    stage2_eval_episodes: int = 30
    # Stage II selection uses the held-out val scenarios.
    stage2_eval_split: str = "val"

    # Stage III: long PPO on the final population.
    stage3_train_steps: int = STAGE3_STEPS
    stage3_eval_episodes: int = 30
    # Off: Stage III compares candidates at K3 steps each, and a warm start
    # would add the Stage II steps to some of them.
    stage3_warm_start: bool = False
    # Skip Stage-II collision-collapsed candidates in Stage III (default on).
    stage3_skip_collapsed: bool = True
    collapse_cr: float = 0.8
    # Independent PPO seeds per Stage III finalist (1 = current behaviour).
    stage3_seeds: int = 1
    # After collapse filter, keep only top-k by Stage II fitness (None = all).
    stage3_top_k: Optional[int] = None
    # Episodes of the blind re-evaluation of the Stage III winner (0 = skip).
    final_eval_episodes: int = 100
    # Also blind-eval every Stage III statistical tie (report only; never select).
    eval_tied_finalists: bool = True

    # Stub trainers (pytest / --fast).
    use_stub: bool = False
    # Carry the generation's best genome into the next generation.
    elitism: bool = True
    # Next-generation children per operator; None -> default split
    # (N=6 with elitism: 1 elite + 2 crossover + 2 mutation + 1 random).
    n_crossover: Optional[int] = None
    n_mutation: Optional[int] = None
    n_random: Optional[int] = None
    # Mutation parents: lower (bottom half) | rank_weighted | tournament.
    mutation_parents: str = "lower"
    # Drop Stage-II CR>=collapse_cr genomes from the mutation parent pool.
    mutation_exclude_collapsed: bool = False
    # Successive-halving Stage II (off by default).
    stage2_successive_halving: bool = False
    # Crash-safe resume of closed loop + Stage III from output_dir.
    resume: bool = True

    # Trainer knobs.
    n_envs: int = 1
    label_workers: int = 1
    # Off: a warm-started child carries its parent's PPO steps, so its fitness
    # would be compared against siblings trained on a smaller budget.
    warm_start: bool = False
    eval_mode: str = "holdout_only"
    # Robot / action space (domains.crowdnav.settings).
    action_mode: str = "continuous"
    robot_kinematics: str = "unicycle"
    action_speed_samples: int = 5
    action_rotation_samples: int = 16
    env_config_path: Optional[str] = None
    # Diagnostic feedback card + LLM input-token budget.
    feedback: FeedbackCardConfig = field(default_factory=FeedbackCardConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    device: str = "cuda"
    # Intra-op CPU threads for torch / OpenMP / MKL (0 = library default).
    # The PPO MLPs are tiny; uncapped, torch keeps every core busy for no gain.
    torch_threads: int = 1
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
        self.final_eval_episodes = 2
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


def limit_cpu_threads(n: int) -> None:
    """Cap torch / OpenMP / MKL intra-op threads to ``n`` (``n <= 0`` is a no-op).

    The env vars only take effect if torch is not imported yet; they are also
    inherited by spawned workers.
    """
    if n <= 0:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(n)


def _eval_tied_finalists(
    runner: Stage3Runner,
    *,
    stage3_pop: List[RewardCandidate],
    best: RewardCandidate,
    selection: Dict[str, Any],
    winner_final_eval: Optional[Dict[str, Any]],
    episodes: int,
    seed_base: int,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    """Blind-eval the Stage III winner and every statistical tie (report only).

    Selection already happened on val; these test numbers must never change
    ``best_stage3``. Returns ``None`` when disabled or episodes <= 0.
    """
    if not enabled or int(episodes) <= 0:
        return None
    tied = [str(x) for x in (selection.get("tied_ids") or [])]
    ids = [str(best.candidate_id)]
    for cid in tied:
        if cid and cid not in ids:
            ids.append(cid)
    by_id = {str(c.candidate_id): c for c in stage3_pop}
    evaluations: List[Dict[str, Any]] = []
    for cid in ids:
        if cid == str(best.candidate_id) and winner_final_eval is not None:
            row = {k: v for k, v in winner_final_eval.items() if k != "metrics"}
            row["role"] = "winner"
            row["metrics_summary"] = {
                k: (winner_final_eval.get("metrics") or {}).get(k)
                for k in ("SR", "CR", "TR", "fitness", "fitness_se")
            }
            evaluations.append(row)
            continue
        cand = by_id.get(cid)
        if cand is None:
            evaluations.append(
                {
                    "candidate_id": cid,
                    "role": "tie",
                    "status": "missing",
                    "error": "not in Stage III trained population",
                }
            )
            continue
        console.status(
            f"blind test eval of tied finalist {cid} ({episodes} episodes)",
            stage="pipeline",
        )
        row = runner.final_eval(cand, episodes=episodes, seed_base=seed_base)
        if row is None:
            evaluations.append(
                {
                    "candidate_id": cid,
                    "role": "tie",
                    "status": "skipped",
                    "error": "trainer cannot reevaluate",
                }
            )
            continue
        slim = {k: v for k, v in row.items() if k != "metrics"}
        slim["role"] = "tie" if cid != str(best.candidate_id) else "winner"
        slim["metrics_summary"] = {
            k: (row.get("metrics") or {}).get(k)
            for k in ("SR", "CR", "TR", "fitness", "fitness_se")
        }
        evaluations.append(slim)
    return {
        "note": (
            "Report only: blind test fitness of the Stage III winner and every "
            "statistical tie. Never used for selection."
        ),
        "seed_base": int(seed_base),
        "episodes": int(episodes),
        "winner_id": str(best.candidate_id),
        "tied_ids": tied,
        "evaluations": evaluations,
    }


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
        limit_cpu_threads(int(self.config.torch_threads))
        self.domain_pack = domain_pack or load_domain(self.config.domain)
        # Smoke states read the active action space (int vs tuple action).
        self.domain_pack.apply_config(self.config)
        self.validator = make_validator_for_domain(self.domain_pack)

    def _build_llm(self) -> LLMClient:
        if self.llm is not None:
            return self.llm
        seed = self.domain_pack.seed_reward_source
        kwargs: Dict[str, Any] = {
            "base_code": seed,
            "variant_fn": getattr(self.domain_pack.prompts, "seed_variant", None),
        }
        if self.config.llm_model is not None:
            kwargs["model"] = self.config.llm_model
        if self.config.llm_max_tokens is not None:
            kwargs["max_tokens"] = int(self.config.llm_max_tokens)
        if self.config.llm_reasoning_effort is not None:
            kwargs["reasoning_effort"] = str(self.config.llm_reasoning_effort)
        return make_llm_client(self.config.llm_provider, **kwargs)

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
            mutation_parents=str(cfg.mutation_parents),
            mutation_exclude_collapsed=bool(cfg.mutation_exclude_collapsed),
            collapse_cr=float(cfg.collapse_cr),
            resume=bool(cfg.resume),
            domain=str(self.domain_pack.name),
            eval_episodes=int(cfg.stage2_eval_episodes),
            stage2_eval_split=str(cfg.stage2_eval_split or "val"),
            stage2_successive_halving=bool(cfg.stage2_successive_halving),
            n_envs=max(1, int(cfg.n_envs or 1)),
            label_workers=max(1, int(cfg.label_workers or 1)),
            warm_start=bool(cfg.warm_start),
            eval_mode=str(cfg.eval_mode or "both"),
            feedback=cfg.feedback,
            selection=cfg.selection,
            **self._space_knobs(),
        )

    def _space_knobs(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "action_mode": str(cfg.action_mode or "discrete"),
            "robot_kinematics": str(cfg.robot_kinematics or "unicycle"),
            "action_speed_samples": int(cfg.action_speed_samples),
            "action_rotation_samples": int(cfg.action_rotation_samples),
            "env_config_path": cfg.env_config_path,
        }

    def _stage3_config(self) -> Stage3Config:
        cfg = self.config
        return Stage3Config(
            train_env_steps=int(cfg.stage3_train_steps),
            eval_episodes=int(cfg.stage3_eval_episodes),
            seed=int(cfg.seed),
            output_root=os.path.join(cfg.output_dir, "stage3_train"),
            device=str(cfg.device),
            resume=bool(cfg.resume),
            eval_mode=str(cfg.eval_mode or "both"),
            warm_start=bool(cfg.stage3_warm_start),
            n_envs=max(1, int(cfg.n_envs or 1)),
            eval_seed_base=int(cfg.seed) + STAGE3_EVAL_SEED_OFFSET,
            skip_collapsed=bool(cfg.stage3_skip_collapsed),
            collapse_cr=float(cfg.collapse_cr),
            n_seeds=max(1, int(cfg.stage3_seeds)),
            top_k=cfg.stage3_top_k,
            **self._space_knobs(),
        )

    def run(self) -> RaiseArtifacts:
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)
        wall_t0 = time.perf_counter()
        try:
            from raise_core.provenance import write_provenance

            write_provenance(cfg.output_dir, torch_threads=int(cfg.torch_threads))
        except Exception as exc:  # noqa: BLE001 - provenance must never abort a run
            console.warn(f"provenance write skipped ({type(exc).__name__}: {exc})")
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
        config_payload = _persist_run_config(cfg.output_dir, cfg)
        manifest = _manifest_start(
            cfg.output_dir, pack=pack, cfg=cfg, config_payload=config_payload
        )
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
        best_s2 = (
            stage2_pop[0]
            if cfg.selection.mode == "llm_rank" and stage2_pop
            else pick_best_trained(stage2_pop) or stage2_pop[0]
        )
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
        stage3_selection = s3_runner.selection_summary()
        stage3_rank = None
        if cfg.selection.mode == "llm_rank" and stage3_pop:
            stage3_archive = CandidateArchive()
            stage3_archive.add_population(
                stage3_pop,
                generation=max(0, int(cfg.generations) - 1),
            )
            stage3_cards = render_candidate_cards(
                stage3_pop,
                archive=stage3_archive,
                card_module=pack.feedback_card,
                config=cfg.feedback,
                population_note=" | ".join(
                    cl_result.manifest.get("selection_notes") or ()
                ),
            )
            stage3_fallback = sorted(
                stage3_pop, key=candidate_fitness, reverse=True
            )
            stage3_rank = LLMRanker(
                llm,
                cfg.selection,
                log_path=os.path.join(
                    cfg.output_dir, "closed_loop", "llm_selection.jsonl"
                ),
            ).rank(
                stage3_cards,
                fallback_order=[
                    candidate.candidate_id for candidate in stage3_fallback
                ],
                seed=int(cfg.selection.seed) + 1_000_000,
                stage="stage3",
                generation=max(0, int(cfg.generations) - 1),
                flags={
                    candidate.candidate_id: candidate_guard_flags(
                        candidate, cfg.selection
                    )
                    for candidate in stage3_pop
                },
            )
            stage3_by_id = {
                candidate.candidate_id: candidate for candidate in stage3_pop
            }
            stage3_pop = [
                stage3_by_id[candidate_id] for candidate_id in stage3_rank.ranking
            ]
            llm_selected = stage3_pop[0]
            fitness_best_id = stage3_selection.get("best_id")
            fitness_best = stage3_by_id.get(str(fitness_best_id))
            stage3_selection = {
                **stage3_selection,
                "fitness_best_id": fitness_best_id,
                "fitness_best_fitness": (
                    candidate_fitness(fitness_best) if fitness_best is not None else None
                ),
                "fitness_tied_ids": list(stage3_selection.get("tied_ids") or []),
                "best_id": stage3_rank.ranking[0],
                "fitness": candidate_fitness(llm_selected),
                "fitness_se": candidate_fitness_se(llm_selected),
                "tied_ids": statistical_ties(s3_runner.trained, llm_selected),
                "llm_ranking": stage3_rank.ranking,
                "llm_scores": stage3_rank.scores,
                "llm_weaknesses": stage3_rank.weaknesses,
                "llm_rationale": stage3_rank.rationale,
                "llm_fallback": stage3_rank.fallback,
                "llm_fallback_reason": stage3_rank.fallback_reason,
            }
            stage2_order = [
                candidate.candidate_id for candidate in cl_result.population
            ]
            stage3_selection["rank_consistency"] = append_consistency_csv(
                os.path.join(cfg.output_dir, "closed_loop", "rank_consistency.csv"),
                stage="stage3_vs_stage2",
                generation=max(0, int(cfg.generations) - 1),
                previous=stage2_order,
                current=stage3_rank.ranking,
                top_k=cfg.selection.top_k,
            )
        final_eval: Optional[Dict[str, Any]] = None
        if s3_runner.best_trained is None:
            console.warn(
                f"no Stage III candidate trained; falling back to the Stage II best "
                f"{best_s2.candidate_id} (its fitness is a Stage II score)",
                stage="pipeline",
            )
            best_s3 = best_s2
        else:
            best_s3 = (
                stage3_pop[0]
                if stage3_rank is not None
                else s3_runner.best_trained
            )
            final_eval = s3_runner.final_eval(
                best_s3,
                episodes=int(cfg.final_eval_episodes),
                seed_base=int(cfg.seed) + FINAL_EVAL_SEED_OFFSET,
            )
            if final_eval is not None:
                best_s3.metadata = {**(best_s3.metadata or {}), "final_eval": final_eval}
        tied_final_evals = _eval_tied_finalists(
            s3_runner,
            stage3_pop=stage3_pop,
            best=best_s3,
            selection=stage3_selection,
            winner_final_eval=final_eval,
            episodes=int(cfg.final_eval_episodes),
            seed_base=int(cfg.seed) + FINAL_EVAL_SEED_OFFSET,
            enabled=bool(cfg.eval_tied_finalists),
        )
        if tied_final_evals is not None:
            write_json(
                os.path.join(cfg.output_dir, "tied_final_evals.json"),
                tied_final_evals,
            )
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
        manifest["stage3_selection"] = stage3_selection
        # Full metrics (per-episode rows) stay on final_candidate.json.
        manifest["final_eval"] = (
            None
            if final_eval is None
            else {k: v for k, v in final_eval.items() if k != "metrics"}
        )
        manifest["tied_final_evals"] = (
            None
            if tied_final_evals is None
            else {
                "n": len(tied_final_evals.get("evaluations") or []),
                "note": tied_final_evals.get("note"),
            }
        )
        manifest["closed_loop_summary"] = dict(cl_result.manifest)
        write_json(os.path.join(cfg.output_dir, "manifest.json"), manifest)

        console.final_run_summary(
            output_dir=cfg.output_dir,
            wall_seconds=time.perf_counter() - wall_t0,
            best_stage1=best_s1,
            best_stage2=best_s2,
            best_stage3=best_s3,
        )

        if not bool(cfg.fast) and pack.post_run is not None:
            console.banner(f"{pack.display_name} plots + animation")
            art = pack.post_run(
                cfg.output_dir,
                episodes=2,
                seed=int(cfg.seed) + FINAL_EVAL_SEED_OFFSET,
            )
            manifest["post_run"] = (
                pack.post_run_summary(art) if pack.post_run_summary is not None else art
            )
            write_json(os.path.join(cfg.output_dir, "manifest.json"), manifest)
            n_plots = manifest["post_run"]["n_plots"]
            viz_ok = manifest["post_run"]["viz_ok"]
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

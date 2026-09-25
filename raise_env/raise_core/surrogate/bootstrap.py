"""
Bootstrap: build label population → Stage-II-short labels → fit surrogate.

CLI: ``scripts/bootstrap_surrogate.py``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from raise_core.explore import RewardCandidate
from raise_core.selection import navigation_scalar
from raise_core.surrogate import FEATURE_SCHEMA_VERSION
from raise_core.surrogate.dataset_io import (
    LABEL_SCHEMA_VERSION,
    append_example,
    existing_example_ids,
    load_table,
    read_manifest,
    write_manifest,
)
from raise_core.surrogate.features import (
    code_sha256,
    example_id_for_features,
    extract_candidate_features,
)
from raise_core.surrogate.model import SurrogateModel, default_model_dir

logger = logging.getLogger(__name__)

LABEL_BUDGET = "stage2_short"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _stage1_dataset_ready(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    npz = os.path.join(path, "stage1_dataset.npz")
    if os.path.isfile(npz):
        return True
    # Highway Stage I uses trajectories.jsonl (not CrowdNav npz).
    traj = os.path.join(path, "trajectories.jsonl")
    return os.path.isfile(traj) and os.path.getsize(traj) > 0


def _stage1_config_like_pipeline(n_candidates: int) -> "StageIConfig":
    """Match ``RaisePipeline`` Stage-I size split (Gen0 uses ``population_size`` only)."""
    from raise_core.explore import StageIConfig

    n = max(1, int(n_candidates))
    n_crossover = min(2, n)
    n_mutation = min(4, max(0, n - n_crossover))
    n_random = n - n_crossover - n_mutation
    return StageIConfig(
        population_size=n,
        generations=1,
        n_crossover=n_crossover,
        n_mutation=n_mutation,
        n_random=n_random,
        # Same Gen0 path as main run; larger N needs a proportionally larger regen budget.
        max_invalid_replacements=max(16, n * 2),
    )


def _build_population(
    *,
    n_candidates: int,
    llm_provider: str,
    seed: int,
    pack: Any = None,
) -> List[RewardCandidate]:
    """
    Build bootstrap candidates with the **exact** Stage-I Gen0 path used by
    ``RaisePipeline`` / ``StageIEvolver.initialize_population`` (same prompts,
    validator, batch+regen). Not a parallel prompt reimplementation.
    """
    del seed  # reserved for Stage-II / model fit; Gen0 LLM has its own sampling.
    from raise_core.domains import load_domain, make_validator_for_domain
    from raise_core.explore import StageIEvolver
    from raise_core.llm import make_llm_client
    from raise_core.scoring import make_smoke_score_fn

    if pack is None:
        pack = load_domain("crowdnav")
    llm = make_llm_client(
        llm_provider,
        base_code=getattr(pack, "seed_reward_source", None),
    )
    evolver = StageIEvolver(
        llm,
        score_fn=make_smoke_score_fn(),  # Gen0 only — Score1 not used here
        validator=make_validator_for_domain(pack),
        config=_stage1_config_like_pipeline(n_candidates),
        prompts=pack.prompts,
    )
    population = evolver.initialize_population()
    valid = [c for c in population if c.valid and c.reward_fn is not None]
    if len(valid) < 1:
        raise RuntimeError("bootstrap population is empty — Stage-I Gen0 produced no valid rewards")
    if len(valid) < n_candidates:
        logger.warning(
            "Stage-I Gen0 returned %d/%d valid candidates",
            len(valid),
            n_candidates,
        )
    return valid[:n_candidates]


def _labels_from_metrics(
    metrics: Any,
    *,
    example_id: str,
    env_steps: int,
    k2_unit: str,
    train_steps_config: int,
    eval_episodes: int,
    wall_seconds: float,
    seed: int,
    ok: bool,
) -> Dict[str, Any]:
    if hasattr(metrics, "_highway_extras") and not isinstance(metrics, dict):
        from domains.highway.adapter import metrics_to_highway_dict

        md = metrics_to_highway_dict(metrics)
    elif hasattr(metrics, "as_dict"):
        md = metrics.as_dict()
        extras = getattr(metrics, "_highway_extras", None)
        if isinstance(extras, dict):
            md = dict(md)
            md.update({k: float(v) for k, v in extras.items()})
    elif isinstance(metrics, dict):
        md = dict(metrics)
    else:
        md = {}
    sr = float(md.get("SR", md.get("sr", 0.0)))
    cr = float(md.get("CR", md.get("cr", 0.0)))
    tr = float(md.get("TR", md.get("tr", 0.0)))
    pl = float(md.get("PL", md.get("pl", 0.0)))
    itr = float(md.get("ITR", md.get("itr", 0.0)))
    mean_speed = float(md.get("mean_speed", itr))
    mean_progress = float(md.get("mean_progress", pl))
    soft_success = float(md.get("soft_success", 0.0))
    # CrowdNav log scalar stays SR−CR−0.5·TR; highway stores fitness (+ alias).
    classic = float(navigation_scalar(sr, cr, tr))
    if (
        str(md.get("domain", "")).lower() == "highway"
        or "soft_success" in md
        or "mean_speed" in md
        or md.get("fitness") is not None
        or md.get("selection_scalar") is not None
    ):
        from domains.highway.metrics import highway_fitness

        if md.get("fitness") is not None:
            selection = float(md["fitness"])
        elif md.get("selection_scalar") is not None:
            selection = float(md["selection_scalar"])
        else:
            selection = float(
                highway_fitness(
                    {
                        "SR": sr,
                        "CR": cr,
                        "TR": tr,
                        "PL": pl,
                        "mean_speed": mean_speed,
                        "mean_progress": mean_progress,
                        "soft_success": soft_success,
                    }
                )
            )
    else:
        selection = classic
    return {
        "example_id": example_id,
        "schema_version": LABEL_SCHEMA_VERSION,
        "SR": sr,
        "CR": cr,
        "TR": tr,
        "NT": float(md.get("NT", md.get("nt", 0.0))),
        "PL": pl,
        "ITR": itr,
        "SD": float(md.get("SD", md.get("sd", 0.0))),
        "mean_speed": mean_speed,
        "mean_progress": mean_progress,
        "soft_success": soft_success,
        "fitness": selection,
        "selection_scalar": selection,
        "scalar": classic,
        "env_steps": int(env_steps),
        "k2_unit": str(k2_unit),
        "train_steps_config": int(train_steps_config),
        "eval_episodes": int(eval_episodes),
        "wall_seconds": float(wall_seconds),
        "seed": int(seed),
        "ok": bool(ok),
        "label_budget": LABEL_BUDGET,
    }


def label_and_append_candidate(
    candidate: RewardCandidate,
    *,
    score_fn: Any,
    trainer: Any,
    stage2_cfg: Any,
    out_dir: str,
    round_index: int = 0,
    known_ids: Optional[set[str]] = None,
    label_rejected: bool = True,
    use_stub: bool = False,
    smoke_states: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Score1 → features → Stage II short → append jsonl.

    Shared by bootstrap and active-learning ``stage2_label`` acquire.
    Returns a result dict with ``status`` in {``ok``, ``skipped``, ``failed``}.
    """
    known = known_ids if known_ids is not None else existing_example_ids(out_dir)
    score1_result = score_fn(
        candidate.as_reward_function(), candidate_id=candidate.candidate_id
    )
    rejected = bool(getattr(score1_result, "rejected", False))
    if rejected and not label_rejected:
        return {"status": "skipped", "reason": "score1_rejected"}

    features = extract_candidate_features(
        candidate,
        score1_result=score1_result,
        label_budget=LABEL_BUDGET,
        smoke_states=smoke_states,
    )
    eid = example_id_for_features(features)
    if eid in known:
        return {"status": "skipped", "reason": "already_labeled", "example_id": eid}

    t0 = time.perf_counter()
    ok = True
    try:
        metrics = trainer.train_and_eval(
            candidate, round_index=round_index, config=stage2_cfg
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stage II label failed for %s: %s", candidate.candidate_id, exc)
        ok = False
        from raise_core.refine import ProxyMetrics

        metrics = ProxyMetrics()
    wall = time.perf_counter() - t0

    if not ok:
        # Do not poison the surrogate with all-zero fake metrics.
        return {
            "status": "failed",
            "reason": "train_and_eval_exception",
            "wall_seconds": wall,
        }

    labels = _labels_from_metrics(
        metrics,
        example_id=eid,
        env_steps=0 if use_stub else int(stage2_cfg.train_env_steps),
        k2_unit=str(getattr(stage2_cfg, "k2_unit", "gradient_steps")),
        train_steps_config=int(stage2_cfg.train_env_steps),
        eval_episodes=int(stage2_cfg.eval_episodes),
        wall_seconds=wall,
        seed=int(stage2_cfg.seed),
        ok=ok,
    )
    append_example(out_dir, features=features, labels=labels, example_id=eid)
    known.add(eid)
    # Keep Stage II metrics on the live candidate so R2 / proxy_consistency
    # / best_stage2 see SR/CR/TR (closed-loop has no separate Stage2Runner).
    md = dict(candidate.metadata or {})
    # Highway trainer already wrote rich last_metrics + fitness; keep them.
    if (
        md.get("fitness") is None and md.get("selection_scalar") is None
    ) or not isinstance(md.get("last_metrics"), dict):
        if hasattr(metrics, "_highway_extras"):
            from domains.highway.adapter import metrics_to_highway_dict

            rich = metrics_to_highway_dict(metrics)
            md["last_metrics"] = rich
            fit = float(rich.get("fitness", rich.get("selection_scalar", 0.0)))
            md["fitness"] = fit
            md["selection_scalar"] = fit
        else:
            md["last_metrics"] = metrics.as_dict()
    else:
        # Ensure alias keys stay in sync on already-labeled highway candidates.
        fit = md.get("fitness", md.get("selection_scalar"))
        if fit is not None:
            md["fitness"] = float(fit)
            md["selection_scalar"] = float(fit)
    md["last_stage2_ok"] = bool(ok)
    md["last_stage2_example_id"] = eid
    candidate.metadata = md
    return {
        "status": "ok" if ok else "failed",
        "example_id": eid,
        "features": features,
        "labels": labels,
        "wall_seconds": wall,
    }


def run_bootstrap(
    *,
    domain: str = "crowdnav",
    stage1_dataset_path: str = "domains/crowdnav/data/stage1_dataset",
    out_dir: str = "domains/crowdnav/data/surrogate_dataset",
    model_dir: str = "artifacts/surrogate",
    n_candidates: int = 60,
    stage2_train_steps: int = 8_000,
    k2_unit: str = "gradient_steps",
    use_stub: bool = False,
    seed: int = 425,
    force: bool = False,
    llm_provider: str = "seed",
    device: str = "cpu",
    label_rejected: bool = True,
    num_processes: int = 2,
    human_num: int = 5,
    predict_method: str = "inferred",
    randomization_regime: str = "with_random",
    horizon_steps: int = 100,
    eval_episodes: Optional[int] = None,
) -> dict[str, Any]:
    """
    End-to-end surrogate bootstrap (idempotent unless ``force``).

    ``domain`` selects the pack (``crowdnav`` default preserves thesis CrowdNav
    bootstrap; ``highway`` uses SB3 PPO + env_steps Score1 dataset).

    Defaults match the 12h closed-loop profile for CrowdNav: human_num=5,
    with_random, inferred GST, K2=8000 gradient steps.
    """
    if use_stub:
        n_candidates = min(int(n_candidates), 4)
        stage2_train_steps = min(int(stage2_train_steps), 64)

    from raise_core.surrogate.targets import filter_ok_examples, target_keys_for_domain

    domain_key = str(domain or "crowdnav").strip().lower() or "crowdnav"
    targets = list(target_keys_for_domain(domain_key))

    metrics_path = os.path.join(model_dir, "metrics.json")
    prior = read_manifest(out_dir)
    if (
        not force
        and os.path.isfile(metrics_path)
        and prior
        and str(prior.get("feature_schema_version")) == FEATURE_SCHEMA_VERSION
        and list(prior.get("target_keys") or []) == targets
    ):
        logger.info("bootstrap idempotent hit — returning existing artifacts")
        return {
            "status": "skipped_existing",
            "out_dir": out_dir,
            "model_dir": model_dir,
            "manifest": prior,
        }

    score1_mode = "smoke" if use_stub else "dataset"
    if score1_mode == "dataset" and not _stage1_dataset_ready(stage1_dataset_path):
        raise FileNotFoundError(
            f"Stage I dataset not found at {stage1_dataset_path!r}. "
            "Run the domain collect script or pass --fast for smoke/stub bootstrap."
        )

    from raise_core.domains import load_domain, make_stage2_trainer_for_domain
    from raise_core.refine import Stage2Config
    from raise_core.surrogate.features import set_behavior_smoke_states

    pack = load_domain(domain_key)
    if pack.smoke_states_fn is not None:
        set_behavior_smoke_states(pack.smoke_states_fn())
    else:
        set_behavior_smoke_states(None)

    score_fn, _dataset = pack.make_score_fn(
        mode=score1_mode,
        dataset_path=None if score1_mode == "smoke" else stage1_dataset_path,
    )
    trainer = make_stage2_trainer_for_domain(pack, use_stub=use_stub)
    smoke_states = pack.smoke_states_fn() if pack.smoke_states_fn is not None else None

    population = _build_population(
        n_candidates=n_candidates,
        llm_provider=llm_provider,
        seed=seed,
        pack=pack,
    )

    os.makedirs(out_dir, exist_ok=True)
    known_ids = existing_example_ids(out_dir) if not force else set()
    if force:
        for name in ("features.jsonl", "labels.jsonl"):
            path = os.path.join(out_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        known_ids = set()

    nproc = 1 if use_stub else max(1, int(num_processes))
    pred = str(predict_method)
    if eval_episodes is not None:
        n_eval = max(1, int(eval_episodes))
    else:
        n_eval = 8 if use_stub else (20 if domain_key == "highway" else 50)

    env_name = "highway-fast-v0"
    if domain_key == "crowdnav":
        from domains.crowdnav.regime import env_name_for_predict_method

        env_name = env_name_for_predict_method(pred)

    stage2_cfg = Stage2Config(
        train_env_steps=int(stage2_train_steps),
        k2_unit=str(k2_unit),
        eval_episodes=n_eval,
        horizon_steps=20 if use_stub else max(1, int(horizon_steps)),
        seed=int(seed),
        device=str(device),
        output_root=os.path.join(model_dir, "_stage2_runs"),
        num_processes=nproc,
        human_num=max(1, int(human_num)),
        predict_method=pred,
        randomization_regime=str(randomization_regime),
        env_name=env_name,
    )

    labeled = 0
    skipped = 0
    for round_index, candidate in enumerate(population):
        result = label_and_append_candidate(
            candidate,
            score_fn=score_fn,
            trainer=trainer,
            stage2_cfg=stage2_cfg,
            out_dir=out_dir,
            round_index=round_index,
            known_ids=known_ids,
            label_rejected=label_rejected,
            use_stub=use_stub,
            smoke_states=smoke_states,
        )
        status = result.get("status")
        eid = result.get("example_id")
        wall = float(result.get("wall_seconds") or 0.0)
        from raise_core import console

        console.status(
            f"bootstrap label {round_index + 1}/{len(population)} "
            f"{candidate.candidate_id} → {status}"
            + (f" id={eid}" if eid else "")
            + f" ({console.format_seconds(wall)})",
            stage="surrogate",
        )
        if result.get("status") == "ok" and result.get("example_id"):
            labeled += 1
        else:
            skipped += 1

    feats, labs = load_table(out_dir)
    if len(feats) < 1:
        raise RuntimeError("no surrogate examples on disk after bootstrap labeling")

    feats, labs = filter_ok_examples(feats, labs, target_keys=targets)
    if len(feats) < 1:
        raise RuntimeError("no ok=True surrogate examples after bootstrap labeling")
    model = SurrogateModel(random_seed=int(seed), n_bags=3 if use_stub else 5)
    fit_metrics = model.fit(feats, labs, target_keys=tuple(targets))
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(fit_metrics, fh, indent=2)
        fh.write("\n")

    manifest: Dict[str, Any] = {
        "n": len(feats),
        "domain": domain_key,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_budget": LABEL_BUDGET,
        "target_keys": list(targets),
        "n_candidates_requested": int(n_candidates),
        "n_labeled_this_run": int(labeled),
        "n_skipped": int(skipped),
        "use_stub": bool(use_stub),
        "score1_mode": score1_mode,
        "stage1_dataset_path": stage1_dataset_path if score1_mode == "dataset" else None,
        "stage2_train_steps": int(stage2_train_steps),
        "k2_unit": str(k2_unit),
        "eval_episodes": int(n_eval),
        "human_num": max(1, int(human_num)),
        "predict_method": str(predict_method),
        "randomization_regime": str(randomization_regime),
        "horizon_steps": int(stage2_cfg.horizon_steps),
        "llm_provider": str(llm_provider),
        "num_processes": int(nproc),
        "seed": int(seed),
        "created_at": _utc_now(),
        "model_dir": model_dir,
        "fit_metrics": fit_metrics,
    }
    write_manifest(out_dir, manifest)

    return {
        "status": "ok",
        "out_dir": out_dir,
        "model_dir": model_dir,
        "n": len(feats),
        "metrics": fit_metrics,
        "manifest": manifest,
        "domain": domain_key,
    }

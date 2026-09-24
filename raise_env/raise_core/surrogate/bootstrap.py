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
    return os.path.isfile(npz)


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
) -> List[RewardCandidate]:
    """
    Build bootstrap candidates with the **exact** Stage-I Gen0 path used by
    ``RaisePipeline`` / ``StageIEvolver.initialize_population`` (same prompts,
    validator, batch+regen). Not a parallel prompt reimplementation.
    """
    del seed  # reserved for Stage-II / model fit; Gen0 LLM has its own sampling.
    from raise_core.explore import StageIEvolver
    from raise_core.llm import make_llm_client
    from raise_core.scoring import make_smoke_score_fn

    llm = make_llm_client(llm_provider)
    evolver = StageIEvolver(
        llm,
        score_fn=make_smoke_score_fn(),  # Gen0 only — Score1 not used here
        config=_stage1_config_like_pipeline(n_candidates),
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
    if hasattr(metrics, "as_dict"):
        md = metrics.as_dict()
    elif isinstance(metrics, dict):
        md = dict(metrics)
    else:
        md = {}
    sr = float(md.get("SR", md.get("sr", 0.0)))
    cr = float(md.get("CR", md.get("cr", 0.0)))
    tr = float(md.get("TR", md.get("tr", 0.0)))
    return {
        "example_id": example_id,
        "schema_version": LABEL_SCHEMA_VERSION,
        "SR": sr,
        "CR": cr,
        "TR": tr,
        "NT": float(md.get("NT", md.get("nt", 0.0))),
        "PL": float(md.get("PL", md.get("pl", 0.0))),
        "ITR": float(md.get("ITR", md.get("itr", 0.0))),
        "SD": float(md.get("SD", md.get("sd", 0.0))),
        "scalar": float(navigation_scalar(sr, cr, tr)),
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
    md["last_metrics"] = metrics.as_dict()
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
    stage1_dataset_path: str = "data/stage1_dataset",
    out_dir: str = "data/surrogate_dataset",
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
) -> dict[str, Any]:
    """
    End-to-end surrogate bootstrap (idempotent unless ``force``).

    ``num_processes`` defaults to 1 — Windows + GST + CUDA with the Stage II
    default (up to 16 workers) can OOM the host / kill the IDE.

    Defaults match the 12h closed-loop profile: human_num=5, with_random,
    inferred GST, K2=8000 gradient steps.

    See ``PLAN.md`` §5.
    """
    if use_stub:
        n_candidates = min(int(n_candidates), 4)
        stage2_train_steps = min(int(stage2_train_steps), 64)

    metrics_path = os.path.join(model_dir, "metrics.json")
    prior = read_manifest(out_dir)
    if (
        not force
        and os.path.isfile(metrics_path)
        and prior
        and str(prior.get("feature_schema_version")) == FEATURE_SCHEMA_VERSION
        and list(prior.get("target_keys") or []) == ["SR", "CR", "TR"]
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
            "Run scripts/collect_stage1_dataset.py or pass --fast for smoke/stub bootstrap."
        )

    from raise_core.domains import load_domain, make_stage2_trainer_for_domain
    from raise_core.refine import Stage2Config
    from domains.crowdnav.regime import env_name_for_predict_method

    pack = load_domain("crowdnav")
    score_fn, _dataset = pack.make_score_fn(
        mode=score1_mode,
        dataset_path=None if score1_mode == "smoke" else stage1_dataset_path,
    )
    trainer = make_stage2_trainer_for_domain(pack, use_stub=use_stub)

    population = _build_population(
        n_candidates=n_candidates,
        llm_provider=llm_provider,
        seed=seed,
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
    stage2_cfg = Stage2Config(
        train_env_steps=int(stage2_train_steps),
        k2_unit=str(k2_unit),
        eval_episodes=8 if use_stub else 50,
        horizon_steps=20 if use_stub else max(1, int(horizon_steps)),
        seed=int(seed),
        device=str(device),
        output_root=os.path.join(model_dir, "_stage2_runs"),
        num_processes=nproc,
        human_num=max(1, int(human_num)),
        predict_method=pred,
        randomization_regime=str(randomization_regime),
        env_name=env_name_for_predict_method(pred),
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
        )
        if result.get("status") == "ok" or result.get("status") == "failed":
            # failed still appends a row; count as labeled attempt on disk
            if result.get("example_id"):
                labeled += 1
            else:
                skipped += 1
        else:
            skipped += 1

    feats, labs = load_table(out_dir)
    if len(feats) < 1:
        raise RuntimeError("no surrogate examples on disk after bootstrap labeling")

    model = SurrogateModel(random_seed=int(seed), n_bags=3 if use_stub else 5)
    fit_metrics = model.fit(feats, labs, target_keys=("SR", "CR", "TR"))
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(fit_metrics, fh, indent=2)
        fh.write("\n")

    manifest: Dict[str, Any] = {
        "n": len(feats),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "label_budget": LABEL_BUDGET,
        "target_keys": ["SR", "CR", "TR"],
        "n_candidates_requested": int(n_candidates),
        "n_labeled_this_run": int(labeled),
        "n_skipped": int(skipped),
        "use_stub": bool(use_stub),
        "score1_mode": score1_mode,
        "stage1_dataset_path": stage1_dataset_path if score1_mode == "dataset" else None,
        "stage2_train_steps": int(stage2_train_steps),
        "k2_unit": str(k2_unit),
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
    }

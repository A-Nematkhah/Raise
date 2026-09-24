# RAISE repository architecture

This document maps **where things live** after the organizational audit.
Scientific behavior (Score1, Stage II/III trainers, prompts, seeds) is unchanged.

## Repository layout

```text
Evonav/
├── README.md                 # Project entry
├── docs/
│   └── ARCHITECTURE.md       # This file
├── baselines_openai/         # Trimmed OpenAI Baselines (vec_env / logger)
└── raise_env/                # CrowdNav++ fork + RAISE + domain packs
    ├── scripts/              # CLI entry points (run / collect / report / plot)
    ├── crowd_nav/
    │   ├── domains/          # Environment packs (crowdnav, highway)
    │   ├── reward_search/    # RAISE core algorithm
    │   │   ├── explore.py    # Stage I (Score1 evolution)
    │   │   ├── refine.py     # Stage II (short RL + D.3)
    │   │   ├── validate.py   # Stage III (long RL + H-sweep)
    │   │   ├── pipeline.py   # Orchestrator (Alg.1 + --closed-loop)
    │   │   ├── raise_loop/   # Innovation multi-fidelity loop
    │   │   ├── surrogate/    # Stage-II proxy model + gate
    │   │   ├── active_learning/
    │   │   ├── sandbox/      # Reward AST validation
    │   │   └── tests/
    │   ├── configs/          # CrowdNav++ Config (simulator)
    │   └── policy/           # CrowdNav policies
    ├── crowd_sim/            # Gym envs (CrowdNav++)
    ├── rl/                   # A2C / PPO / vec_env (CrowdNav++ train stack)
    ├── gst_updated/          # Gumbel Social Transformer
    ├── data/                 # Datasets (gitignored contents)
    ├── artifacts/            # Fitted models (gitignored; keep fixtures)
    └── results/              # Experiment runs (gitignored)
```

## Two ways to run RAISE

| Mode | How | Code path |
|------|-----|-----------|
| **Paper Algorithm 1** | `python scripts/run_raise.py` (no `--closed-loop`) | `pipeline` → explore → refine → validate |
| **Innovation closed loop** | `--closed-loop` or `run_raise_12h.py` | `pipeline` → `raise_loop.ClosedLoopRunner` → optional Stage III |

On-disk innovation artifacts still use the subdirectory name `closed_loop/`
(resume-compatible). Public classes are `ClosedLoop*`; aliases `RaiseLoop*`
are identical objects.

## Domains

| Domain | Pack | Role |
|--------|------|------|
| `crowdnav` | `crowd_nav/domains/crowdnav/` | Paper baseline (GST / SRNN / Score1) — **frozen** |
| `highway` | `crowd_nav/domains/highway/` | Diagnostic HighwayEnv (`highway-fast-v0`) — additive |

Select with `--domain crowdnav|highway`. Do not put Highway assumptions into
CrowdNav Score1 / trainers; use pack factories.

## Artifacts vs source

| Path | Source-controlled? | Notes |
|------|--------------------|-------|
| `crowd_nav/`, `scripts/`, `rl/` | Yes | Code |
| `results/` | No | Per-run outputs |
| `artifacts/surr_warm/` | No | Warm surrogate (regenerate via `bootstrap_surrogate.py`) |
| `artifacts/surrogate_test_fix/` | Yes | Tiny test fixture |
| `data/stage1_dataset/` | No | Collect or copy locally |
| `data/highway_stage1_dataset/` | No | Highway collector |

## Primary scripts

| Script | Purpose |
|--------|---------|
| `run_raise.py` | Canonical CLI |
| `run_raise_12h.py` | Overnight closed-loop profile (+ `--warm-surrogate`) |
| `run_raise_1h.py` | Short closed-loop smoke |
| `bootstrap_surrogate.py` | Pre-fit Stage-II surrogate |
| `collect_stage1_dataset.py` | CrowdNav Score1 data |
| `collect_highway_stage1_dataset.py` | Highway Stage I data |
| `plot_raise_run.py` / `print_raise_report.py` | Analysis |

## Intentionally deferred (technical debt)

1. Renaming package `crowd_nav` → `raise` (import blast radius; CrowdNav freeze).
2. Moving CrowdNav `RealPolicyTrainer` bodies out of `refine.py` / `validate.py`
   into `domains/crowdnav/` (safe only with identical re-exports + freeze lift).
3. Renaming on-disk `closed_loop/` → `raise_loop/` (needs dual-read forever for old runs).
4. Unifying Stage II and Stage III trainer implementations.

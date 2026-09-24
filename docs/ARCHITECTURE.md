# RAISE repository architecture

This document describes the **actual** layout after the domain-isolation
restructure. Scientific behavior (Score1, trainers, prompts, seeds) is unchanged;
only package ownership and import paths moved.

## Top-level map (`raise_env/`)

```text
raise_env/
├── raise_core/                 # RAISE algorithm (environment-agnostic)
│   ├── explore / refine / validate / pipeline
│   ├── raise_loop / surrogate / active_learning / sandbox
│   ├── domains/                # DomainPack registry ONLY (not env code)
│   └── tests/                  # Core / innovation tests
│
├── domains/                    # First-class environment packs
│   ├── crowdnav/               # CrowdNav++-specific RAISE surfaces
│   │   ├── pack, prompts, adapter, explore_score, state, dataset, …
│   │   ├── trainers still façade over raise_core.refine/validate RealPolicyTrainer
│   │   └── tests/              # Score1 locks, regime, reward adapter, …
│   └── highway/                # HighwayEnv diagnostic pack
│       ├── pack, prompts, adapter, env_wrapper, stage1, state
│       └── tests/
│
├── crowd_nav/                  # CrowdNav++ simulator stack + COMPAT SHIMS
│   ├── configs/, policy/       # Original CrowdNav++ modules
│   ├── reward_search/          # Thin re-exports → raise_core / domains.crowdnav
│   └── domains/                # Thin re-exports → raise_core.domains / domains.*
│
├── crowd_sim/                  # Gym CrowdNav++ environments
├── rl/                         # A2C / PPO / vec_env train stack (CrowdNav++)
├── gst_updated/                # Gumbel Social Transformer
├── scripts/                    # Experiment / CLI entry points
├── data/  results/  artifacts/ # Generated outputs (gitignored)
└── docs → ../docs/ARCHITECTURE.md
```

## Ownership rules

| Concern | Lives in |
|---------|----------|
| Search / evolution / Score1 math / LLM / sandbox AST / surrogate / AL | `raise_core` |
| CrowdNav prompts, RewardState, Stage I dataset, GST regime, DS-RNN | `domains.crowdnav` |
| Highway prompts, HighwayRewardState, SB3 trainers, env wrapper | `domains.highway` |
| Gym env + ORCA/SRNN policies + Config | `crowd_sim` / `crowd_nav.policy` / `crowd_nav.configs` / `rl` |
| CLI experiments | `scripts/` |
| Frozen import paths for Score1 / prompt regression | `crowd_nav.reward_search.*` shims |

## How RAISE talks to an environment

```text
scripts/run_raise.py --domain <name>
  → raise_core.domains.load_domain(name)
  → DomainPack  (prompts, make_score_fn, trainers, smoke_states)
  → raise_core.pipeline.RaisePipeline
```

Adding a third environment: create `domains/<name>/` with `get_pack()`, register
in `raise_core.domains._REGISTRY`, add tests under `domains/<name>/tests/`.

## Compatibility shims (CrowdNav freeze)

Imports of the form `crowd_nav.reward_search.*` and `crowd_nav.domains.*` still
work. They re-export the new packages so Score1 baseline locks and
`crowd_sim` wiring do not need simultaneous rewrites of every caller.

Prefer new imports in new code:

- `from raise_core.pipeline import RaisePipeline`
- `from domains.crowdnav.prompts import D5_SEED_FUNCTION`
- `from domains.highway.pack import get_pack`

## Experiments

| Goal | Entry |
|------|-------|
| Alg.1 (paper) | `python scripts/run_raise.py` |
| Closed loop | `python scripts/run_raise.py --closed-loop` or `run_raise_12h.py` |
| Highway smoke | `python scripts/run_raise.py --domain highway --fast` |
| Warm surrogate | `python scripts/bootstrap_surrogate.py` |

## Intentionally deferred

1. Moving `RealPolicyTrainer` bodies from `raise_core.refine` / `validate` into
   `domains.crowdnav.trainers_*` (façade already exists; extraction is mechanical).
2. Splitting `RewardFunction` ABC into `raise_core.state` vs CrowdNav dataclasses
   (currently co-located in `domains.crowdnav.state` for freeze safety).
3. Deleting compatibility shims after all callers are retargeted.

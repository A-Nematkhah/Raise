# RAISE repository architecture

This document describes the **actual** layout after per-domain isolation.
Scientific behavior (Score1, trainers, prompts, seeds) is unchanged; only
package ownership and on-disk paths moved.

## Top-level map (`raise_env/`)

```text
raise_env/
├── raise_core/                 # RAISE algorithm (environment-agnostic)
├── raise_paths.py              # sys.path bootstrap for domain runtimes
├── domains/
│   ├── crowdnav/               # CrowdNav++ domain (frozen baseline)
│   │   ├── pack, prompts, adapter, explore_score, state, dataset, …
│   │   ├── tests/              # Score1 locks, regime, reward adapter, …
│   │   ├── data/               # stage1 / AL / surrogate datasets
│   │   └── runtime/            # importable env stack (on sys.path)
│   │       ├── crowd_sim/      # Gym environments
│   │       ├── crowd_nav/      # configs, policy, compat shims
│   │       ├── rl/             # A2C / PPO / vec_env
│   │       └── gst_updated/    # Gumbel Social Transformer
│   └── highway/                # HighwayEnv diagnostic pack
│       ├── pack, prompts, adapter, env_wrapper, stage1, state
│       ├── tests/
│       └── data/               # highway Stage I datasets
├── scripts/                    # Experiment / CLI entry points
├── configs/                    # Shared experiment YAML (e.g. paper_scale)
├── results/  artifacts/        # Generated outputs (gitignored)
└── docs → ../docs/ARCHITECTURE.md
```

## Ownership rules

| Concern | Lives in |
|---------|----------|
| Search / evolution / Score1 math / LLM / sandbox AST / surrogate / AL | `raise_core` |
| CrowdNav prompts, RewardState, Stage I dataset, GST regime, DS-RNN | `domains.crowdnav` |
| CrowdNav simulator + policies + GST + RL stack | `domains/crowdnav/runtime/*` |
| CrowdNav datasets | `domains/crowdnav/data/` |
| Highway prompts, wrapper, SB3 trainers | `domains.highway` |
| Highway datasets | `domains/highway/data/` |
| CLI experiments | `scripts/` |
| Frozen import paths (`crowd_nav.reward_search.*`) | shims under `runtime/crowd_nav/` |

## How RAISE talks to an environment

```text
scripts/run_raise.py --domain <name>
  → raise_paths / domains/<name> arm runtime on sys.path
  → raise_core.domains.load_domain(name)
  → DomainPack  (prompts, make_score_fn, trainers, smoke_states)
  → raise_core.pipeline.RaisePipeline
```

## Compatibility

`import crowd_sim`, `import crowd_nav`, and `import rl` still work: pytest and
scripts put `domains/crowdnav/runtime` on `sys.path`. Prefer new imports in new
code:

- `from raise_core.pipeline import RaisePipeline`
- `from domains.crowdnav.prompts import D5_SEED_FUNCTION`
- `from domains.highway.pack import get_pack`

## Default dataset paths

| Domain | Stage I default |
|--------|-----------------|
| crowdnav | `domains/crowdnav/data/stage1_dataset` |
| highway | `domains/highway/data/stage1_dataset` |

GST checkpoints: `domains/crowdnav/runtime/gst_updated/results/...`

## Intentionally deferred

1. Moving `RealPolicyTrainer` bodies from `raise_core.refine` / `validate` into
   `domains.crowdnav` (façade already exists).
2. Deleting `crowd_nav.reward_search.*` shims after all callers are retargeted.

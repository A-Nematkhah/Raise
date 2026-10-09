# RAISE repository architecture

## Top-level map (`raise_env/`)

```text
raise_env/
├── raise_core/                 # RAISE algorithm
│   ├── llm.py                  # LLM clients (groq / ollama / vllm / seed / scripted)
│   ├── sandbox/                # AST policy, spawned smoke test, compile
│   ├── explore.py              # Gen0 proposal, Score1 scoring, crossover / mutation
│   ├── raise_loop/             # closed-loop runner, checkpoints, epoch logs, report
│   ├── refine.py               # Stage II config / metrics / stub trainer
│   ├── validate.py             # Stage III runner (long PPO, best by fitness)
│   ├── pipeline.py             # RaisePipeline: closed loop → Stage III → artifacts
│   └── domains/                # DomainPack + registry / factories
├── domains/
│   └── highway/                # HighwayEnv pack (see domains/highway/spec.md)
│       ├── pack, prompts, adapter, env_wrapper, stage1, state, metrics
│       ├── tests/
│       └── data/               # highway Stage I dataset
├── scripts/                    # CLI entry points
└── results/                    # generated outputs (gitignored)
```

## Ownership rules

| Concern | Lives in |
|---------|----------|
| Evolution loop / Score1 math / LLM / sandbox / checkpoints | `raise_core` |
| Highway prompts, reward state, env wrapper, SB3 PPO trainer, fitness | `domains.highway` |
| Highway Stage I dataset | `domains/highway/data/` |
| CLI experiments | `scripts/` |

## Execution flow

```text
scripts/run_raise.py
  → RaiseRunConfig → raise_core.pipeline.RaisePipeline
  → load_domain("highway") → DomainPack (prompts, Score1, trainers, smoke states)
  → ClosedLoopRunner: Gen0 → for each generation:
        Score1 → PPO train + eval new candidates → rank by fitness
        → LLM crossover / mutation / random (+ elite) → checkpoint
  → Stage3Runner: long PPO per final candidate → best by fitness
  → JSON artifacts + plots
```

# RAISE repository architecture

## Top-level map (`raise_env/`)

```text
raise_env/
├── raise_core/                 # RAISE algorithm
│   ├── llm.py                  # LLM clients (groq / ollama / vllm / seed / scripted)
│   ├── sandbox/                # AST policy, spawned smoke test, compile
│   ├── explore.py              # Gen0 proposal, Score1 scoring, crossover / mutation
│   ├── feedback.py             # feedback-card config, candidate archive, lineage, code diff
│   ├── token_budget.py         # LLM input-token counting + budget fitting
│   ├── llm_call_log.py         # llm_calls.jsonl + rendered prompt snapshots
│   ├── raise_loop/             # closed-loop runner, checkpoints, epoch logs, report
│   ├── refine.py               # Stage II config / metrics / stub trainer
│   ├── validate.py             # Stage III runner (long PPO, best by fitness)
│   ├── pipeline.py             # RaisePipeline: closed loop → Stage III → artifacts
│   └── domains/                # DomainPack + registry / factories
├── domains/
│   └── crowdnav/               # CrowdNav++ pack (see domains/crowdnav/spec.md)
│       ├── pack, prompts, adapter, env_wrapper, stage1, state, metrics, feedback_card, viz
│       ├── tests/
│       └── data/               # CrowdNav Stage I dataset (generated, gitignored)
├── third_party/crowdnavpp/     # CrowdNav++ CrowdSimVarNum @ 3907731 (unmodified, MIT)
├── scripts/                    # CLI entry points
└── results/                    # generated outputs (gitignored)
```

## Ownership rules

| Concern | Lives in |
|---------|----------|
| Evolution loop / Score1 math / LLM / sandbox / checkpoints | `raise_core` |
| CrowdNav++ prompts, reward state, env wrapper, SB3 PPO trainer, fitness, viz | `domains.crowdnav` |
| Simulator (variable humans, ORCA, collision / goal / timeout) | `third_party/crowdnavpp/crowd_sim` |
| CrowdNav Stage I dataset | `domains/crowdnav/data/` |
| CLI experiments | `scripts/` |

## Execution flow

```text
scripts/run_raise.py
  → RaiseRunConfig → raise_core.pipeline.RaisePipeline
  → load_domain("crowdnav") → CrowdNav++ DomainPack (prompts, Score1, trainers, smoke states)
  → ClosedLoopRunner: Gen0 → for each generation:
        Score1 → PPO train (train cases) + eval new candidates (val cases) → rank by fitness
        → archive → LLM crossover / mutation / random (+ elite), each prompt
          with a feedback card fitted to the input-token budget → checkpoint
  → Stage3Runner: long PPO per final candidate → best by fitness
  → final blind eval of the winner on upstream test cases
  → JSON artifacts + plots + policy animation
```

# raise_env

RAISE experiment tree: LLM-guided reward evolution on HighwayEnv.

| Area | Location |
|------|----------|
| RAISE core | `raise_core/` |
| Highway domain | `domains/highway/` (spec: [`domains/highway/spec.md`](domains/highway/spec.md)) |
| CLIs | `scripts/` |

```bash
python scripts/run_raise.py --fast                       # stub wiring smoke (seconds)
python scripts/run_raise.py --llm groq                   # full run (N=6, G=4, K2=12k, K3=70k)
python scripts/run_raise.py --llm seed --allow-seed-llm  # real PPO, no LLM cost
```

A non-`--fast` run with the default `--llm seed` (no real LLM) is refused
unless you pass `--allow-seed-llm` (debug only). Run budgets are CLI flags
(`--population`, `--generations`, `--stage2-train-steps`,
`--stage3-train-steps`, …); defaults live in `raise_core/pipeline.py`
(`RaiseRunConfig`). Install, keys and run details: [`README_RAISE.md`](README_RAISE.md).

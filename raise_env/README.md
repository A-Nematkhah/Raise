# raise_env

RAISE experiment tree: LLM-guided reward evolution on the variable-human
CrowdNav++ simulator (`CrowdSimVarNum`).

| Area | Location |
|------|----------|
| RAISE core | `raise_core/` |
| CrowdNav++ domain | `domains/crowdnav/` (spec: [`domains/crowdnav/spec.md`](domains/crowdnav/spec.md)) |
| Active simulator | `third_party/crowdnavpp/` ([`UPSTREAM.md`](third_party/crowdnavpp/UPSTREAM.md)) |
| CLIs | `scripts/` |

```bash
python scripts/run_raise.py --fast                       # stub wiring smoke (seconds)
python scripts/run_raise.py --llm groq                   # full run (N=6, G=4, K2=12k, K3=70k)
python scripts/run_raise.py --llm seed --allow-seed-llm  # real PPO, no LLM cost
```

A non-`--fast` run with the default `--llm seed` (no real LLM) is refused
unless you pass `--allow-seed-llm` (debug only). Run budgets are CLI flags
(`--population`, `--generations`, `--stage2-train-steps`,
`--set budget.stage3.train_steps=…`, …); defaults live in `raise_config.py`
(`RaiseRunConfig`). Install (including Python-RVO2), keys and run details:
[`README_RAISE.md`](README_RAISE.md).

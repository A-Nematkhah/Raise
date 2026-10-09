# RAISE — LLM-guided reward evolution on HighwayEnv

This repository implements **RAISE**: an LLM proposes reward functions for
`highway-fast-v0`, each candidate is trained with PPO and evaluated, and the
LLM breeds the next generation from the evaluation feedback. The active tree
is `raise_env/`.

| Area | Location |
|------|----------|
| **RAISE core** (LLM, sandbox, evolution loop, Stage III) | `raise_env/raise_core/` |
| **HighwayEnv domain** (env wrapper, prompts, PPO trainer, Score1) | `raise_env/domains/highway/` |
| **CLI** | `raise_env/scripts/` |
| **Results** | `raise_env/results/` (gitignored) |

Architecture map: **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**

## Quick start

```bash
cd raise_env
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_raise.py --fast --output-dir results/raise_fast   # seconds, stub trainers
pytest -m "not slow"

python scripts/collect_highway_stage1_dataset.py                    # once, for Score1
python scripts/run_raise.py --llm groq --output-dir results/highway_run
```

`raise_env` started as a derivative of
[CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
(MIT — see `raise_env/LICENSE` and `raise_env/NOTICE.md`).

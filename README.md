# RAISE — LLM-guided reward evolution on CrowdNav

This repository implements **RAISE**: an LLM proposes reward functions for
crowd-aware robot navigation in the [vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav)
simulator, each candidate is trained with PPO and evaluated, and the LLM
breeds the next generation from the evaluation feedback. The active tree is
`raise_env/`.

| Area | Location |
|------|----------|
| **RAISE core** (LLM, sandbox, evolution loop, Stage III) | `raise_env/raise_core/` |
| **CrowdNav domain** (env wrapper, prompts, PPO trainer, Score1) | `raise_env/domains/crowdnav/` |
| **Vendored simulator** (`crowd_sim`, unmodified, MIT) | `raise_env/third_party/crowdnav/` |
| **CLI** | `raise_env/scripts/` |
| **Results** | `raise_env/results/` (gitignored) |

Architecture map: **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**

## Quick start

```bash
cd raise_env
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install Cython==0.29.36 cmake
pip install --no-build-isolation "pyrvo2 @ git+https://github.com/sybrenstuvel/Python-RVO2.git@c2c46ba8d59556aa10faf03479293236efea154d"

python scripts/run_raise.py --fast --output-dir results/raise_fast   # seconds, stub trainers
pytest -m "not slow"

python scripts/collect_crowdnav_stage1_dataset.py                   # once, for Score1
python scripts/run_raise.py --llm groq --output-dir results/crowdnav_run
```

`raise_env` vendors `crowd_sim` from vita-epfl/CrowdNav (MIT) and started as
a derivative of
[CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
(MIT). See `raise_env/LICENSE`, `raise_env/NOTICE.md` and
`raise_env/third_party/crowdnav/UPSTREAM.md`.

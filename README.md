# RAISE — adaptive / evolutionary reward search

This repository implements **RAISE**: LLM-guided reward-function search with
RL evaluation. The active tree is `raise_env/`.

| Area | Location |
|------|----------|
| **RAISE core** (search / evolution) | `raise_env/raise_core/` |
| **CrowdNav++ domain** | `raise_env/domains/crowdnav/` (+ `crowd_sim/`, `rl/`, `gst_updated/`) |
| **HighwayEnv domain** | `raise_env/domains/highway/` |
| **Experiments / CLI** | `raise_env/scripts/` |
| **Results / artifacts** | `raise_env/results/`, `raise_env/artifacts/` (gitignored) |

Architecture map: **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**

| Mode | Command |
|------|---------|
| Algorithm 1 (paper) | `cd raise_env && python scripts/run_raise.py` |
| Closed loop | `python scripts/run_raise.py --closed-loop` or `run_raise_12h.py` |
| Highway diagnostic | `python scripts/run_raise.py --domain highway --fast` |

`raise_env` is a **derivative work** of
[CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
(MIT — see `raise_env/LICENSE` and `raise_env/NOTICE.md`).

## Quick start

```bash
cd raise_env
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements_pinned.txt
pip install -e ../baselines_openai --no-build-isolation

python scripts/run_raise.py --fast --output-dir results/raise_fast
pytest -m "not slow"
```

Highway extras: `pip install -r requirements_highway.txt`

Compatibility: historical imports `crowd_nav.reward_search.*` still work via thin shims.
New code should import `raise_core` and `domains.*`.

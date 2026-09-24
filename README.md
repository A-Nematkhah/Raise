# RAISE — adaptive / evolutionary reward search

This repository implements **RAISE**: LLM-guided reward-function search with
RL evaluation. The active tree is `raise_env/` (a CrowdNav++ derivative).

| Mode | Meaning |
|------|---------|
| **Algorithm 1** | Linear Explore → Refine → Validate (paper baseline) |
| **Closed loop** | Score1 ↔ Stage-II short ↔ Surrogate + AL across generations |

**Environments (domain packs):**

| `--domain` | Role |
|------------|------|
| `crowdnav` | CrowdNav++ / GST / SRNN — paper claims (**frozen** baseline) |
| `highway` | HighwayEnv diagnostic (`highway-fast-v0`) — algorithm convergence only |

| Directory | Role |
|-----------|------|
| `raise_env/` | Simulator + RAISE (`crowd_nav/reward_search`) + domain packs |
| `baselines_openai/` | Trimmed OpenAI Baselines (vec_env / logger / bench) |
| `docs/ARCHITECTURE.md` | Layout map, entry points, artifact policy |

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
# Install PyTorch (pinned) and Python-RVO2 per raise_env/README.md

# Fast wiring test (~seconds) — CrowdNav
python scripts/run_raise.py --fast --output-dir results/raise_fast

# Highway diagnostic (install extras first)
pip install -r requirements_highway.txt
python scripts/run_raise.py --domain highway --fast --output-dir results/highway_fast

# Tests
pytest crowd_nav/reward_search/tests -m "not slow"
```

More detail:

- RAISE runs, paper-scale, API keys → **`raise_env/README_RAISE.md`**
- Architecture map → **`docs/ARCHITECTURE.md`**
- Simulator train/test → **`raise_env/README.md`**
- Domain packs → **`raise_env/crowd_nav/domains/README.md`**

## Groq API keys

Copy `raise_env/groq_keys.json.example` → `raise_env/groq_keys.json` (gitignored).
Never commit real keys.

## Citations

```bibtex
@article{raise2026,
  title   = {RAISE},
  eprint  = {arXiv:2605.11859},
  year    = {2026}
}

@inproceedings{liu2023crowdnavpp,
  title     = {Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author    = {Liu, Shuijing and Chang, Peixin and Huang, Zhe and others},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  pages     = {12015--12021}
}
```

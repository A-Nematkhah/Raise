# RAISE — reward search on CrowdNav++

This repository implements **RAISE**: an LLM-guided reward-function search
pipeline for robot crowd navigation on a CrowdNav++ fork (`raise_env/`).

It can reproduce the linear **RAISE Algorithm 1** baseline (explore → refine →
validate) and also run the **RAISE closed loop** (Score1 + Surrogate + Active
Learning across generations). **AMFRS** is not included.

| Directory | Role |
|-----------|------|
| `raise_env/` | Simulator fork + `crowd_nav/reward_search` (RAISE Alg. 1) |
| `baselines_openai/` | Trimmed OpenAI Baselines (vec_env / logger / bench only) |

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

# Fast wiring test (~seconds)
python scripts/run_raise.py --fast --output-dir results/raise_fast

# Tests
pytest crowd_nav/reward_search/tests -m "not slow"
```

More detail:

- RAISE / Alg. 1 runs, paper-scale, API keys → **`raise_env/README_RAISE.md`**
- Simulator train/test → **`raise_env/README.md`**
- Architecture notes → **`raise_env/AUDIT.md`**
- Domain packs → **`raise_env/crowd_nav/domains/README.md`**
- Surrogate / AL / raise-loop plans → under `raise_env/crowd_nav/reward_search/`

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

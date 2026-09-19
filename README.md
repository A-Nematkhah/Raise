# EvoNav faithful replication (CrowdNav++ baseline)

**This public release contains only the EvoNav Algorithm 1 baseline** — a
faithful replication on the CrowdNav++ simulator (`evonav_env/`), including
`crowd_nav/reward_search/`.

**AMFRS** (the thesis's novel multi-objective evolution contribution) is
**not included** in this repository.

| Directory | In this release? | Role |
|-----------|------------------|------|
| `evonav_env/` | **Yes** | EvoNav baseline on CrowdNav++ fork |
| `baselines_openai/` | **Yes** | Trimmed OpenAI Baselines (vec_env / logger / bench only) |

`evonav_env` is a **derivative work** of
[CrowdNav_Prediction_AttnGraph](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
(MIT — see `evonav_env/LICENSE` and `evonav_env/NOTICE.md`).

## Quick start

```bash
cd evonav_env
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements_pinned.txt
pip install -e ../baselines_openai --no-build-isolation
# Install PyTorch (pinned) and Python-RVO2 per evonav_env/README.md

# Fast wiring test (~seconds)
python scripts/run_evonav.py --fast --output-dir results/evonav_fast

# Tests
pytest crowd_nav/reward_search/tests -m "not slow"
```

See **`evonav_env/README_EVONAV.md`** for Algorithm 1, paper-scale runs, and API keys.
Simulator train/test docs: **`evonav_env/README.md`**. Architecture notes: **`evonav_env/AUDIT.md`**.
Domain packs (pluggable env backends): **`evonav_env/crowd_nav/domains/README.md`**.

## Groq API keys

Copy `evonav_env/groq_keys.json.example` → `evonav_env/groq_keys.json` (gitignored). Never commit real keys.

## Citations

```bibtex
@article{evonav2026,
  title   = {EvoNav},
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

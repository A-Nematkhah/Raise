# RAISE (this fork)

Extension of CrowdNav++ for **RAISE** reward search. Compatible with RAISE
Algorithm 1 (arXiv:2605.11859); AMFRS mechanisms are not included.

Upstream simulator docs: `README.md` in this directory. Architecture audit: `AUDIT.md`.

## Install

```bash
cd raise_env
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements_pinned.txt
.venv\Scripts\python.exe -m pip install git+https://github.com/sybrenstuvel/Python-RVO2.git
.venv\Scripts\python.exe -m pip install -e ../baselines_openai --no-build-isolation
```

Use this single `.venv` for the project. Do not mix it with system Python or
another environment, and do not use a `PYTHONPATH` workaround.

The pinned GPU wheel is `torch==2.11.0+cu128`. Adjust the `cu1xx` suffix and
the official PyTorch index URL in `requirements_pinned.txt` to match the local
CUDA driver before installing on another machine.

## Ollama (local, no API key)

For predictable latency, use the non-thinking model variant:

```bash
ollama pull frob/qwen3.5-instruct:4b
```

The default Ollama model is the reasoning model `qwen3.5:4b`, which can emit
`<think>` content before the code fence. If using it explicitly:

```bash
ollama pull qwen3.5:4b
python scripts/run_raise.py --llm ollama --llm-model qwen3.5:4b --output-dir results/ollama_run
```

The server URL defaults to `http://localhost:11434/v1`; override it with
`OLLAMA_BASE_URL`. If Gen0 rejection rates are unexpectedly high, inspect
`results/.../gen0_rejections.jsonl` for the truncation-specific
`no closing code fence found` error before assuming a sandbox problem.

## API keys (Groq)

1. Copy `groq_keys.json.example` → `groq_keys.json`
2. Add keys: `{"keys": ["gsk_...", "..."]}`
3. Or set `GROQ_API_KEY` for a single key (pool disabled)

## Run matrix

| Goal | Command | Hardware | Time |
|------|---------|----------|------|
| Wiring smoke | `python scripts/run_raise.py --fast` | CPU | seconds |
| Stage I dataset (M=100) | `python scripts/collect_stage1_dataset.py --regime without_random` | CPU | ~tens of min |
| Local validation | `python scripts/run_raise.py --llm groq --device cuda --regime without_random --stage1-dataset data/stage1_dataset --stage3-train-steps 500000` | GPU + Groq | hours |
| Paper scale | `python scripts/run_raise_paper_scale.py --device cuda --llm groq` | GPU + Groq | days (K3=1e7 × seeds) |

Defaults (AUDIT.md §8): `without_random`, Stage II/III `predict_method=inferred`, GST `...-seed_1000/sj`.

## Domain packs

Reward search is wired through a **domain pack** (default `crowdnav`). Prompts,
Stage I Score1, and Stage II/III trainers are resolved via
`crowd_nav.domains` so another environment can be added later as a sibling
folder without rewriting the pipeline.

- Guide: [`crowd_nav/domains/README.md`](crowd_nav/domains/README.md)
- CLI: `python scripts/run_raise.py --domain crowdnav ...`

Do not add stub domains; only register a pack when the real env is ready.

## Stage I dataset

Collect once (paper: M=100, N_traj=10). Our behavior mix is **not** claimed as
paper text — see `scripts/collect_stage1_dataset.py` and
`crowd_nav/domains/README.md`.

After the fidelity collector fix (diverse ORCA/SF/noise/random), **recollect**
before new Stage I science runs:

```bash
python scripts/collect_stage1_dataset.py --out data/stage1_dataset
```

Older archives may have ~5 unique trajs / 10 (duplicate deterministic ORCA/SF).

## Tests

```bash
pytest crowd_nav/reward_search/tests -m "not slow"   # CI default
pytest crowd_nav/reward_search/tests -m slow         # 1 real-env collect test
```

## Stage III H-sweep (Table 6)

After Stage III training, the best-ever candidate is re-evaluated at each
human count in `{5, 10, 15, 20}` (clipped to `H ≤ train human_num`). **Cost:**
each H value is a **full extra eval pass** of `E3` episodes (default 500) —
not a cheap metric reuse. Disable with `--no-h-sweep` when iterating locally.
Paper-scale YAML keeps `human_counts: [5, 10, 15, 20]`.

## Baseline checkpoints

Pretrained ORCA/SF/GST under `trained_models/` (see `scripts/report.py`). GST weights under `gst_updated/results/`.

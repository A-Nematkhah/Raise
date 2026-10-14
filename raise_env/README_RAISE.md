# RAISE (highway)

LLM-guided reward evolution for `highway-fast-v0` with Stable-Baselines3 PPO.

| Package | Role |
|---------|------|
| `raise_core/` | LLM client, reward sandbox, evolution loop, Stage III |
| `domains/highway/` | HighwayEnv pack (prompts, env wrapper, PPO trainer, Score1) + `data/` |
| `raise_paths.py` | Puts `raise_env/` on `sys.path` for scripts |

Repository layout map: [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

## Install

```bash
cd raise_env
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The pinned GPU wheel is `torch==2.11.0+cu128`. Adjust the `cu1xx` suffix and
the PyTorch index URL in `requirements.txt` to match the local CUDA driver, or
pass `--device cpu`.

## API keys (Groq)

1. Copy `groq_keys.json.example` → `groq_keys.json`
2. Add keys: `{"keys": ["gsk_...", "..."]}`
3. Or set `GROQ_API_KEY` for a single key (pool disabled)

## Ollama (local, no API key)

For predictable latency, use the non-thinking model variant:

```bash
ollama pull frob/qwen3.5-instruct:4b
python scripts/run_raise.py --llm ollama --llm-model frob/qwen3.5-instruct:4b --output-dir results/ollama_run
```

The server URL defaults to `http://localhost:11434/v1`; override it with
`OLLAMA_BASE_URL`. If Gen0 rejection rates are unexpectedly high, inspect
`results/.../closed_loop/stage1_rejections.jsonl` for the truncation-specific
`no closing code fence found` error before assuming a sandbox problem.

## Run matrix

| Goal | Command | Hardware | Time |
|------|---------|----------|------|
| Wiring smoke | `python scripts/run_raise.py --fast` | CPU | seconds |
| Stage I dataset | `python scripts/collect_highway_stage1_dataset.py` | CPU | minutes |
| Real PPO, no LLM | `python scripts/run_raise.py --llm seed --allow-seed-llm --population 2 --generations 1 --stage2-train-steps 256 --stage3-train-steps 512 --device cpu` | CPU | minutes |
| Full run | `python scripts/run_raise.py --llm groq --output-dir results/highway_run` | GPU/CPU + Groq | hours |

Resume: rerun with the same `--output-dir` (on by default; `--no-resume`
starts fresh). Checkpoints: `closed_loop/checkpoint.json` (per trained
candidate) and `stage3/checkpoint.json` (per Stage III candidate).

## What a run does

1. Gen0: the LLM proposes N reward functions; each passes the sandbox
   (AST policy → spawned smoke test → compile).
2. Each of G generations: Score1 on the Stage I dataset, PPO train + holdout
   eval of every new candidate, rank by `highway_fitness`, LLM crossover /
   mutation (with the evaluation metrics in the prompt) / random restarts,
   elitism. Default next generation at N=6: 1 elite + 2 crossover + 2
   mutation + 1 random; `--n-crossover`, `--n-mutation`, `--n-random` and
   `--no-elitism` change the split (elite + children must equal
   `--population`; an unset count absorbs the remainder).
3. Stage III: long PPO on the final population (on eval seeds no generation
   saw); the final candidate is the best Stage III fitness.
4. Final blind eval: the winner alone is re-evaluated on further unseen seeds
   (`--final-eval-episodes`, default 100). Report `manifest.json` →
   `final_eval.fitness`, not the selection score, which is biased upward.

Outputs under `--output-dir`: `config.json`, `seed_reward.py`,
`closed_loop/` (`epochs.jsonl`, `pop_epoch_*.json`, `REPORT.txt`,
`stage1_rejections.jsonl`, `stage2_train/` PPO models),
`stage2_population.json`, `stage3_train/`, `stage3_population.json`,
`best_stage{1,2,3}.json`, `final_candidate.json`, `manifest.json`, and
`plots/` (non-fast runs).

## Tests

```bash
pytest -m "not slow"   # CI default
pytest -m slow         # 1 real-env rollout test
```

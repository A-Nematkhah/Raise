# RAISE (CrowdNav++)

LLM-guided reward evolution for crowd-aware robot navigation in the
[variable-human CrowdNav++](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
`CrowdSimVarNum` simulator with Stable-Baselines3 PPO.

| Package | Role |
|---------|------|
| `raise_core/` | LLM client, reward sandbox, evolution loop, Stage III |
| `domains/crowdnav/` | CrowdNav++ pack (prompts, env wrapper, PPO trainer, Score1, viz) + `data/` |
| `third_party/crowdnavpp/` | CrowdNav++ `CrowdSimVarNum` @ `3907731`, unmodified (MIT) |
| `raise_paths.py` | Puts `raise_env/` and the vendored simulator on `sys.path` |

Repository layout map: [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

## Install

```bash
cd raise_env
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install Cython==0.29.36 cmake
.venv\Scripts\python.exe -m pip install --no-build-isolation "pyrvo2 @ git+https://github.com/sybrenstuvel/Python-RVO2.git@c2c46ba8d59556aa10faf03479293236efea154d"
```

The simulator needs `gym==0.15.7` (in `requirements.txt`) and Python-RVO2
(`import rvo2`, ORCA for the humans). Python-RVO2 has no wheel: its
`setup.py` runs CMake and Cython, so it needs a C++ compiler (MSVC Build
Tools on Windows, `build-essential` on Linux) and `--no-build-isolation`
(Cython must already be installed). CI builds it on Ubuntu with the commands
above. On Windows, `cmake --build` defaults to the Debug configuration and
writes `RVO.lib` under `build/RVO2/src/Debug/`, where the pinned `setup.py`
does not look; if linking fails, build the checkout manually with
`cmake --build . --config Release` and copy the library to
`build/RVO2/src/`.

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
| Stage I dataset | `python scripts/collect_crowdnav_stage1_dataset.py` | CPU | seconds |
| Real PPO, no LLM | `python scripts/run_raise.py --llm seed --allow-seed-llm --population 2 --generations 1 --stage2-train-steps 256 --stage3-train-steps 512 --device cpu` | CPU | minutes |
| Full run | `python scripts/run_raise.py --llm groq --output-dir results/crowdnav_run` | GPU/CPU + Groq | hours |

Resume: rerun with the same `--output-dir` (on by default; `--no-resume`
starts fresh). Checkpoints: `closed_loop/checkpoint.json` (per trained
candidate) and `stage3/checkpoint.json` (per Stage III candidate).

## What a run does

1. Gen0: the LLM proposes N reward functions; each passes the sandbox
   (AST policy → spawned smoke test → compile).
2. Each of G generations: Score1 on the Stage I dataset, PPO train + holdout
   eval of every new candidate, rank by fitness (mean upstream episode task
   return on `val` scenarios), LLM crossover /
   mutation (with the evaluation metrics in the prompt) / random restarts,
   elitism. Default next generation at N=6: 1 elite + 2 crossover + 2
   mutation + 1 random; `--n-crossover`, `--n-mutation`, `--n-random` and
   `--no-elitism` change the split (elite + children must equal
   `--population`; an unset count absorbs the remainder).
3. Stage III: long PPO on the final population (evaluated on `val`
   scenarios drawn from a separate seed base; `val` has only 100 cases, so
   they can overlap with Stage II ones). By default candidates whose Stage II
   holdout CR ≥ `--collapse-cr` (0.8) are skipped (`--no-stage3-skip-collapsed`
   to train everyone; if all finalists are collapsed they are all trained).
   Optional `--stage3-top-k` keeps only the best Stage-II survivors;
   `--stage3-seeds K` trains each finalist K times and selects by mean
   fitness (default K=1). The winner is the best Stage III fitness; ties
   within 2 SE use paired per-case SEs when episode `case` lists match.
4. Final blind eval: the winner is re-evaluated on upstream `test`
   scenarios (`--final-eval-episodes`, default 100). By default every
   statistical tie is also re-evaluated (`tied_final_evals.json`, report
   only — never used for selection). Report `manifest.json` →
   `final_eval.fitness`, not the selection score, which is biased upward.
5. Optional baselines (`--baselines`, off by default): native-reward PPO,
   best non-collapsed Stage II genome, and ORCA/Linear controllers on the
   same test protocol → `baselines.json` with paired deltas vs the winner.

**Stage II eval precision.** With 50 holdout episodes, most of the
between-candidate fitness variance in a 16h CrowdNav run was evaluation
noise. Prefer `--stage2-eval-episodes 200` (or more); evaluation is ~0.02 s
per episode. `--stage2-eval-split {val,train}` chooses the holdout pool
(default `val`; Stage III/final stay on val/test).
`--stage2-successive-halving` (off by default) trains all at K2/3 then
continues the top half from their own checkpoints for the remaining 2K2/3.
`--mutation-exclude-collapsed` (off by default) drops CR≥`--collapse-cr`
genomes from the mutation parent pool.

Outputs under `--output-dir`: `config.json`, `config.resume_*.json` (on
resume), `provenance.json`, `run_<UTC>.log`, `seed_reward.py`,
`closed_loop/` (`epochs.jsonl`, `pop_epoch_*.json`, `REPORT.txt`,
`stage1_rejections.jsonl`, `stage2_train/` PPO models),
`stage2_population.json`, `stage3_train/`, `stage3_population.json`,
`best_stage{1,2,3}.json`, `final_candidate.json`, `tied_final_evals.json`,
`manifest.json`, `baselines.json` (if `--baselines`), and `plots/`
(non-fast runs: metric plots plus `plots/viz/` with a GIF, frame strip and
trajectory plot of the best policy on `test` scenarios; stochastic by
default).

Robot / action options: `--robot-kinematics {holonomic,unicycle}`,
`--action-mode {discrete,continuous}` (discrete uses the configured wrapper
action table; defaults to 81 actions), `--env-config` (a CrowdNav++ Python
config file exposing `Config` or `config`).

## Sanity checks and ablations

```bash
# CrowdNav++ ORCA baseline report on the test split
python scripts/check_orca_reference.py --out results/orca_report.json
# Baselines on a finished run; --seeds applies to native-reward PPO (A)
python scripts/run_baselines.py --run-dir results/crowdnav_16h --seeds 3 --skip-b --skip-c
# PPO discount sweep: fixed rewards and blind eval, K3 per (reward, gamma, seed)
python scripts/run_gamma_sweep.py --run-dir results/crowdnav_16h --gammas matched,0.99,0.995 --rewards native,final --seeds 2
```

CrowdNav++'s evaluator sums native rewards within each episode and reports the
mean; RAISE uses that same undiscounted episode return for fitness. PPO's
`PPO_GAMMA` discounts learned rewards during training and is independent of
the selection metric. In the PPO discount sweep, `matched` means `1.0` (no
extra evaluation discount). `run_baselines.py --ppo-gamma G` writes
`baselines_gamma<G>.json`; the sweep writes `gamma_sweep.json` (resumable,
paired per-seed deltas vs the reference gamma). Score1 continues to use
`PPO_GAMMA` for its Stage I preference proxy.

LLM: Groq default `max_tokens` is 3500 (gpt-oss reasoning tokens count);
override with `--llm-max-tokens` / `--llm-reasoning-effort`.

## Tests

```bash
pytest -m "not slow"   # CI default
pytest -m slow         # end-to-end CLI run with real PPO (about 1 min on CPU)
```

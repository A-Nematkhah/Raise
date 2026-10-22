# RAISE (CrowdNav++)

LLM-guided reward evolution for crowd-aware robot navigation in the
[variable-human CrowdNav++](https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)
`CrowdSimVarNum` simulator with Stable-Baselines3 PPO.

| Package | Role |
|---------|------|
| `raise_core/` | LLM client, reward sandbox, evolution loop, Stage III |
| `domains/crowdnav/` | CrowdNav++ pack (prompts, env wrapper, PPO trainer, EvoNav Score_1, viz) + `data/` |
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

Default `requirements.txt` pins CPU-friendly `torch==2.11.0`. For CUDA wheels,
install `requirements-cuda.txt` (pinned `torch==2.11.0+cu128`; adjust the
`cu1xx` suffix / index URL to match the local driver), or pass `--device cpu`.

## API keys (Groq)

1. Copy `groq_keys.json.example` → `groq_keys.json`
2. Add keys: `{"keys": ["gsk_...", "..."]}`
3. Or set `GROQ_API_KEY` for a single key (pool disabled)

## API keys (xAI / Grok)

1. Copy `xai_key.example` → `xai_key`
2. Put your key on one line (starts with `xai-…`)
3. Or set `XAI_API_KEY`
4. Run with `--set llm.provider=xai --set llm.model=grok-4.20-0309-reasoning`  
   (or use `configs/raise_2h.yaml`, which defaults to xAI)

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
| Wiring smoke | `python scripts/run_raise.py --dry-run` | CPU | seconds |
| Stage I dataset | `python scripts/collect_crowdnav_stage1_dataset.py` | CPU | seconds |
| Real PPO, no LLM | `python scripts/run_raise.py --allow-seed-llm --set budget.population_size=2 --set budget.generations=1 --set budget.stage2_train_steps=256 --set budget.stage3.train_steps=512 --set ppo.device=cpu` | CPU | minutes |
| Full run | `python scripts/run_raise.py --set llm.provider=groq --output-dir results/crowdnav_run` | GPU/CPU + Groq | hours |

## Configuration

All simulator and budget defaults live in **`configs/raise.yaml`**.
Open that file to edit values; its header lists run commands and the old-flag
→ key map. Schema/validation is in `raise_config.py` (not for editing numbers).

Override at runtime:

- **`--config path.yaml`** — alternate YAML (or `.py` exporting `CONFIG`)
- **`--set section.key=value`** — repeatable; nested keys use dots (e.g.
  `budget.stage3.train_steps=50000`)
- Precedence: `configs/raise.yaml` < `--config` < `--set`

On run start, `scripts/run_raise.py` writes **`config_snapshot.json`** (with
sha256) under `--output-dir`. Resume compares the snapshot unless
`--allow-config-drift` is set (legacy runs without a snapshot skip the check).

`run_raise.py` flags (everything else moves to `--set`):

| Flag | Purpose |
|------|---------|
| `--config`, `--set` | Effective `raise_config` |
| `--output-dir` / `--run-dir` | Run directory |
| `--resume` / `--no-resume` | Checkpoint resume (default: resume) |
| `--allow-config-drift` | Resume despite snapshot mismatch |
| `--dry-run` / `--fast` | Stub trainers + smoke Score_1 |
| `--print-config` | JSON dump of effective config and exit |
| `--verbose` | DEBUG logging |
| `--baselines` | Run `scripts/run_baselines.py` after the pipeline |
| `--allow-seed-llm` | Allow `llm.provider=seed` on non-dry runs |

### `run_raise.py` flag migration (`--set` keys)

| Removed flag | `raise_config` key |
|--------------|-------------------|
| `--seed` | `seed` |
| `--llm` | `llm.provider` |
| `--llm-model` | `llm.model` |
| `--llm-max-tokens` | `llm.max_tokens` |
| `--llm-reasoning-effort` | `llm.reasoning_effort` |
| `--llm-seed` | `llm.seed` |
| `--device` | `ppo.device` |
| `--torch-threads` | `ppo.torch_threads` |
| `--score1` | (use `--dry-run` for smoke; dataset is default) |
| `--stage1-dataset` | `budget.stage1.dataset_path` |
| `--population` | `budget.population_size` |
| `--generations` | `budget.generations` |
| `--stage2-train-steps` | `budget.stage2_train_steps` |
| `--stage2-eval-episodes` | `budget.stage2_eval_episodes` |
| `--stage2-eval-split` | `budget.stage2_eval_split` |
| `--stage2-seeds` | `budget.stage2_seeds` |
| `--stage2-paired-seeds` / `--no-stage2-paired-seeds` | `budget.stage2_paired_seeds` |
| `--stage2-successive-halving` | `budget.stage2_successive_halving` |
| `--stage3-train-steps` | `budget.stage3.train_steps` |
| `--stage3-eval-episodes` | `budget.stage3.eval_episodes` |
| `--stage3-warm-start` | `budget.stage3.warm_start` |
| `--stage3-skip-collapsed` | `budget.stage3.skip_collapsed` |
| `--stage3-seeds` | `budget.stage3.n_seeds` |
| `--stage3-top-k` | `budget.stage3.top_k` |
| `--final-eval-episodes` | `budget.final_eval_episodes` |
| `--eval-tied-finalists` | `budget.eval_tied_finalists` |
| `--elitism` | `budget.elitism` |
| `--n-crossover` / `--n-mutation` / `--n-random` | `budget.n_crossover` / `n_mutation` / `n_random` |
| `--mutation-parents` | `selection.mutation_parents` |
| `--mutation-exclude-collapsed` | `budget.mutation_exclude_collapsed` |
| `--collapse-cr` | `budget.collapse_cr` |
| `--n-envs` | `budget.n_envs` |
| `--label-workers` | `budget.label_workers` |
| `--warm-start` | `budget.warm_start` |
| `--eval-mode` | `budget.eval_mode` |
| `--action-mode` | `env.action_mode` |
| `--robot-kinematics` | `env.kinematics` |
| `--action-speed-samples` | `env.speed_samples` |
| `--action-rotation-samples` | `env.rotation_samples` |
| `--env-config` | edit `configs/raise.yaml` `env:` (or `--set env.human_num=…`) |
| `--feedback-card` | `selection.feedback_card` |
| `--max-input-tokens` | `selection.max_input_tokens` |
| `--target-input-tokens` | `selection.target_input_tokens` |
| `--feedback-history` | `selection.feedback_history` |
| `--feedback-trajectories` | `selection.feedback_trajectories` |
| `--feedback-random-calls` | `selection.feedback_random_calls` |
| `--feedback-training-dynamics` | `selection.feedback_training_dynamics` |
| `--feedback-components` | `selection.feedback_components` |
| `--selection-mode` | `selection.mode` |
| `--selection-top-k` | `selection.top_k` |
| `--selection-votes` | `selection.n_votes` |
| `--selection-temperature` | `selection.temperature` |
| `--selection-restart-probability` | `selection.p_restart` |
| `--selection-guards` | `selection.guards` |
| `--selection-guard-success-threshold` | `selection.guard_success_threshold` |
| `--selection-guard-min-path-length` | `selection.guard_min_path_length` |
| `--selection-reflection-note-cap` | `selection.reflection_note_cap` |
| `--selection-seed` | `seed` (ranking shuffle seed) |
| `--infra-error-policy` | `selection.infra_error_policy` |
| `--max-fallback-fraction` | `selection.max_fallback_fraction` |
| `--strict-fallback` | `selection.strict_fallback` |
| `--ppo-normalize-reward` | `ppo.normalize_reward` |
| `--domain` | (CrowdNav-only entry point; domain fixed to `crowdnav`) |

Environment presets: **`evaluation_variants.humans_{10,15,20}`** (and
`*_random` for the EvoNav w/ random protocol). Default simulator fields live
in `CONFIG.env` / `configs/raise.yaml`.

Resume: rerun with the same `--output-dir` (on by default; `--no-resume`
starts fresh). Checkpoints: `closed_loop/checkpoint.json` (per trained
candidate) and `stage3/checkpoint.json` (per Stage III candidate).

## What a run does

1. Gen0: the LLM proposes N reward functions; each passes the sandbox
   (AST policy → spawned smoke test → compile).
2. Each of G generations: Score1 on the Stage I dataset, PPO train + holdout
   eval of every new candidate, rank by mean outcome score (+1 success,
   -1 collision, -0.5 timeout) on held-out `val` scenarios, LLM crossover /
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
per episode. Stage II selection always uses the held-out `val` pool; training scenarios
cannot be selected as holdout fitness.
`--stage2-successive-halving` (off by default) trains all at K2/3 then
continues the top half from their own checkpoints for the remaining 2K2/3.
`--mutation-exclude-collapsed` (off by default) drops CR≥`--collapse-cr`
genomes from the mutation parent pool.

The optional EvoNav-style selector is enabled with
`--selection-mode llm_rank`; `legacy` remains the default. In this mode the
shared LLM ranks all generation candidates from their existing diagnostic
cards, keeps the top `--selection-top-k` elites, and generates the remaining
slots by elite mutation, top-two crossover, or a random restart selected with
`--selection-restart-probability`. Each ranking uses
`--selection-votes` shuffled votes and Borda aggregation. Set
`--selection-guards` to append optional sanity flags after each card (without
changing its contents). The selector applies at every Stage II generation and
again to trained Stage III candidates. It writes prompts/responses and parsed
votes to `closed_loop/llm_selection.jsonl`, with rank consistency metrics in
`closed_loop/rank_consistency.csv`; invalid JSON after three attempts falls
back to the existing fitness-based ordering.

```powershell
python scripts\run_raise.py --set llm.provider=groq --set selection.mode=llm_rank --set selection.top_k=2 --set selection.n_votes=3 --set selection.temperature=0.1 --set selection.p_restart=0.1 --output-dir results\crowdnav_llm_rank
```

Outputs under `--output-dir`: `config.json`, `config.resume_*.json` (on
resume), `provenance.json`, `run_<UTC>.log`, `seed_reward.py`,
`closed_loop/` (`epochs.jsonl`, `pop_epoch_*.json`, `REPORT.txt`,
`stage1_rejections.jsonl`, optional `llm_selection.jsonl` and
`rank_consistency.csv`, `stage2_train/` PPO models),
`stage2_population.json`, `stage3_train/`, `stage3_population.json`,
`best_stage{1,2,3}.json`, `final_candidate.json`, `tied_final_evals.json`,
`manifest.json`, `baselines.json` (if `--baselines`), and `plots/`
(non-fast runs: metric plots plus `plots/viz/` with a GIF, frame strip and
trajectory plot of the best policy on `test` scenarios; stochastic by
default).

Robot / action options: `--set env.kinematics=…`, `--set env.action_mode=…`
(discrete uses the wrapper action table). Default env: 30 s time limit, 10
humans (`human_num=10`, `human_num_range=0`), holonomic kinematics,
constant-velocity prediction (5 steps, 0.25 s), `robot.visible=False`
(EvoNav-style: humans ignore the robot). Stage III can run
`budget.stage3.refine_rounds` LLM refine+retrain cycles (G3). Stage I
collection uses `budget.stage1.collect_kinematics`. The wrapper corrects
CrowdNav++'s unicycle reset sampling to honor the configured population range.
Timeout is always reported independently as TR and as an episode outcome.

Before a long evolution, train a PPO policy with CrowdNav++'s native reward
and evaluate it on 100 held-out validation cases:

```bash
python scripts/dev/calibrate_crowdnav_native.py --train-steps 300000 --eval-episodes 100 --seed 425 --n-envs 4 --device cpu
```

To continue training from an existing native-reward PPO checkpoint, use
`--resume-from` and write to a new output directory. `--train-steps` is the
additional budget; the checkpoint's timestep counter is preserved:

```bash
python scripts/dev/calibrate_crowdnav_native.py --output-dir results/calibration_600k --train-steps 300000 --eval-episodes 100 --seed 425 --n-envs 4 --device cpu --resume-from results/calibration/native_reward_ppo.zip
```

The optional fixed-17-human generalization config can evaluate a completed
Stage III checkpoint without retraining it:

```bash
python scripts/evaluate_crowdnav_generalization.py --run-dir results/crowdnav_run --episodes 100
```

Its report includes outcome-based fitness, separate SR/CR/TR, and per-episode
navigation time, path length and minimum separation.

## Sanity checks and ablations

```bash
# CrowdNav++ ORCA baseline report on the test split
python scripts/dev/check_orca_reference.py --out results/orca_report.json
# Baselines on a finished run; --seeds applies to native-reward PPO (A)
python scripts/run_baselines.py --run-dir <run-dir> --seeds 3 --skip-b --skip-c
# PPO discount sweep: fixed rewards and blind eval, K3 per (reward, gamma, seed)
python scripts/dev/run_gamma_sweep.py --run-dir <run-dir> --gammas matched,0.99,0.995 --rewards native,final --seeds 2
```

CrowdNav++ native episode return is retained as a diagnostic. RAISE selection
fitness is the mean outcome score (+1 success, -1 collision, -0.5 timeout);
PPO's `PPO_GAMMA` discounts learned rewards during training and is independent
of selection. In the PPO discount sweep, `matched` means `1.0` (no extra
evaluation discount). `run_baselines.py --ppo-gamma G` writes
`baselines_gamma<G>.json`; the sweep writes `gamma_sweep.json` (resumable,
paired per-seed deltas vs the reference gamma). Score1 continues to use
`PPO_GAMMA` for its Stage I preference proxy.

LLM: Groq default `max_tokens` is 3500 (gpt-oss reasoning tokens count);
override with `--llm-max-tokens` / `--llm-reasoning-effort`.

## Behavior changes

Defaults that changed relative to earlier CrowdNav runs (all overridable):

- **`StageIConfig()` defaults** match closed-loop **`raise_config`** (**N=6 /
  G=4**). Historical standalone explorers that expected 8/10 must pass explicit
  `population_size` / `generations`.
- **Stage II paired seeds** default **on** (`--stage2-paired-seeds` /
  `--no-stage2-paired-seeds`). Every candidate in a generation shares the same
  PPO train seed(s); `--stage2-seeds K` averages fitness over K seeds.
  Pre-patch checkpoints missing these keys resume with **legacy** values
  (`paired=False`, `seeds=1`) without drift-abort.
- **Cross-seed SE** (`fitness_se_source=across_seeds`) requires **≥3**
  successful seeds; fewer keep `episodes_only` (with a warning).
- **`successive_halving` + `stage2_seeds>1`** is rejected (unsupported combo).
- Closed-loop manifest uses **`run_quality`** (`ok` / `contaminated`) so it
  does not collide with checkpoint ``status`` (`running` / `completed` /
  `aborted_infra`).
- **`infra_error_policy`** defaults to **`retry_then_abort`**
  (`--infra-error-policy {retry_then_abort,abort,penalize}`).
- **Config drift on resume aborts** unless `--allow-config-drift` is set
  (result-changing keys: train steps, seeds, population, eval episodes,
  kinematics, gamma, PPO knobs).
- **`max_fallback_fraction`** defaults to **0.5** (mark contaminated when more
  than half a generation is seed/clone fallback); `--strict-fallback` aborts.
- **Stage III final eval** with `--stage3-seeds K>1` re-evaluates **all**
  successful seed checkpoints and reports the **mean** fitness (cross-seed SE);
  `checkpoint_path` is the **median**-fitness seed for visualization.
- **`ppo_normalize_reward`** defaults to **False** (`--ppo-normalize-reward`
  enables VecNormalize reward normalization on the training VecEnv only).
- **Sandbox**: `RewardValidator` requires a domain attribute allowlist (or
  explicit `allow_unrestricted=True`); `range` / `**` / episode `memory` are
  bounded; POSIX smoke children get a 2 GiB `RLIMIT_AS` cap.
- **`min_separation_post`**: post-step robot–human gap on `CrowdNavRewardState`
  (legacy datasets that omit it recompute from stored robot/humans geometry).
- **LLM `seed`**: forwarded from `--llm-seed` / run seed to Groq/vLLM when
  supported; logged and omitted when the provider rejects it.

## Tests

```bash
pytest -m "not slow"   # CI default
pytest -m slow         # end-to-end CLI run with real PPO (about 1 min on CPU)
```

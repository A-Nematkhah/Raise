# Scripts index

All commands from `raise_env/`.

## RAISE orchestration

| Script | Purpose |
|--------|---------|
| `run_raise.py` | Canonical CLI (Alg.1 or `--closed-loop`) |
| `run_raise.py --profile <name>` | Named presets from `presets.CLOSED_LOOP_PROFILES`: `smoke` (real A2C/PPO, tiny budgets), `1h`, `12h` (+ `--warm-surrogate`), `18h`, `paper_scale` (multi-seed). Resume = same `--output-dir` |
| `run_raise_highway.py` | Highway closed-loop run; budget in `presets.CLOSED_LOOP_PROFILES["highway"]` |

## Data / models

| Script | Purpose |
|--------|---------|
| `collect_stage1_dataset.py` | CrowdNav Score1 trajectories |
| `collect_highway_stage1_dataset.py` | Highway Stage I trajectories |
| `bootstrap_surrogate.py` | Fit Stage-II surrogate (warm start) |
| `run_active_learning_step.py` | One offline AL acquire/refit |

## Smokes / analysis

| Script | Purpose |
|--------|---------|
| `print_raise_report.py` | Closed-loop REPORT.txt |
| `plot_raise_run.py` / `visualize_raise.py` | Plots / viz |
| `eval_raise_checkpoint.py` | Re-eval a checkpoint |
| `report.py` | Table 1/2 style summaries |
| `train_ds_rnn.py` | DS-RNN baseline |
| `run_p0_*.py` | Thesis/audit helpers |
| `label_reliability_highway.py` / `analyze_label_reliability.py` | Highway Stage II label noise across PPO seeds |

Shared preflight: `_prereqs.py` (imported by `run_raise_highway.py`).

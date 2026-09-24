# Scripts index

All commands from `raise_env/`.

## RAISE orchestration

| Script | Purpose |
|--------|---------|
| `run_raise.py` | Canonical CLI (Alg.1 or `--closed-loop`) |
| `run_raise_1h.py` / `run_raise_1h.ps1` | Short closed-loop smoke |
| `run_raise_12h.py` | Overnight closed-loop (+ `--warm-surrogate`) |
| `run_raise_18h.py` | Longer closed-loop + Stage III |
| `run_raise_paper_scale.py` | Multi-seed paper budgets |

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
| `run_stage2_smoke.py` / `run_stage3_smoke.py` | Real trainer smokes |
| `print_raise_report.py` | Closed-loop REPORT.txt |
| `plot_raise_run.py` / `visualize_raise.py` | Plots / viz |
| `eval_raise_checkpoint.py` | Re-eval a checkpoint |
| `report.py` | Table 1/2 style summaries |
| `train_ds_rnn.py` | DS-RNN baseline |
| `run_p0_*.py` | Thesis/audit helpers |

Shared preflight: `_prereqs.py` (imported by overnight scripts).

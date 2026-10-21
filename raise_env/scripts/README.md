# Scripts index

All commands from `raise_env/`.

## Production / workflow

| Script | Purpose |
|--------|---------|
| `run_raise.py` | The RAISE run (closed loop + Stage III). Resume = same `--output-dir` |
| `collect_crowdnav_stage1_dataset.py` | CrowdNav Stage I trajectories for Score1 |
| `print_raise_report.py` | Print / regenerate `closed_loop/REPORT.txt` |
| `plot_raise_run.py` | Plots for a finished run |
| `run_baselines.py` | Native / Stage-II / ORCA baselines on a finished `--run-dir` |
| `evaluate_crowdnav_generalization.py` | Holdout generalization eval for a trained policy |
| `smoke_crowdnavpp.py` | Minimal CrowdNav++ import + step smoke |

## Dev / analysis (`scripts/dev/`)

| Script | Purpose |
|--------|---------|
| `dev/run_gamma_sweep.py` | Discount-factor sweep over rewards in a `--run-dir` |
| `dev/calibrate_crowdnav_native.py` | Calibrate / characterize the native CrowdNav++ reward |
| `dev/check_orca_reference.py` | ORCA reference policy sanity check |
| `dev/analyze_score1_vs_fitness.py` | Compare Score1 ranking vs Stage II fitness (CLI) |
| `dev/bench_speed.py` | Throughput / profiling benches (optional `--run-dir`) |

Shared logic lives under `raise_core/` (`post_run_baselines`, `score1_fitness`, `script_utils`).

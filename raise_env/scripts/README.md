# Scripts index

All commands from `raise_env/`.

| Script | Purpose |
|--------|---------|
| `run_raise.py` | The RAISE run (closed loop + Stage III). Resume = same `--output-dir` |
| `collect_crowdnav_stage1_dataset.py` | CrowdNav Stage I trajectories for Score1 |
| `print_raise_report.py` | Print / regenerate `closed_loop/REPORT.txt` |
| `plot_raise_run.py` | Plots for a finished run |
| `run_baselines.py` | Native / Stage-II / ORCA baselines on a finished `--run-dir` |
| `run_gamma_sweep.py` | Discount-factor sweep over rewards in a `--run-dir` |
| `evaluate_crowdnav_generalization.py` | Holdout generalization eval for a trained policy |
| `calibrate_crowdnav_native.py` | Calibrate / characterize the native CrowdNav++ reward |
| `check_orca_reference.py` | ORCA reference policy sanity check |
| `analyze_score1_vs_fitness.py` | Compare Score1 ranking vs Stage II fitness |
| `bench_speed.py` | Throughput / profiling benches (optional `--run-dir`) |
| `smoke_crowdnavpp.py` | Minimal CrowdNav++ import + step smoke |

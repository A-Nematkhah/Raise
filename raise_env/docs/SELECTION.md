# Highway final policy selection

At the end of a **highway** RAISE run, do **not** use:

```bash
python scripts/eval_raise_checkpoint.py --run-dir <run> --best-ever
```

`--best-ever` ranks Stage III history by `SR − CR − 0.5·TR` only. It ignores
mean speed, progress, and `soft_success`, and it is **not** Pareto ranking.
That scalar remains valid for **CrowdNav** only.

## Deliberate pick from the Pareto front

1. List feasible non-dominated candidates (front 0) with raw trade-offs:

```bash
python scripts/eval_raise_checkpoint.py --run-dir <run> --pareto-front
```

2. Inspect SR / CR / TR / mean_speed / progress / soft_success side by side.
   There is **no** automatic single winner.

3. Choose one id explicitly:

```bash
python scripts/eval_raise_checkpoint.py --run-dir <run> \
  --pareto-front --candidate-id <id>
```

This writes `selected_candidate.json` under the run dir. Deploy / further
eval that id only after this human choice.

## What stays hand-specified

- The six raw objectives (SR, CR, TR, progress, mean_speed, soft_success).
- Calibration percentiles for feasibility (`pareto_rank.py`).
- `soft_success` speed bar is a **fixed** nominal traffic target (`V_TARGET`),
  intentionally separate from auto-calibrated `v_floor` (see comment in
  `domains/highway/adapter.py`).

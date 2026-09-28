# Highway final policy selection

At the end of a **highway** RAISE run, do **not** use:

```bash
python scripts/eval_raise_checkpoint.py --run-dir <run> --best-ever
```

`--best-ever` ranks Stage III history by `SR − CR − 0.5·TR` only. It ignores
mean speed and progress, and it is **not** Pareto ranking.
That scalar remains valid for **CrowdNav** only.

## Deliberate pick from the Pareto front

1. List feasible non-dominated candidates (front 0) with raw trade-offs:

```bash
python scripts/eval_raise_checkpoint.py --run-dir <run> --pareto-front
```

2. Inspect SR / CR / TR / mean_speed / progress side by side
   (`soft_success` may appear in logs as a human diagnostic only).
   There is **no** automatic single winner.

3. Choose one id explicitly:

```bash
python scripts/eval_raise_checkpoint.py --run-dir <run> \
  --pareto-front --candidate-id <id>
```

This writes `selected_candidate.json` under the run dir. Deploy / further
eval that id only after this human choice.

## What stays hand-specified

- The five raw Pareto objectives (SR, CR, TR, progress, mean_speed).
- Calibration percentiles for feasibility (`pareto_rank.py`).
- `soft_success` (fixed `V_TARGET` bar) is a **legacy human diagnostic only** —
  not a Pareto / LLM selection objective (see `domains/highway/adapter.py`).

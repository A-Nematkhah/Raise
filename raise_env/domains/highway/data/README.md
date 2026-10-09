# Highway domain data

| Path | Role |
|------|------|
| `stage1_dataset/` | Score1 trajectories (`trajectories.jsonl`) |
| `surrogate_dataset/` | Surrogate feature/label jsonl (or under `results/…`) |
| `active_learning/` | AL queue when not isolating under the run dir |

## Stage I dataset

Collect a **balanced** set for hybrid Score1 (fast success vs crawl success vs collision):

```bash
python scripts/collect_highway_stage1_dataset.py
```

Expect roughly:
- `success` with `behavior=safe_fast` (cruise [22, 25] m/s)
- `success` with `behavior=crawl` ([10, 14] m/s — hard negative for Score1)
- `success` with `behavior=decoy_lag` ([15, 17.5] m/s — soft lag decoy)
- `collision` / `timeout` (label `timeout` = off-road / non-collision failure;
  its frames have `off_road=True, timeout=False`)

Manifest lists labels and `state_schema_version`; each traj stores
`metadata.mean_speed` / `mean_progress`.

Datasets collected before `state_schema_version=2` have a wrong
`ego.lane_index` and `timeout=True` on off-road frames. Upgrade in place
(a timestamped backup is written next to the dataset):

```bash
python scripts/migrate_highway_stage1_dataset.py --dataset domains/highway/data/stage1_dataset
```

## Warm surrogate (optional)

```bash
python scripts/bootstrap_surrogate.py --domain highway --llm groq
python scripts/run_raise_highway.py --llm groq --warm-surrogate artifacts/highway_surr_warm
```

Highway surrogate fits **SR/CR/TR + mean_speed + mean_progress + soft_success**.
Warm datasets whose manifest `feature_schema_version` differs from the current
highway version are relabeled from scratch automatically. The default `highway`
profile (`warm_root=""`) skips warm and labels inside closed-loop.

# Highway domain data

| Path | Role |
|------|------|
| `stage1_dataset/` | Score1 trajectories (`trajectories.jsonl`) |

## Stage I dataset

```bash
python scripts/collect_highway_stage1_dataset.py
```

All rollouts are genuine (no speed rescaling), 40 s long, and alternate
between the training and holdout traffic profiles. The default run (~6 min)
gives about 124 trajectories:

| role | source |
|------|--------|
| `positive` (~30) | `overtake` successes with ≥ 1 pass, 21–25 m/s |
| `neutral` (~16) | lane-locked survivors: `follow` (≤ 25 m/s behind leads), `safe_cruise` (20 m/s) |
| `lag` (~19) | `lag` (16 m/s), `crawl` (12 m/s) successes |
| `failure` (~59) | collisions of every behaviour (incl. slow ones) + 5 synthesized off-road (label `timeout`, frames `off_road=True, timeout=False`) |

Each trajectory stores `metadata.role`, `traffic`, `goal_speed`,
`mean_speed`, `mean_progress` and `overtakes`; Score1 reads `role`.
`state.action` is the meta action index, mapped to the run's action mode
when Score1 loads the dataset.

Datasets collected before `state_schema_version=2` have a wrong
`ego.lane_index` and `timeout=True` on off-road frames — recollect them.

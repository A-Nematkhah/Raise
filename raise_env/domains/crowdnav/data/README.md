# CrowdNav Stage I dataset

`stage1_dataset/` (git-ignored) holds the trajectories Score1 ranks reward
candidates on. Regenerate it with (about 10 s on CPU):

```bash
python scripts/collect_crowdnav_stage1_dataset.py --out domains/crowdnav/data/stage1_dataset
```

Defaults: 40 upstream `train` scenarios (seed 7) × 12 robot behaviours
(ORCA at several `safety_space` / speed / `time_horizon` settings, plus
Linear, slow Linear, stop and random), all rollouts of the vendored,
unmodified `crowd_sim`. Labels are the upstream episode outcomes
(`ReachGoal` → `success`, `Collision`, `Timeout`).

Files:

- `trajectories.jsonl`: one trajectory per line (`trajectory_id`,
  `scenario_id`, `label`, `behavior`, `metadata`, per-step reward `states`).
- `manifest.json`: `domain: "crowdnav"`, label counts, state schema
  version, upstream commit and collection settings. Datasets of another
  domain are refused on load.

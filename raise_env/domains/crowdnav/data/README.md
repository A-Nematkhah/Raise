# CrowdNav Stage I dataset

`stage1_dataset/` (git-ignored) holds the trajectories Score1 ranks reward
candidates on. Regenerate it with (about 10 s on CPU):

```bash
python scripts/collect_crowdnav_stage1_dataset.py --out domains/crowdnav/data/stage1_dataset
```

To collect with a different CrowdNav++ Python config, pass `--env-config`,
for example:

```bash
python scripts/collect_crowdnav_stage1_dataset.py --env-config third_party/crowdnavpp/crowd_nav/configs/config.py --out domains/crowdnav/data/stage1_dataset_custom
```

Defaults: 40 CrowdNav++ `train` scenarios (seed 7) × 12 robot behaviours
(ORCA at several `safety_space` / speed / `time_horizon` settings, plus
Linear, slow Linear, stop and random), all rollouts of the vendored,
unmodified `CrowdSimVarNum`. ORCA only receives humans visible in the current
upstream FOV. Holonomic rollouts are the default; `--kinematics unicycle`
collects unicycle rollouts when the supplied config satisfies the simulator's
unicycle human-count constraints. Score1 can re-express holonomic actions for
unicycle runs. Labels come from CrowdNav++'s native wrapper outcomes.

Files:

- `trajectories.jsonl`: one trajectory per line (`trajectory_id`,
  `scenario_id`, `label`, `behavior`, `metadata`, per-step reward `states`).
- `manifest.json`: `domain: "crowdnav"`, label counts, state schema
  version, upstream commit and collection settings. Datasets of another
  domain are refused on load.

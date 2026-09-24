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
- `success` with `behavior=safe_fast` (high speed/progress)
- `success` with `behavior=crawl` (low speed — negative example for Score1)
- `collision` / `timeout`

Manifest lists labels; each traj stores `metadata.mean_speed` / `mean_progress`.

## Warm surrogate (optional)

```bash
python scripts/bootstrap_surrogate.py --domain highway --llm groq
python scripts/run_raise_highway_4h.py --llm groq --warm-surrogate artifacts/highway_surr_warm
```

Highway surrogate fits **SR/CR/TR + mean_speed + mean_progress + soft_success**.
Older warm models trained on SR/CR/TR only should be rebuilt (`--force`) after
this schema change. Default 4h runner skips warm and labels inside closed-loop.

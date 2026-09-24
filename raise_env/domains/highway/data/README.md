# Highway domain data

| Path | Role |
|------|------|
| `stage1_dataset/` | Score1 trajectories (`trajectories.jsonl`) — collect via `scripts/collect_highway_stage1_dataset.py` |
| `surrogate_dataset/` | Surrogate feature/label jsonl (or run-isolated under `results/…`) |
| `active_learning/` | AL queue when not isolating under the run dir |

Collect (needs label diversity: success / collision / timeout):

```bash
python scripts/collect_highway_stage1_dataset.py
```

Warm bootstrap is **optional**; the 4h runner defaults to collecting Stage-II labels inside closed-loop:

```bash
python scripts/run_raise_highway_4h.py --llm groq
# later, if desired:
python scripts/bootstrap_surrogate.py --domain highway --llm groq
python scripts/run_raise_highway_4h.py --llm groq --warm-surrogate artifacts/highway_surr_warm
```

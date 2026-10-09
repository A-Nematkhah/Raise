# Domain pack

The environment lives in a **pack** under `domains/<name>/` with its own
tests and data. RAISE currently has one domain:

| Domain | Package | Role |
|--------|---------|------|
| `highway` | `domains.highway` | HighwayEnv (`highway-fast-v0`) + SB3 PPO |

Registry: `raise_core.domains` (`load_domain`, trainer / validator / Score1
factories).

## Layout

```text
domains/
  highway/
    pack.py, prompts.py, env_wrapper.py, adapter.py, …
    tests/
    data/                                # highway Stage I dataset
  README.md
```

See [`highway/spec.md`](highway/spec.md) for the state contract, Score1,
sandbox, fitness and run flow.

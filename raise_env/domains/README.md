# Domain pack

The environment lives in a **pack** under `domains/<name>/` with its own
tests and data. RAISE currently has one domain:

| Domain | Package | Role |
|--------|---------|------|
| `crowdnav` | `domains.crowdnav` | vita-epfl/CrowdNav `crowd_sim` (vendored) + SB3 PPO |

Registry: `raise_core.domains` (`load_domain`, trainer / validator / Score1
factories).

## Layout

```text
domains/
  crowdnav/
    pack.py, prompts.py, env_wrapper.py, adapter.py, …
    tests/
    data/                                # CrowdNav Stage I dataset (generated)
  README.md
```

See [`crowdnav/spec.md`](crowdnav/spec.md) for the state contract,
observation / action spaces, Score1 and fitness.

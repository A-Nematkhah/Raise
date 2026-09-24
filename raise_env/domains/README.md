# Domain packs

Each simulation backend is a **first-class package** under `domains/<name>/`
with its own **pack**, **tests**, **data**, and (when needed) **runtime** stack.

| Domain | Package | Role |
|--------|---------|------|
| `crowdnav` | `domains.crowdnav` | CrowdNav++ / GST / SRNN — paper baseline (**frozen**) |
| `highway` | `domains.highway` | HighwayEnv diagnostic (`highway-fast-v0`) |

Registry: `raise_core.domains` (`load_domain`, factories).

## Layout

```text
domains/
  crowdnav/
    pack.py, prompts.py, adapter.py, …   # RAISE surfaces
    tests/
    data/                                # stage1 / AL / surrogate
    runtime/                             # on sys.path via raise_paths
      crowd_sim/  crowd_nav/  rl/  gst_updated/
  highway/
    pack.py, prompts.py, env_wrapper.py, …
    tests/
    data/                                # highway stage1
  README.md
```

## Imports

```text
CLI / pytest
  → raise_paths.ensure_raise_paths()
  → import crowd_sim / crowd_nav / rl   # from domains/crowdnav/runtime
  → domains.crowdnav / domains.highway  # RAISE packs
```

## Adding a third environment

1. Create `domains/<name>/` with `get_pack() -> DomainPack` and `data/`
2. Optionally add `runtime/` if the env ships local Python packages
3. Register in `raise_core.domains._REGISTRY`
4. Do **not** put env-specific code into `raise_core`

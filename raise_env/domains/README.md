# Domain packs

Each simulation backend is a **first-class package** under `domains/<name>/`.

| Domain | Package | Role |
|--------|---------|------|
| `crowdnav` | `domains.crowdnav` | CrowdNav++ / GST / SRNN — paper baseline (**frozen**) |
| `highway` | `domains.highway` | HighwayEnv diagnostic (`highway-fast-v0`) |

Registry (environment-agnostic): `raise_core.domains` (`load_domain`, factories).

## Layout

```text
domains/
  crowdnav/     pack, prompts, adapter, state, dataset, regime, …
  highway/      pack, prompts, adapter, env_wrapper, stage1, state, …
  README.md     this file
```

## How RAISE uses a pack

```text
CLI --domain <name>
  → raise_core.domains.load_domain(name)
  → DomainPack (prompts, Score1, trainers, smoke states)
  → raise_core.pipeline
```

## Adding a third environment

1. Create `domains/<name>/` with `get_pack() -> DomainPack`
2. Register in `raise_core.domains._REGISTRY`
3. Add `domains/<name>/tests/`
4. Do **not** put env-specific code into `raise_core`

Compatibility shims remain at `crowd_nav.domains.*` for older imports.

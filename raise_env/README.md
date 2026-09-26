# raise_env

Active RAISE / Evonav experiment tree (LLM-guided reward search + domain packs).

## Architecture / repo layout

**`raise_core/` is the only live reward-search pipeline.** Closed-loop
(`raise_loop/`), Stage I–III orchestration, sandbox, surrogate, and AL all
live there.

**`domains/crowdnav/runtime/crowd_nav/reward_search/` is a frozen legacy
compatibility shim** (pre–domain-isolation import paths). Do not edit it for
algorithm work — see
[`domains/crowdnav/runtime/crowd_nav/reward_search/README_LEGACY.md`](domains/crowdnav/runtime/crowd_nav/reward_search/README_LEGACY.md).

| Area | Location |
|------|----------|
| RAISE core | `raise_core/` |
| CrowdNav domain | `domains/crowdnav/` |
| Highway domain | `domains/highway/` |
| CLIs | `scripts/` |
| Highway final pick | `docs/SELECTION.md` |

```bash
python scripts/run_raise.py --domain highway --closed-loop
python scripts/run_raise_highway_4h.py
```

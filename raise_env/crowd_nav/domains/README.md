# Domain packs — add a new environment

Each simulation / training backend the reward-search algorithm can target is a
**domain pack**: a folder under `crowd_nav/domains/<name>/`.

Today only **`crowdnav`** is registered (EvoNav Algorithm 1 baseline). Do **not**
add placeholder / stub packs. Add a real domain only when env code, metrics, and
prompts are ready.

## How the core talks to a pack

```text
CLI --domain <name>
  → load_domain(name)
  → DomainPack
       ├─ prompts + seed_reward_source     (LLM D.1–D.5)
       ├─ make_score_fn(...)               (Stage I)
       └─ make_stage2/3_trainer_for_domain (Stage II/III via adapter)
```

Factories live in `crowd_nav.domains`:

| Function | Role |
|----------|------|
| `load_domain(name)` | Import pack, return `DomainPack` |
| `make_score_fn_for_domain(pack, mode=, dataset_path=)` | Stage I score callable |
| `make_stage2_trainer_for_domain(pack, use_stub=)` | Stage II PolicyTrainer bridge |
| `make_stage3_trainer_for_domain(pack, use_stub=)` | Stage III PolicyTrainer bridge |
| `register_domain(name, module_path)` | Runtime registry (tests / plugins) |

Default: `--domain crowdnav` (see `scripts/run_evonav.py`).

## Minimum layout (real domain)

```text
domains/<name>/
  __init__.py      # from .pack import get_pack ; __all__ = ["get_pack"]
  pack.py          # get_pack(*, with_adapter=True) -> DomainPack
  prompts.py       # D1/D2/D3 templates + D5 seed source
  stage1.py        # make_score_fn(mode=, dataset_path=) -> (score_fn, dataset|None)
  adapter.py       # EnvAdapter + make_stage2_trainer / make_stage3_trainer
  spec.md          # task, state contract, metrics (for humans + prompt fill)
```

Reference implementation: **`crowdnav/`** (copy contracts from there; do not
weaken CrowdNav baseline behavior).

## Required `DomainPack` fields

`get_pack(*, with_adapter: bool = True) -> DomainPack` must set:

| Field | Meaning |
|-------|---------|
| `name` | Registry key; must match folder / `_REGISTRY` key |
| `display_name` | Human label (logged in manifest) |
| `prompts` | Module with CrowdNav-compatible formatter API (see below) |
| `seed_reward_source` | Sandbox-valid `compute_reward(state, memory)` source string |
| `spec_path` | Absolute/relative path to `spec.md` |
| `stage1_dataset_default` | Default offline dataset path (or `None` if N/A) |
| `make_score_fn` | Callable `make_score_fn(mode=, dataset_path=) -> (fn, dataset\|None)` |
| `adapter` | Optional `EnvAdapter` instance when `with_adapter=True` |
| `selection_scalar_name` | Document which scalar Stage II/III use (CrowdNav: `navigation_scalar`) |
| `metadata` | Free-form (env ids, policy name, paper refs, …) |

### Prompt module API (must match CrowdNav names)

Expose at least:

- `D1_SYSTEM_PROMPT`, `D3_SYSTEM_PROMPT`, `D5_SEED_FUNCTION`
- `format_d1_initial`, `format_d1_initial_batch`
- `format_d2_crossover`, `format_d2_mutation`
- `format_d3_refinement`, `format_d3_repair`

Task / state wording lives in those templates; keep `spec.md` aligned so
future prompt fill stays consistent.

### Stage I (`stage1.py`)

```python
def make_score_fn(*, mode: str = "dataset", dataset_path: str | None = None):
    # mode "smoke" → cheap fixture (optional)
    # mode "dataset" → load domain trajectories + analytical / other score
    return score_fn, dataset_or_none
```

CrowdNav uses Spearman Score1 over a fixed ORCA-like trajectory dataset
(`reward_search.scoring` / `rules` / `dataset`). Another domain may use a
different offline score, but the pipeline only needs a callable
`score_fn(reward_fn, *, candidate_id=) ->` float-like / `Score1Result`.

### Stage II/III (`adapter.py`)

Pipeline does **not** import CrowdSim trainers directly. It calls:

```python
make_stage2_trainer_for_domain(pack, use_stub=...)
make_stage3_trainer_for_domain(pack, use_stub=...)
```

For a new domain, either:

1. Extend `make_stage*_trainer_for_domain` in `domains/__init__.py` with a
   `pack.name == "<name>"` branch, **or**
2. Prefer a pack-local factory and teach `__init__.py` to call it (same pattern
   as CrowdNav).

Trainer objects must match the existing Stage runner contracts:

- Stage II: `train_and_eval(candidate, *, round_index, config) -> ProxyMetrics`
- Stage III: same → `TrainEvalBundle`, plus
  `evaluate_at_human_counts(candidate, bundle, *, config, human_counts=) -> HumanSweepReport`
  (or a domain-appropriate sweep; Stage3Runner expects this API today)

`use_stub=True` is only for `--fast` / unit tests (deterministic no-op). Real
runs use the domain’s real trainer.

## Registration checklist

1. Create `crowd_nav/domains/<name>/` with the layout above.
2. Add `"<name>": "crowd_nav.domains.<name>"` to `_REGISTRY` in
   [`__init__.py`](__init__.py).
3. Extend `make_stage2_trainer_for_domain` / `make_stage3_trainer_for_domain`
   (and rely on `pack.make_score_fn` for Stage I).
4. Run: `python scripts/run_evonav.py --domain <name> --fast ...` once wiring
   works; then non-fast with real LLM/GPU as needed.
5. Add pack-specific tests; **never** change CrowdNav prompt bytes or Score1
   lock tests unless intentionally revising the baseline.

Runtime alternative (without editing `_REGISTRY`):

```python
from crowd_nav.domains import register_domain, load_domain

register_domain("myenv", "my_package.domains.myenv")
pack = load_domain("myenv")
```

## `spec.md` contents (recommended)

Write for humans and for future LLM prompt assembly:

1. **Task** — goal of the agent / episode
2. **State contract** — exact fields a sandboxed reward may read
3. **Design principles** — what a good reward should encourage / penalize
4. **Metrics** — SR/CR/… or domain equivalents; selection scalar
5. **Env / policy** — Gym id(s), observation mode, default algo (A2C/PPO/…)
6. **Pack layout** — pointer to local files

CrowdNav example: [`crowdnav/spec.md`](crowdnav/spec.md).

## Baseline rule (non-negotiable)

- Default domain remains **`crowdnav`**.
- New domains are **additive**. They must not change CrowdNav Algorithm 1
  numbers, prompts, or Score1 padding/degeneracy behavior.
- Regression anchors:
  - `crowd_nav/reward_search/tests/test_domain_pack_baseline.py`
  - `crowd_nav/reward_search/tests/test_prompt_schema_regression.py`
  - `crowd_nav/reward_search/tests/test_score1_baseline_lock.py`

## What is intentionally not here

- No second domain package / stub folder in this repo yet
- No AMFRS (Pareto / archive) inside packs
- Physical Score1/rules code for CrowdNav stays under `reward_search/` (lock);
  the pack only owns the **resolution** entry point

## Quick verify (CrowdNav)

```bash
cd evonav_env
pytest crowd_nav/reward_search/tests/test_domain_pack_baseline.py -q
python scripts/run_evonav.py --domain crowdnav --fast --output-dir results/evonav_fast
```

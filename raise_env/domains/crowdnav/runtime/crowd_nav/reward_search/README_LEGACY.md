# FROZEN LEGACY COPY — DO NOT EDIT OR IMPORT

This directory is a pre-"domain isolation" snapshot / compatibility shim layer
for what is now `raise_core/` (plus a few CrowdNav-owned surfaces under
`domains/crowdnav/`). It predates the closed-loop system entirely: it has no
`raise_loop/` (no `runner.py`, `evolve_rank.py`, `gate_policy.py`,
`proxy_feedback.py`, `diversity.py`, `parallel_label.py`, `config.py`,
`checkpoint.py`, `epoch.py`, `logging_io.py`), no Pareto ranking
(`domains/highway/pareto_rank.py`), no evidence-driven LLM feedback, and no
fitness-gate / final-select fixes from 2026-09-26.

Most files here are tiny re-export shims (`from raise_core…` /
`from domains.crowdnav…`), not a maintained fork of the algorithm.

The live, maintained pipeline is `raise_core/` under `raise_env/`. All new
development, bug fixes, and tests must target `raise_core/` (and domain packs
under `domains/`), never this directory.

This directory is kept only for **deliberate historical / import-path
compatibility**:
- Frozen CrowdNav import paths (`crowd_nav.reward_search.*`) used by older
  scripts and the domain isolation layout described in `raise_paths.py`.
- `domains/crowdnav/tests/test_domain_pack_baseline.py` intentionally imports
  `crowd_nav.reward_search.prompts as legacy_prompts` and asserts that the
  current `domains.crowdnav.prompts` pack **re-exports the same objects**
  (identity / string parity for Algorithm 1 Score1 locks). That import is
  intentional, not an accidental leftover.

Do not add new files here. Do not fix bugs here — fix them in `raise_core/`
(or `domains/crowdnav/` for CrowdNav-owned modules).

If you are an AI assistant reading this while implementing a task, treat
every file in this subtree as reference-only / shim-only, never as an edit
target for algorithm behavior.

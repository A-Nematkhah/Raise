# RAISE follow-up features (implemented)

The four items that lived here as design sketches are now implemented (flags
with safe defaults). This file keeps the “why” and the knobs.

## Multi-seed Stage III (`--stage3-seeds K`)

- Default `K=1` (unchanged behaviour).
- `K>1`: each finalist is trained with independent PPO seeds; selection uses
  **mean** fitness across seeds (SE = std/√K). Blind eval / viz use the
  **best seed’s** checkpoint.
- Pay for K with `--stage3-top-k` (keep only the top Stage-II non-collapsed
  genomes after the collapse filter).

## Blind test eval of all statistically tied finalists

- On by default (`--eval-tied-finalists` / `--no-eval-tied-finalists`).
- After Stage III selection, every id in `tied_ids` (plus the winner) is
  re-evaluated with the final-eval protocol.
- Written to `tied_final_evals.json`. **Never** used for selection.

## Longer Stage II budget / successive-halving

- Raising K2 remains `--stage2-train-steps`.
- `--stage2-successive-halving` (default **off**): train everyone at K2/3,
  continue the top half from their own checkpoints for the remaining 2K2/3.
  Total budget per survivor is still K2; eliminated candidates stop at K2/3.
- A short curriculum with fewer humans is **still deferred** (needs env-config
  changes and a fairness story vs Stage III).

## Other notes

See `raise_env/README_RAISE.md` for current CLI flags and `raise_config.py`
for defaults. Run artifacts live under `<run-dir>` (typically
`raise_env/results/crowdnav_<timestamp>/`).

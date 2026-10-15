# Deferred design notes (post crowdnav_16h)

These items are intentionally **not** implemented yet. They need a design
decision or a multi-seed budget before coding.

## Multi-seed Stage III (`--stage3-seeds K`)

**Why.** Stage III fitness of the five non-collapsed finalists in
`crowdnav_16h` was 0.114–0.142 (n=100) and statistically tied. Each reward
was trained with one PPO seed, so seed variance confounds reward ranking.

**Sketch.** For each selected finalist, train K independent PPO seeds
(default K=1 keeps current behaviour). Report mean ± SE across seeds; select
the finalist with the best mean. Pay for K by taking only the top-k
non-collapsed Stage II candidates (collapse filter already exists).

## Blind test eval of all statistically tied finalists

**Why.** Only the Stage III argmax was evaluated on the 500-episode test
split. Four other finalists were within 2 SE of the winner; another tied
candidate could have similar test fitness.

**Sketch.** After Stage III selection, re-evaluate every id in `tied_ids`
(plus the winner) with the final-eval protocol. Write results under
`tied_final_evals.json`. **Never** use these numbers for selection —
report only.

## Longer Stage II budget / curriculum

**Why.** Training-rollout SR for `cro_0037` was still rising (~0.10 → 0.50)
over 150k steps. `mut_0039` had SR=0 / TR=0.9 at Stage II but Stage III
fitness 0.116 after 5M steps — timeout-heavy Stage II reflects under-training,
not collapse.

**Sketch.** Options: raise K2; successive-halving (train all at K2/3, continue
the top half from their own checkpoints); or a short curriculum that starts
with fewer humans. Keep the Stage II / Stage III comparison fair (same
warm-start policy for every candidate).

## Exclude collapsed candidates from mutation parents

**Why.** `mutation_parents=lower` currently mutates the bottom half, which
often includes CR≥0.8 collision-collapsed rewards. Their children frequently
collapsed too (epochs 5–7).

**Sketch.** Flag `--mutation-exclude-collapsed` (default off for
compatibility): when building the mutation parent pool, drop candidates with
Stage II holdout CR ≥ `collapse_cr`. If the filtered pool is empty, fall
back to the unfiltered lower half.

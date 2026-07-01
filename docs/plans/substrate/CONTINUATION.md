# Continuation spec — substrate self-modeling loop

Execution layer on top of `2026-06-27-self-modeling-loop-ladder.md`.

## State — ALL RUNGS BUILT (2026-07-01)

- Rungs 1, 2 (keystone + reducer lanes + runtime bridge + dynamics tick) and 6:
  merged to `main` earlier.
- Rungs 3, 4, 5: on branch `feat/substrate-rungs-3-4-5` (pushed to origin),
  PR body in `PR_substrate_rungs_3_4_5.md`. All flags default-off; nothing is
  enabled anywhere yet.

## Remaining work (post-merge, operational)

1. Create the GitHub PR from `PR_substrate_rungs_3_4_5.md` (no gh auth in this
   checkout; SSH push works, PR objects can't be created programmatically).
2. Apply migrations `manual_migration_substrate_episodes_v1.sql` and
   `manual_migration_attention_broadcast_v1.sql`.
3. Enable rung 4 then rung 3 flags per the rollout order in the PR body.
4. Rung 5 (`ORION_ENDOGENOUS_CURIOSITY_ENABLED`) stays OFF until explicit
   operator sign-off. Kill switch: `ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH`.

## Environment gotchas (still true)

1. A concurrent automation process drives git in this checkout — always work
   in an isolated worktree, never the main checkout.
2. Run pytest from inside the worktree with the main checkout's venv:
   `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest …` (cwd = worktree).
3. Two pre-existing failures in `services/orion-substrate-runtime/tests/`
   (`test_quarantine_truth`, `test_worker_independent_reducers`) also fail on
   `main`; `test_grammar_consumer_integration.py` needs live postgres.
4. No GitHub auth (no gh login, no `GH_TOKEN`).

## Pattern that works here

Each rung = one pure/ctx function + a focused unit test, committed small,
pushed to the branch. Adapters degrade to `None`/`[]` (never raise) on absent
input. Reuse existing accumulators/factories/engines rather than adding
parallel machinery — every rung rode an existing seam (the BFS propagator, the
unifier registry, the mutation pressure accumulator, `select_actions`, the
frontier decision path).

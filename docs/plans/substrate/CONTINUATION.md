# Continuation spec — substrate self-modeling loop

Execution layer on top of `2026-06-27-self-modeling-loop-ladder.md`. Lets a fresh
session finish the remaining rungs without re-deriving context.

## State
- Branch `feat/substrate-self-modeling-loop` (on origin).
- Done + tested (12/12): rung 1 (`prediction_error→pressure`), rung 2 keystone
  (`self_state` lane), rung 6 (`mutation_self_revision.py`).

## Environment gotchas (read first)
1. A concurrent automation process drives git in this checkout (merges PRs,
   switches branches, `pull origin main`) and has moved branches off their commits.
   **Always work in an isolated worktree**, never the main checkout:
   `git worktree add <scratch>/wt feat/substrate-self-modeling-loop`
   (run `git worktree prune` first if a stale registration blocks it).
2. **Run pytest from inside the worktree** (`cd <wt>`), else `orion` resolves to the
   main checkout. `python` isn't on PATH — use `.venv/bin/python`.
3. Edits to `pressure.py`/`dynamics.py` were reverted twice by something external;
   re-verify with `grep` after editing critical files.
4. Stray root-owned dir `services/orion-cortex-exec/services/` blocks `git stash -u`/
   clean; ignore it (needs `sudo rm`).
5. No GitHub auth (no gh login, no `GH_TOKEN`). SSH `git push` works; PR objects
   cannot be created/edited programmatically until a token exists.

## Remaining work, priority order

### A. Rung 1 runtime bridge (first, small)
Dynamics engine seeds from `node.metadata['prediction_error']`, but the runtime
worker only writes prediction error into a field-digester receipt, not onto a
durable substrate node.
- File: `services/orion-substrate-runtime/app/worker.py` (~555, ~645,
  `_prediction_error_receipt`).
- Do: on `error>0`, upsert/refresh a durable node (`node:substrate.execution` /
  `node:substrate.transport`) with `metadata['prediction_error']=error`,
  `observed_at=now`. Needs the worker to reach the substrate graph store (today it
  has only `BiometricsSubstrateStore` SQL). If that boundary is too big, route via
  the rung-2 adapter path and document.
- Accept: surprising batch → node `prediction_error` set → next
  `SubstrateDynamicsEngine.tick()` raises its `dynamic_pressure`.

### B. Rung 2 remainder — reducer lane adapters (mechanical)
Mirror `orion/substrate/relational/adapters/self_state_ctx.py`.
- New: `adapters/execution_ctx.py`, `transport_ctx.py`, `biometrics_ctx.py` —
  `map_*_ctx_to_substrate(ctx) -> SubstrateGraphRecordV1 | None`, carrying
  prediction_error / pressure hints into node metadata. Read from ctx, or DB-pull
  like `self_study.py` if the projection isn't in ctx.
- Register 3 `ProducerEntryV1` in
  `orion/cognition/projection_builder.py::build_projection_unification_registry`
  (tier `SNAPSHOT_EPHEMERAL`, short TTL).
- Accept: adapter unit tests; registry lists all 12 producers.

### C. Rung 4 — episodic continuity (safe, fully unit-testable; recommended next)
- New: `orion/substrate/episodic_consolidation.py` — evaluator alongside
  `GraphConsolidationEvaluator`, windowed by `window_seconds`, rolling a **capped**
  list of reduction receipts → one `EpisodeSummaryV1` (proposal-marked).
- New schema `EpisodeSummaryV1` under `orion/core/schemas/`.
- Rules: output marked proposal → pending/generated; cap receipts per episode;
  episode id derived from inputs (idempotent replay).
- Accept: window → one queryable episode; replay idempotent; excluded from
  execution context by default; cap enforced.

### D. Rung 3 — continuous broadcast (bigger; new runtime loop)
- Files: `orion/substrate/attention_frame.py`, `attention/*`,
  `services/orion-substrate-runtime/app/worker.py`,
  `orion/schemas/attention_frame.py`.
- Do: build `AttentionFrameV1` from unified beliefs + active pressure every tick
  (not just chat turns); persist selected coalition as a projection. New flag
  `ORION_ATTENTION_BROADCAST_ENABLED` (default false). Reuse `select_actions`.
- Accept: flag on → one selected coalition per tick; high-pressure (incl.
  prediction-error) nodes win; flag off → identical to today.

### E. Rung 5 — endogenous agency (build flag-gated; DO NOT enable without sign-off)
- Files: `orion/substrate/frontier_curiosity.py`, `frontier_expansion.py`, settings.
- Do: curiosity seed source reading intrinsic signals (sustained prediction error,
  `appraisal/repair_pressure.py`, rung-3 open-loops) emitting `curiosity_candidate`s
  without operator trigger — behind `ORION_ENDOGENOUS_CURIOSITY_ENABLED`
  (default false), per-cycle budget cap + kill switch. Proposal-only; routes through
  rung-6 governance.
- Accept: flag on → bounded candidate set; budget cap + kill switch enforced; flag
  off → today's operator-gated behaviour.

## Sequencing
A → B → C → D → E. A/B/C are safe and unit-testable. D adds a runtime loop. E is
the only one that changes autonomy at runtime — ship code, leave flag off, get
explicit approval before flipping it.

## Pattern that works here
Each rung = one pure/ctx function + a focused unit test, committed small, pushed to
the branch. Adapters degrade to `None` (never raise) on absent input. Reuse existing
accumulators/factories/engines rather than adding parallel machinery — every rung so
far rode an existing seam (the BFS propagator, the unifier registry, the mutation
pressure accumulator).
</content>

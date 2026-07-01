# PR: substrate self-modeling loop — rungs 3, 4, 5

Branch: `feat/substrate-rungs-3-4-5` → `main`
(PR body kept in-repo because this checkout has no GitHub auth; create the PR
from this file.)

## Summary

Completes the self-modeling-loop ladder
(`docs/plans/substrate/2026-06-27-self-modeling-loop-ladder.md`). Rungs 1, 2
and 6 landed earlier; this PR adds the remaining three, each default-off,
fail-open, and unit-tested. **No flag in this PR is enabled anywhere.**

### Rung 4 — episodic continuity (`3cc6d29e`)
- `EpisodeSummaryV1` (`orion/core/schemas/substrate_episodes.py`): proposal-marked
  rollup of one time-window of reduction receipts.
- `EpisodicConsolidationEvaluator` (`orion/substrate/episodic_consolidation.py`):
  pure, windowed, idempotent (episode id derived from inputs), receipts per
  episode hard-capped at 64.
- Runtime tick in `orion-substrate-runtime` consolidates the last *completed*
  clock-aligned window; `ON CONFLICT DO NOTHING` insert; bounded retention
  (default 14 days). Flag: `SUBSTRATE_EPISODIC_TICK_ENABLED`.
- Migration: `manual_migration_substrate_episodes_v1.sql`
  (`substrate_episode_summaries`).

### Rung 3 — continuous attention broadcast (`0c2efd93`)
- `orion/substrate/attention_broadcast.py`: runs the existing workspace
  competition (`merge_signals` → `build_open_loops` → `select_actions`, no new
  selection policy) over substrate graph nodes carrying `dynamic_pressure`
  (rung 1) and `prediction_error`, plus rung-2 belief-derived nodes.
- `max_asks=0`: the broadcast never generates chat questions and takes no
  action; the winning coalition is persisted as a single-row projection
  (`AttentionBroadcastProjectionV1` → `substrate_attention_broadcast_projection`)
  other organs can query. Flag: `ORION_ATTENTION_BROADCAST_ENABLED`.
- Chat-scoped `build_attention_frame` path untouched; flag off ⇒ behaviour
  identical to today.
- Migration: `manual_migration_attention_broadcast_v1.sql`.

### Rung 5 — endogenous curiosity seeds (`1a9ce17f`) — **DO NOT ENABLE without sign-off**
- `orion/substrate/endogenous_curiosity.py`: intrinsic signals (sustained
  prediction error, elevated repair-pressure appraisals, unresolved rung-3
  open-loops) seed `curiosity_candidate` signals with no operator trigger.
- Seeds ride the existing `FrontierCuriosityEvaluator` decision/plan path via
  a new optional `endogenous_signals` parameter — no new invocation
  authority; strict/autonomy zone guardrails unchanged; candidates target
  `concept_graph` only.
- Guardrails: `ORION_ENDOGENOUS_CURIOSITY_ENABLED` default false; kill switch
  `ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH` beats the enable flag; per-cycle
  budget hard-capped at 8 in code. Signals only — change proposals still go
  through rung-6 trials + rollback governance.

## Acceptance checks (all covered by tests)

- Rung 4: window → one queryable episode; replay idempotent; proposal-marked;
  cap enforced; empty window → no write; fails open. (`test_episodic_consolidation.py`,
  `test_worker_episodic_tick.py`)
- Rung 3: flag on → one selected coalition per tick; high-pressure and
  prediction-error nodes beat calm ones; projection queryable; no questions
  generated; flag off → no-op. (`test_attention_broadcast.py`,
  `test_worker_attention_broadcast_tick.py`)
- Rung 5: flag on → bounded candidate set; budget + hard ceiling + kill switch
  enforced; flag off → today's operator-gated behaviour; weak seeds decide to
  noop, strong seeds to invoke through the unchanged policy.
  (`test_endogenous_curiosity.py`)

## Test status

- `orion/substrate/tests/`: 127 passed.
- `services/orion-substrate-runtime/tests/`: 48 passed;
  `test_quarantine_truth.py::test_truth_healthy_when_quarantine_acknowledged`
  and `test_worker_independent_reducers.py::test_start_spawns_independent_reducer_poll_tasks`
  fail identically on `main` (pre-existing, stale expectations after the chat
  reducer landed); `test_grammar_consumer_integration.py` needs live postgres.

## Rollout order (when ready)

1. Apply the two migrations.
2. `SUBSTRATE_EPISODIC_TICK_ENABLED=true` (safe: bounded writes + retention).
3. `ORION_ATTENTION_BROADCAST_ENABLED=true` (needs the shared sparql store,
   same as the dynamics tick).
4. Rung 5 stays **off** pending explicit operator sign-off.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

# PR: heartbeat ignition — retention + first workspace consumers

Branch: `feat/heartbeat-ignition-v1` → `main`
(PR body kept in-repo because this checkout has no GitHub auth; create the PR
from this file.)

Plan: `docs/superpowers/plans/2026-07-02-heartbeat-ignition-retention-and-workspace-consumers.md`

## Summary

Makes the always-on substrate heartbeat safe to run and gives the Global
Workspace broadcast its first audience. Follows heartbeat-pacemaker-v1
(PR #769) and the self-modeling-loop rungs 3-4-5 (PR #776), whose flags are
all still off in production — partly because the pacemaker chain's tables
had no retention (the same unbounded-write pattern behind the earlier
postgres TOAST-OOM host freeze), and because nothing consumed the rung-3
broadcast or rung-4 episodes.

### Retention (`a326eecd`, `cf03d32b`, `665d37ef`)
- Batched (`LIMIT :batch_size`, 5000/loop), guard-railed prune loops in
  `orion-field-digester` (`substrate_field_state`), `orion-attention-runtime`
  (`substrate_attention_frames`), and `orion-self-state-runtime`
  (`substrate_self_state`, `self_state_predictions`, `identity_snapshots`).
- Every prune structurally excludes the newest row (by `generated_at`, the
  same ordering the latest-row readers use), so `load_latest_*` can never
  observe an empty table even when the writer is paused.
- Defaults ON (protective): 72h retention, hourly cadence. `*_RETENTION_HOURS=0`
  disables. New env vars: `FIELD_STATE_RETENTION_HOURS`,
  `ATTENTION_FRAME_RETENTION_HOURS`, `SELF_STATE_RETENTION_HOURS`
  (+ matching `*_PRUNE_INTERVAL_SEC`).

### Felt-state reader lanes (`5dd01280`)
- `LaneSpec` gains a per-lane `max_age_sec` override (None → global 120s).
- New lanes: `ctx["attention_broadcast"]` (single-row projection
  `substrate.attention.broadcast.v1`) and `ctx["episode_summary"]` (latest
  `substrate_episode_summaries` row, `max_age_sec=1800` because episodes are
  15-minute windows the global gate would always reject).

### Workspace consumers (`7a4cc484`, `83eab9ea`, `2f1a20e8`)
- `attention_ctx.py`: rung-3 consumer — maps the broadcast winner to one
  `attending:current_focus` belief node (attended ids capped at 8); returns
  `None` when nothing is attended. The broadcast now has an audience.
- `episodes_ctx.py`: rung-4 consumer — maps the latest `EpisodeSummaryV1` to
  one proposal-marked `episode:latest` node (organ counts top-5, notes ≤4,
  salience = receipt_count/64 clamped). Episodes inform stance; they never
  become accepted truth.
- Both registered in `build_projection_unification_registry` (12 → 14
  producers, `SNAPSHOT_EPHEMERAL`, ctx-sourced, no cold fan-out). Both
  adapters degrade to `None` on absent/garbage input — never raise.

## Test status

- `services/orion-field-digester/tests/`: 11 passed (3 new).
- `services/orion-attention-runtime/tests/`: 3 passed (new suite + conftest).
- `services/orion-self-state-runtime/tests/`: 3 passed (new suite + conftest).
- cortex-exec `test_felt_state_reader_new_lanes.py` + pre-existing
  `test_substrate_felt_state_reader.py`: 11 passed. (Whole cortex-exec dir has
  pre-existing collection errors on `main`, unrelated.)
- `orion/substrate/relational/tests` + `orion/cognition`: 81 passed
  (8 new adapter tests; registry-shape test updated 12 → 14).

## Rollout runbook (operator, after merge + deploy)

0. **DB identity check:** confirm substrate-runtime `POSTGRES_URI` and
   cortex-exec `SUBSTRATE_FELT_STATE_DATABASE_URL` reach the same database:
   `psql "$POSTGRES_URI" -c "\dt substrate_*"` must show
   `substrate_attention_broadcast_projection` and `substrate_episode_summaries`
   from the DSN cortex-exec reads, else the new lanes are silently empty.
1. Apply migrations:
   `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_substrate_episodes_v1.sql`
   `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_broadcast_v1.sql`
2. Baseline before the 2s heartbeat:
   `psql "$POSTGRES_URI" -c "SELECT pg_size_pretty(pg_total_relation_size('substrate_field_state')), (SELECT count(*) FROM substrate_field_state)"`
   — re-check after 24h of idle tick; growth must be bounded by retention.
3. `SUBSTRATE_EPISODIC_TICK_ENABLED=true` (orion-substrate-runtime); restart;
   after ≥15 min expect one row in `substrate_episode_summaries`.
4. Deploy the retention services; expect `field_state_pruned` /
   `attention_frames_pruned` / `self_state_history_pruned` log lines within
   one prune interval wherever >72h-old rows exist.
5. `FIELD_DIGESTER_IDLE_TICK_ENABLED=true` (orion-field-digester); restart;
   with all chat/biometrics quiet >30s, `substrate_field_state.tick_id`,
   `substrate_attention_frames`, and `substrate_self_state` keep advancing
   ~every 2s.
6. Confirm `SUBSTRATE_STORE_BACKEND=sparql` + Fuseki reachable, then
   `SUBSTRATE_DYNAMICS_TICK_ENABLED=true` and
   `ORION_ATTENTION_BROADCAST_ENABLED=true`; expect
   `substrate_dynamics_tick_completed` log lines and a fresh
   `substrate_attention_broadcast_projection` row.
7. End-to-end: one Hub chat turn; the unified belief set should contain
   `attending:current_focus` and `episode:latest` nodes for the `orion` anchor.
8. `ORION_ENDOGENOUS_CURIOSITY_ENABLED` **stays false** — rung 5 needs
   explicit operator sign-off (kill switch:
   `ORION_ENDOGENOUS_CURIOSITY_KILL_SWITCH`).

## Non-goals

- No rung-5 enablement, no changes to rung-3/4/5 engine logic, no
  `chat_stance.py` projector for `attending:*` (follow-up), no pruning of
  `substrate_episode_summaries` (rung 4 has its own retention) or
  `substrate_reduction_receipts` (owned by `receipt_pruner.py`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

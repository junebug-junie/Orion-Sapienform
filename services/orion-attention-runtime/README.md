# orion-attention-runtime

Layer 5 substrate service: polls latest `FieldStateV1` from Postgres and builds deterministic `FieldAttentionFrameV1` snapshots.

## Behavior

- Polls `substrate_field_state` every `ATTENTION_POLL_INTERVAL_SEC` (default 2s)
- Skips if an attention frame already exists for the latest field `tick_id` (idempotent)
- Persists to `substrate_attention_frames`
- Does **not** publish bus events or mutate field state
- Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
  `HEARTBEAT_INTERVAL_SEC` (default 10s), on its own independent bus connection -- otherwise
  this service has no other bus traffic (Postgres-poll only worker)

## Prerequisites

Apply migrations:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_attention_frame_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_node_prediction_error_baseline_v1.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_goal_provenance_streak_v1.sql
```

Note: `load_node_dominance_streak`/`save_node_dominance_streak` (below) degrade silently
on any DB error, including a missing table -- skipping this migration does not crash the
service, it just silently keeps the node-target dominance streak cold on every restart,
which is exactly the bug the 2026-07-31 fix below exists to remove. Apply it.

Requires `orion-field-digester` (or equivalent) writing `substrate_field_state`.

## Candidate A precision-weighted salience: persisted EWMA baseline (2026-07-30 fix)

Each of the five `node:substrate.*` targets in `PREDICTION_ERROR_NATIVE_TARGETS`
(`orion/attention/field_attention/selectors.py`) gets its own persisted, incrementally-
updated running baseline in `substrate_node_prediction_error_baseline`
(`AttentionRuntimeStore.advance_node_prediction_error_baseline`), advanced by exactly one
real new `substrate_reduction_receipts` row at a time. This replaced a per-tick
recompute over that table's own ~30-minute retention window, which let a target with as
few as 2 real samples surviving the window win a fully-confident-looking
`salience_score=1.0` -- see `orion/attention/field_attention/candidate_precision_weighted.py`'s
module docstring and `orion/sentience_striving_program/README.md` section 12 for the full
live-incident record. `observation_count` on the persisted baseline is a real cumulative
count of every receipt this target has ever incorporated, immune to that retention prune.

## Node-target dominance streak: restart persistence (2026-07-31 fix)

`orion.attention.field_attention.goal_provenance.DominanceStreak` (the consecutive-real-tick
counter gating whether a node-target goal-provenance record gets emitted at all) is persisted
to `substrate_goal_provenance_streak` via `AttentionRuntimeStore.load_node_dominance_streak`/
`save_node_dominance_streak`, lazy-loaded once on the worker's first real tick instead of
always starting cold. Previously this streak lived only in-process, resetting to count=0 on
every restart -- an accepted gap when its only consumer was an internal emit-debounce, but
no longer acceptable once a real, still-unimplemented downstream consumer (a design doc,
PR #1543) proposed surfacing this exact count directly into a real LLM-facing prompt. See
`orion/sentience_striving_program/README.md` section 14 for the full incident record.

## Streak-tick telemetry: min_streak calibration (2026-08-11, Part H)

`ORION_GOAL_PROVENANCE_MIN_STREAK`'s value (default `3`) is an unmeasured, disclosed
placeholder debounce. To calibrate it against the true streak-length distribution --
`orion:memory:goals:proposed` alone is a censored sample that only ever shows streaks that
already survived the debounce -- this worker also publishes `DominanceStreakTickV1` on
`orion:debug:attention:streak_tick` on EVERY real tick (not just qualifying emissions),
gated by `ORION_GOAL_PROVENANCE_STREAK_TICK_TELEMETRY_ENABLED` (default `true`,
independent of the main producer). `orion-sql-writer` persists these to
`goal_provenance_streak_ticks` (bounded by `GOAL_PROVENANCE_STREAK_TICKS_RETENTION_DAYS`,
default 14 days, applied at that service's boot). Once a few days of real data have
accumulated, run `python scripts/analysis/measure_goal_provenance_streak_distribution.py`
from repo root to see the real streak-length distribution and candidate `min_streak`
qualification rates. Meant to be temporary: once calibration is done, retire the channel.

## Run

```bash
cp .env_example .env
docker compose up -d --build
curl -s http://localhost:8117/health
curl -s http://localhost:8117/latest | jq .
```

## Smoke

From repo root:

```bash
./scripts/smoke_attention_frame_v1.sh
```

## Health monitor

A background health monitor (`ATTENTION_RUNTIME_HEALTH_CHECK_INTERVAL_SEC`, default 900s) watches `substrate_attention_frames`'s oldest row: if it exceeds `ATTENTION_FRAME_STALL_MULTIPLIER` (default `1.5`) x `ATTENTION_FRAME_RETENTION_HOURS`, the hourly pruner may have stopped running. Staleness is keyed on `created_at` -- the same column the prune SQL's cutoff filters on -- so the two can never disagree about what "age" means.

The check is edge-triggered: an alert (via `orion-notify`'s `POST /attention/request`, surfacing in Hub's existing Pending Attention panel) fires only on a healthy->unhealthy transition, plus a lower-severity recovery note on the way back, so a persisting condition does not spam a fresh attention item every check. On worker restart mid-incident, it first checks `orion-notify` for an already-open alert for this service+reason before firing a duplicate. If `orion-notify` is unreachable at the exact moment of a transition, the alert retries every subsequent tick until delivery is actually confirmed -- it is never silently dropped.

Mirrors the identical pattern in `orion-field-digester` (`app/health_monitor.py`), adapted to this service's single table.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `ATTENTION_RUNTIME_HEALTH_CHECK_INTERVAL_SEC` | `900.0` | Health-monitor check cadence |
| `ATTENTION_FRAME_STALL_MULTIPLIER` | `1.5` | Alert if `substrate_attention_frames`'s oldest row exceeds this x retention hours |
| `NOTIFY_BASE_URL` | `http://orion-athena-notify:7140` | `orion-notify` base URL for health-monitor attention alerts |
| `NOTIFY_API_TOKEN` | (empty) | `orion-notify` auth token, if configured |
| `ORION_GOAL_PROVENANCE_STREAK_TICK_TELEMETRY_ENABLED` | `true` | Publish `DominanceStreakTickV1` on every real tick (see above) |
| `CHANNEL_GOAL_PROVENANCE_STREAK_TICK` | `orion:debug:attention:streak_tick` | Streak-tick telemetry channel |

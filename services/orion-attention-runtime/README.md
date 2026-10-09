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
  < services/orion-sql-db/manual_migration_node_prediction_error_baseline_v2_definition_version.sql
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_node_prediction_error_baseline_v3_last_value_observed_at.sql
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

**Definition-version reset (2026-09-25).** A baseline only describes the formula that
produced its numbers. `orion/schemas/prediction_error_definitions.py` holds the live
formula version per `reducer_key`; the substrate runtime stamps it on every prediction-error
receipt (`after.definition_version`, unstamped = 1). When a row's stored `definition_version`
differs from the live one, the baseline restarts cold (cursor kept) and only receipts carrying
the live version are folded, so receipts the old producer wrote before a deploy never seed the
new baseline. `route_arbitration` and `chat_session` moved to v2 on 2026-09-25 (route now
averages only over runs a batch touched; chat dropped `topic_coherence`). Look for
`node_prediction_error_baseline_definition_reset` in the logs. Without the v2 migration the
store keeps the old behaviour and logs
`node_prediction_error_baseline_definition_version_column_missing`.

**Staleness fade (2026-09-29).** A domain's last reading only refreshes when its reducer
writes a receipt, and chat writes one only when a turn lands, so a quiet domain's last
reading used to stay its "current" error for hours and win the node competition. The
current error is now faded linearly to 0 over 30 minutes
(`PREDICTION_ERROR_STALENESS_HORIZON_SEC`, the same horizon as
`PressureConfig.prediction_error_decay_horizon_seconds`), measured from the receipt behind
`last_value`, persisted as `last_value_observed_at` (not the `last_receipt_created_at`
cursor, which also moves over skipped receipts; the cursor is only a fallback before the
v3 migration). Read-side only: the EWMA baseline still folds only
real receipt values. A faded target's reasons say `stale reading: ... min old, weighted x`.
`chat_session` moved to definition v3 the same day (touched turns only), so its baseline
restarts once more on deploy. Replay: `scripts/analysis/replay_candidate_a_staleness_fade.py`.

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

**The one bridge (2026-09-06).** Before choosing a goal target, `_maybe_build_goal` reads
which node ids the substrate's workspace competition currently holds as open loops
(`AttentionRuntimeStore.load_competing_loop_refs`, the `source_refs` of every loop in the
latest `substrate_attention_broadcast_projection`) and prefers, among the qualified
candidates, the highest-salience one the competition can actually see. The substrate honours
a goal only on an exact id match (`orion/substrate/attention/top_down.py::relevance`), so a
goal about a node that is not competing cannot be acted on -- `goal_matched_no_loop` was 37%
of self-model ticks in the 24h before this shipped. When no candidate is in the competition,
or the projection is missing/older than `ORION_GOAL_PROVENANCE_COMPETITION_MAX_AGE_SEC`, the
selector falls back to the plain top-1, so this never makes the producer emit fewer goals.
`ORION_GOAL_PROVENANCE_READS_COMPETITION=false` restores the pre-bridge behaviour exactly.
The producer logs what it saw on every emission
(`field_goal_provenance_competition_read ... competition_read=in_competition|not_in_competition|unavailable`);
it is deliberately not a schema field, because `FieldGoalProvenanceV1` is `extra="forbid"` on
three consumers and a producer-first deploy of new fields drops every goal until they are rebuilt. Design:
`docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md`, "The read side".

## Focus history

Each change of the **goal-provenance node winner** closes one `field_dominance_run`
row, including runs shorter than `ORION_GOAL_PROVENANCE_MIN_STREAK`. This is the
existing competition-aware internal-signal streak, **not** the frame's overall top
host/capability target, workspace attention, evidence of work, or a calm/arousal signal.
No reader changes Orion's decisions in this patch.

Rows contain target id/kind, start and end timestamps, observed tick count, the
threshold at run start, and first/last attention frame ids. `ended_at` is the next
winner/no-winner frame's timestamp; the last frame id belongs to the old target.
No-winner ticks close a run without opening another. The open tail is a checkpoint
in `substrate_goal_provenance_streak.run_state`, not a completed history row.
On success frame, checkpoint, existing goal debounce and completed row commit together.
Recorder failures roll back only the recording savepoint and log frame/tick ids;
attention still saves its frame and emits the same goals. Recovery detects skipped
recording ticks from saved frames, discards the incomplete open run, and logs the
gap rather than inventing uninterrupted focus. Duplicate frame ids cannot count
twice. Checkpoints survive restarts. Counts cover observed frames, not missed
polls; wall-clock spans may include downtime and must not be treated as continuous
activity. A run already underway at installation starts at the first observation
and is marked `left_censored`; exclude it from complete-duration estimates.

The recorder follows the existing producer enable/bus gates. Disabled/unavailable
producer periods are unobserved, not proof of idleness. The existing debounce
counter remains because it controls goal emission; the retired per-tick telemetry
producer, bus schema/channel, SQL-writer model/routes, retention setting and live
analysis query are removed. SQL-only `FieldDominanceRunV1` is registered as the row
contract. No replacement bus event or SQL-writer route exists.

Before deployment apply `services/orion-sql-db/manual_migration_field_dominance_run_v1.sql`.
After both services are updated and completed rows are verified, the separately
approved `manual_migration_retire_streak_tick_v1.sql` removes the old table (no
`CASCADE`). Never drop it while the old SQL writer can recreate it. Rollback:
revert/redeploy the code and retain new rows; the old writer can recreate its old
telemetry table, but deleted legacy history cannot be recovered without an export.

Inspect completed history:

```sql
SELECT target_id, started_at, ended_at, tick_count, left_censored,
       first_source_attention_frame_id, last_source_attention_frame_id
FROM field_dominance_run ORDER BY ended_at DESC LIMIT 30;
```

Read-only replay against an export made before retirement:

```bash
python services/orion-attention-runtime/evals/replay_focus_runs.py /tmp/orion-focus-history.jsonl
```

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

# Orion SQL Writer

The **SQL Writer** service is a durable consumer that subscribes to various bus channels and persists structured payloads into a relational database (PostgreSQL). It uses a configurable routing map to determine which SQLAlchemy model to use for each message kind.

## Contracts

### Consumed Channels
Configured via `SQL_WRITER_SUBSCRIBE_CHANNELS` (JSON list).

| Default Channel | Kind(s) | Target Table |
| :--- | :--- | :--- |
| `orion:tags:enriched` | `tags.enriched`, `collapse.enrichment` | `CollapseEnrichment` |
| `orion:collapse:sql-write` | `collapse.mirror` | `CollapseMirror` |
| `orion:chat:history:log` | `chat.history.message.v1` | `ChatMessageSQL` |
| `orion:chat:history:turn` | `chat.history`, `chat.log` | `ChatHistoryLogSQL` |
| `orion:chat:gpt:log` | `chat.gpt.message.v1` | `ChatGptMessageSQL` |
| `orion:chat:gpt:turn` | `chat.gpt.log.v1`, `chat.gpt.turn.v1` | `ChatGptLogSQL` |
| `orion:chat:gpt:message:log` | `chat.gpt.message.v1` | `ChatGptMessageSQL` |
| `orion:dream:log` | `dream.result.v1` (canonical), `dream.log` (legacy) | `Dream` |

**Dream persistence:** `dream.result.v1` payloads are validated as `DreamResultV1` and projected into `dreams`. Legacy `dream.log` + `DreamRequest` is still accepted and mapped into the same table (narrative from `context_text`). Extended telemetry lives under `metrics._dream_audit`.
| `orion:telemetry:biometrics` | `biometrics.telemetry` | `BiometricsTelemetry` |
| `orion:biometrics:summary` | `biometrics.summary.v1` | `BiometricsSummarySQL` |
| `orion:biometrics:induction` | `biometrics.induction.v1` | `BiometricsInductionSQL` |
| `orion:spark:telemetry` | `spark.telemetry` | `SparkTelemetrySQL` |
| `orion:vision:events:sql-write` | `vision.event.v1` | `VisionEventSQL` |
| `orion:autonomy:action:outcome` | `action.outcome.emit.v1` | `ActionOutcomeSQL` |
| `orion:debug:attention:streak_tick` | `debug.attention.streak_tick.v1` | `DominanceStreakTickSQL` |

**Action outcome persistence:** `action.outcome.emit.v1` (produced by `orion-spark-concept-induction` after an autonomous readonly fetch) is projected into `action_outcomes` (PK `action_id`, idempotent upsert). `orion-cortex-exec` reads it back per-subject for chat-stance action feedback. DDL is applied on boot (`app/main.py` lifespan) and also lives in `services/orion-sql-db/manual_migration_action_outcomes_v1.sql`.

**Goal-provenance streak-tick telemetry (2026-08-11, Part H, temporary):** `debug.attention.streak_tick.v1` (produced by `orion-attention-runtime` on every real field tick, not just qualifying `FieldGoalProvenanceV1` emissions) is projected into `goal_provenance_streak_ticks` (PK `tick_telemetry_id`, idempotent upsert) so `scripts/analysis/measure_goal_provenance_streak_distribution.py` can measure the true streak-length distribution and calibrate `ORION_GOAL_PROVENANCE_MIN_STREAK`. High-volume (~1 row per real field tick); bounded by `GOAL_PROVENANCE_STREAK_TICKS_RETENTION_DAYS` (default 14 days, applied at boot, same pattern as `DRIVE_AUDITS_RETENTION_DAYS` below). Meant to be temporary -- once calibration is done, this channel/table is a disclosed follow-up to retire.

**Drive audit persistence — REMOVED 2026-08-13:** `orion-spark-concept-induction`'s `DriveEngine` (the sole producer of `memory.drives.audit.v1`) was deleted outright 2026-07-30 (`chore/delete-orion-drives`, PR #1486). The `drive_audits` table was dropped 2026-08-13 (snapshotted first, `/tmp/drive_audits_drop_2026-08-13/`) alongside the Hub Drives Analytics tab that read it (docs/superpowers/pr-reports/2026-08-13-remove-hub-drives-analytics-tab-pr.md, which also removed the boot DDL). This service's write-path wiring for it (`DriveAuditSQL` model, `MODEL_MAP`/`INSERT_ONLY_MODELS`/route-map entries, the channel subscription, `DRIVE_AUDITS_RETENTION_DAYS` and its startup prune job) was fully untangled the same day (docs/superpowers/pr-reports/2026-08-13-untangle-drive-audit-sql-writer-pr.md) — an earlier scope note claiming `DriveAuditSQL` shared a `_JSONB` type declaration with other live models was checked and found wrong (every model declares its own private copy), so nothing blocked full removal. `scripts/drive_history_reflection_synthesis.py` and `scripts/analysis/measure_autonomy_gate.py` both still reference the concept but already degrade safely to "insufficient/missing data" — neither is on any cron/scheduler in this repo.

### Environment Variables
Provenance: repo root `.env` (mesh globals: `ORION_BUS_URL`, `PROJECT`, `NET`, …) → service `.env_example` → `docker-compose.yml` → `settings.py`

**Compose (from repo root):**
```bash
docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build
```

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `SQL_WRITER_SUBSCRIBE_CHANNELS` | (See above) | List of channels to subscribe to. |
| `SQL_WRITER_ROUTE_MAP_JSON` | (See above) | JSON mapping of `kind` → `ModelName`. |
| `SPARK_LEGACY_MODE` | `accept` | Legacy Spark handling: `accept`, `warn`, `drop`. |
| `SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL` | `false` | If `true`, append `orion:spark:state:snapshot` to subscriptions. |
| `POSTGRES_URI` | ... | Database connection string. |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |

### Deprecation Controls
`SPARK_LEGACY_MODE` controls how legacy Spark kinds are handled:
- `accept`: write legacy kinds normally (default).
- `warn`: write legacy kinds and emit a deprecation warning log.
- `drop`: skip legacy writes and emit a warning log.

Example logs:
- `SPARK_LEGACY_DEPRECATED kind=spark.introspection.log mode=warn action=accept_write`
- `SPARK_LEGACY_DEPRECATED kind=spark.introspection.log mode=drop action=skip_write`

`SQL_WRITER_ENABLE_SPARK_SNAPSHOT_CHANNEL` (default `false`) appends the snapshot channel
`orion:spark:state:snapshot` to subscriptions without altering existing lists.

Legacy spark introspection channels/kinds are disabled by default; you can re-add them via
`SQL_WRITER_SUBSCRIBE_CHANNELS` and `SQL_WRITER_ROUTE_MAP_JSON` if needed.

## Grammar/substrate table retention

`grammar_events`, `grammar_edges`, `grammar_atoms`, `substrate_organ_emissions`,
`grammar_traces` and `substrate_proposal_frames` all get bounded retention
(`*_RETENTION_DAYS` env keys, **default 3** for the grammar tables, **10** for
`substrate_proposal_frames`). Each is a batched `DELETE ... LIMIT batch_size` loop
(`GRAMMAR_EVENTS_RETENTION_BATCH_SIZE`/`_MAX_BATCHES_PER_STARTUP`/`_MAX_ELAPSED_SEC`,
shared across all six tables), capped so a huge backlog can't turn a pass into an
unbounded operation. See `app/grammar_truth.py`'s `_apply_bounded_table_retention()`.

Retention runs on a **60-second timer** (`GRAMMAR_RETENTION_INTERVAL_SEC`,
`app/grammar_retention_loop.py`), not only at startup. It used to be startup-only, which
deleted ~365,000 rows per process start against 1,117,440 rows/day of arrival -- it could
not converge against ANY window, and logged a growing `remaining_debt` nobody read.
Periodic cycles use much smaller caps (`GRAMMAR_RETENTION_PERIODIC_MAX_BATCHES`,
`_MAX_ELAPSED_SEC`) than the startup pass.

`grammar_events` and `grammar_traces` additionally respect a **cursor floor**: retention
refuses to delete rows any of the five reducer lanes still has unconsumed below the cutoff
(`_grammar_events_cursor_floor`). When it binds, `/health` reports
`cursor_floor_applied` and `remaining_debt` is measured against the retention window
rather than the clamped cutoff -- so "a reducer is stuck" and "retention is caught up"
cannot look the same.

`grammar_traces` is the parent row the Grammar Atlas lists, and is deliberately covered by
the periodic loop only, not by `main.py`'s startup pass. It was added last (2026-08-20):
until then its children were pruned at 3 days while the trace rows lived forever, so 42%
of traces expanded into empty graphs.

`substrate_proposal_frames` (2026-08-20) is the first stage of the substrate pipeline
(proposal -> policy decision -> execution dispatch -> feedback) and had **no bound at all**:
474,230 rows / 1,758 MB live, oldest 2026-07-23, growing ~27k rows and ~105 MB a day. It
uses its own floor, `_substrate_chain_floor`, not the grammar cursor floor -- three later
stages can reach back into it by `frame_id`, so the floor asks each stage's **pending
marker** (`policy_pending` / `dispatch_pending` / `feedback_pending`, from
`manual_migration_substrate_pending_markers.sql`) for the oldest row still owed work.
Anything at or above that timestamp survives regardless of age.

Two details in that floor are load-bearing and were both wrong in the first draft:

* **Every probe resolves back to the PROPOSAL row**, by joining through
  `source_proposal_frame_id` -- not to the pending row's own timestamp. A pending row in a
  downstream table is always *newer* than the proposal it needs (it is a child, written
  later), so flooring at the child's timestamp leaves the parent below the floor and
  deletable, which is backwards from the safety being claimed. Live: dispatch rows are
  created a mean 123.2s and max 920.2s after their parent proposal; 0 of 55,573 before it.
* **`OFFSET 0` fences the min-aggregate.** Written plainly, Postgres rewrites
  `SELECT MIN(created_at) ... WHERE dispatch_pending` into `ORDER BY created_at LIMIT 1`
  over the *full* `created_at` index with the marker as a filter. Pending rows are always
  the newest rows, so the ascending scan discards the whole table first: measured live at
  490 ms / 102,343 buffers, every 60 seconds. With the fence: 0.18 / 5.4 / 1.6 ms. Note the
  cost was *highest when the pipeline was caught up* and would have fallen as a backlog
  developed, so it would never have looked like a backlog problem.

The floor's distinction from the grammar cursor floor is deliberate and the two are not
interchangeable. A reduction cursor stops moving when its *source* goes quiet, which is
indistinguishable from a stall, so the grammar floor asks "are there unconsumed rows below
the cutoff" instead. A pending marker has much less ambiguity -- it is set at insert and
cleared in the same transaction as the downstream write. The same migration's header records
that a *time* bound was tried for this pipeline and reverted: the dispatch->feedback hop
legitimately ran p50 34.6 hours and max 11.3 days behind. Like `grammar_traces`, this table
is periodic-only and deliberately not in `main.py`'s startup pass, which already blocks the
event loop for ~260s.

**The window is 10 days, and the reason is consumer windows, not disk.** Two live readers
reach back further than a week: `orion/autonomy/evals/run_attention_bound_proposal_eval.py`
uses `WINDOW_DAYS = 7`, and `scripts/analysis/measure_proposal_feedback_correlation.py`
defaults to 200 hours (8.33 days). A 7-day retention window would tie exactly to the first
and sit *inside* the second, racing both at their oldest edge -- and both degrade quietly to
"insufficient data" rather than failing loudly. 10 days clears the longer of the two with a
~1.7-day margin. Live: 10 days keeps 202,276 of 474,861 rows; 7 days would have kept 113,060.

`scripts/analysis/measure_proposal_feedback_correlation.py`'s chain-completeness check was
bounded to its window in the same patch. It previously joined **every** feedback frame ever
written against the upstream tables with no `WHERE` clause at all, which was correct only
while those tables were unbounded; under retention it would have reported `INCOMPLETE`
forever for the entirely expected reason that pruned rows do not resolve. It now reports the
excluded count explicitly, so a shrinking denominator cannot be mistaken for a clean result.

**Two hazards worth knowing about before extending any of this:**

1. **`_verify_delete_safe` does not mean what it says here.** It checks `pg_constraint` for
   incoming foreign keys and reports
   `substrate_proposal_frames_delete_is_child_safe; no incoming FK constraints found`. That
   check was written for the grammar tables, which genuinely have no children. This table
   has three referencing tables -- they store `source_proposal_frame_id` as a plain `text`
   column with **no FK declared** -- so "no incoming FK" here means the database will not
   *tell* you when you orphan something, not that nothing can be orphaned. The chain floor
   is what provides the actual safety; the FK check is not a second opinion.
2. **`reconcile_policy_pending` is a third writer of the marker.**
   `services/orion-policy-runtime/app/store.py` re-sets `policy_pending = true`, on a 900s
   timer, for any proposal with no matching policy decision frame, with **no time bound**.
   It can only ever add work, so today it is harmless. It becomes a trap the moment
   `substrate_policy_decision_frames` gets retention of its own: pruning a decision frame
   makes its proposal look unprocessed, reconcile re-flags it, the chain floor pins at the
   oldest re-flagged row, and proposal retention stops permanently while policy-runtime
   reprocesses hundreds of thousands of ancient proposals. Bound reconcile's anti-join
   first.

**Retention does not reclaim disk.** `DELETE` returns space for reuse inside the relation,
not to the OS. `substrate_proposal_frames` stays ~1.76 GB on disk after pruning; 84% of that
is TOAST (1,471 MB of 1,760 MB), with only 145 MB of heap and 144 MB of indexes. Recovering
it would need `VACUUM FULL` (exclusive lock) or `pg_repack` (not in the image), and neither
is worth it -- the volume has 619 GB free. **The win here is bounding future growth, not
reclaiming what is already allocated.**

**Autovacuum:** these tables now carry per-table autovacuum settings
(`services/orion-sql-db/manual_migration_autovacuum_high_churn_tables.sql`). The cluster
default `autovacuum_vacuum_scale_factor = 0.2` is a proportional trigger, which on a large
churning table means an enormous absolute one -- `grammar_events` had to reach ~1.18M dead
tuples before autovacuum would look at it. Continuous 60-second retention DELETEs make that
worse, not better. The override is `scale_factor 0.05 + threshold 3000`, a **2.3x-4.0x**
increase in vacuum frequency measured against live row counts.

An earlier draft used `0.01 + 10000` (~17x on `grammar_events`) and justified it as "same
total work, spread out". That was wrong: heap work scales with dead tuples, but **index**
vacuum cost scales with index *size*, not dead-tuple count, and these tables carry
3442 MB / 2756 MB / 1184 MB of indexes. 17x passes means ~17x the index-scan I/O, on a host
already I/O-stalled ~22% of wall time. ~3.5x is a deliberate compromise, not a measured
optimum -- if dead tuples still climb, measure the I/O cost before lowering it further.
Vacuum cost limits are deliberately left alone for the same reason.

The four substrate tables additionally get `toast.autovacuum_*` settings. TOAST relations
have their own autovacuum parameters and inherit **nothing** from the main table, and all
four were on `reloptions = NULL`. Since 84% of `substrate_proposal_frames`' bytes are TOAST,
omitting these would have skipped the very relation the file exists to fix. (TOAST relations
are never ANALYZEd, so only the vacuum parameters apply. `toast.*` settings also do not show
up in the *main* table's `reloptions` -- verify them by joining to `reltoastrelid`.)

This depends entirely on each table having a `(created_at, <id>)` index --
`services/orion-sql-db/manual_migration_grammar_atlas.sql` declares
`idx_grammar_events_created_at` / `idx_grammar_edges_created_at` /
`idx_grammar_atoms_created_at` (`substrate_organ_emissions` already had
`idx_substrate_organ_emissions_created` from its original table definition). Without
these, the retention DELETE's `ORDER BY created_at ... LIMIT` has no efficient access
path and forces a full scan on every run -- confirmed live 2026-08-19:
`grammar_events` retention had been silently failing on every single startup with
`psycopg2.errors.QueryCanceled: canceling statement due to statement timeout` since
the table grew past whatever size made that scan exceed
`SQL_WRITER_GRAMMAR_STATEMENT_TIMEOUT_MS` (10s) -- `n_tup_del` on that table was 585
lifetime against 8.58M inserts, i.e. retention had essentially never actually pruned
anything despite being "enabled" the whole time.

Applying the new indexes against a small/fresh database, the plain
`create index if not exists` in the migration file (applied normally) is fine. Against
a large, already-live database, add `CONCURRENTLY` and apply each index as its own
statement (not inside the batch file, since `CONCURRENTLY` cannot run inside an
implicit/explicit multi-statement transaction):

```bash
docker exec -e PGPASSWORD=postgres <sql-db-container> psql -U postgres -d <db> -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_grammar_events_created_at ON grammar_events(created_at, event_id);"
docker exec -e PGPASSWORD=postgres <sql-db-container> psql -U postgres -d <db> -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_grammar_edges_created_at ON grammar_edges(created_at, edge_id);"
docker exec -e PGPASSWORD=postgres <sql-db-container> psql -U postgres -d <db> -c \
  "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_grammar_atoms_created_at ON grammar_atoms(created_at, atom_id);"
```

`build_grammar_truth_snapshot()` (`app/grammar_truth.py`) reports live status for all
six tables' retention under `grammar_retention`/`other_table_retention`, and flags
`degraded_reasons` if an index is missing or a retention run failed/never ran.

**Known gap (updated 2026-08-20):** `grammar_traces` is now covered -- see above. Still
unbounded: `grammar_temporal_hops`, `grammar_compactions`, `grammar_projections`, and the
rest of the substrate frame family. Deleting a `grammar_events`/`grammar_atoms`/
`grammar_edges` row doesn't touch these, so they keep accumulating.

Measured live 2026-08-20, the substrate frame family alone is ~15 GB and only two of its
tables have any bound at all:

```text
substrate_organ_emissions              5,248 MB   bounded
substrate_execution_dispatch_frames    2,126 MB   UNBOUNDED
substrate_proposal_frames              1,758 MB   bounded (this patch)
substrate_field_state                  1,745 MB   UNBOUNDED
substrate_feedback_frames              1,641 MB   UNBOUNDED
substrate_policy_decision_frames       1,485 MB   UNBOUNDED
substrate_attention_frames             1,055 MB   UNBOUNDED
substrate_perception_embedding_baseline  477 MB   UNBOUNDED
```

Extending retention to the rest is mostly mechanical now that `_substrate_chain_floor`
exists -- the same pending markers cover the policy and dispatch stages. It is deliberately
NOT done here: that is ~8.3 GB of substrate history, which is the cognition substrate's own
record of what it proposed, decided and did, and deleting it is Juniper's call to make
explicitly rather than a side effect of a retention patch.

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-sql-writer
```

### Smoke Test
Validate GPT turn ingest end-to-end (bus -> sql-writer -> Postgres):

```bash
python services/orion-sql-writer/scripts/smoke_chatgpt_turn_sql.py
```

Expected output includes `found_in_chat_gpt_log: True`; sql-writer logs should include:
`Written ChatGptLogTurnV1 -> chat_gpt_log`.

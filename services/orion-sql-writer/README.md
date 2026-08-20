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

`grammar_events`, `grammar_edges`, `grammar_atoms`, `substrate_organ_emissions` and
`grammar_traces` all get bounded retention (`*_RETENTION_DAYS` env keys, **default 3**).
Each is a batched `DELETE ... LIMIT batch_size` loop
(`GRAMMAR_EVENTS_RETENTION_BATCH_SIZE`/`_MAX_BATCHES_PER_STARTUP`/`_MAX_ELAPSED_SEC`,
shared across all five tables), capped so a huge backlog can't turn a pass into an
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
four tables' retention under `grammar_retention`/`other_table_retention`, and flags
`degraded_reasons` if an index is missing or a retention run failed/never ran.

**Known gap:** `grammar_traces` (and its other children -- `grammar_temporal_hops`,
`grammar_compactions`, `grammar_projections`) still have no retention. Deleting a
`grammar_events`/`grammar_atoms`/`grammar_edges` row doesn't touch its parent
`grammar_traces` row or these other children, so they'll keep accumulating
unbounded. Not addressed by this patch -- would need its own "safe to delete a
trace" definition (e.g. all children older than the cutoff too) before extending
retention there.

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

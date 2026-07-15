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
| `orion:memory:drives:audit` | `memory.drives.audit.v1` | `DriveAuditSQL` |

**Action outcome persistence:** `action.outcome.emit.v1` (produced by `orion-spark-concept-induction` after an autonomous readonly fetch) is projected into `action_outcomes` (PK `action_id`, idempotent upsert). `orion-cortex-exec` reads it back per-subject for chat-stance action feedback. DDL is applied on boot (`app/main.py` lifespan) and also lives in `services/orion-sql-db/manual_migration_action_outcomes_v1.sql`.

**Drive audit persistence:** `memory.drives.audit.v1` (produced by `orion-spark-concept-induction` on every DriveEngine tick) is projected into the slim measurement table `drive_audits` (PK `artifact_id`, idempotent upsert; `active_count` derived at write time as `len(active_drives)`). Read by `scripts/analysis/measure_autonomy_gate.py` for the drive co-activation verdict — the successor source to the Fuseki DriveAudit graph, frozen since 2026-06-19. DDL is applied on boot (`app/main.py` lifespan) and also lives in `services/orion-sql-db/manual_migration_drive_audits_v1.sql`.

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

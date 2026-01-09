# Orion SQL Writer

The **SQL Writer** service is a durable consumer that subscribes to various bus channels and persists structured payloads into a relational database (PostgreSQL). It uses a configurable routing map to determine which SQLAlchemy model to use for each message kind.

## Contracts

### Consumed Channels
Configured via `SQL_WRITER_SUBSCRIBE_CHANNELS` (JSON list).

| Default Channel | Kind(s) | Target Table |
| :--- | :--- | :--- |
| `orion:tags:enriched` | `tags.enriched`, `collapse.enrichment` | `CollapseEnrichment` |
| `orion:collapse:sql-write` | `collapse.mirror` | `CollapseMirror` |
| `orion:chat:history:log` | `chat.history`, `chat.log` | `ChatHistoryLogSQL` |
| `orion:dream:log` | `dream.log` | `Dream` |
| `orion:telemetry:biometrics` | `biometrics.telemetry` | `BiometricsTelemetry` |
| `orion:spark:introspection:log` | `spark.introspection` | `SparkIntrospectionLogSQL` |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `SQL_WRITER_SUBSCRIBE_CHANNELS` | (See above) | List of channels to subscribe to. |
| `SQL_WRITER_ROUTE_MAP_JSON` | (See above) | JSON mapping of `kind` → `ModelName`. |
| `POSTGRES_URI` | ... | Database connection string. |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health check channel. |

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-sql-writer
```

### Smoke Test
Publish a known kind to a subscribed channel.

```bash
# Using the bus harness to simulate a biometrics payload
python scripts/bus_harness.py tap &
# (Manually publish a message using a helper script or via another service)
```

# Orion Memory Consolidation

Subscribes to `orion:memory:turn:persisted` (sql-writer post-commit outbox), classifies each chat turn via LLM gateway quick lane logprobs, patches `chat_history_log.spark_meta`, tracks consolidation windows, and runs `memory_graph_suggest` on boundary closure.

## Channels

| Direction | Channel |
|-----------|---------|
| In | `orion:memory:turn:persisted` |
| Out | `orion:chat:history:spark_meta:patch` |

## Bring-up

```bash
docker compose --env-file ../../.env --env-file ../orion-bus/.env -f docker-compose.yml up -d --build
```

Apply Postgres migration: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`

## Smoke

```bash
PYTHONPATH=. python scripts/smoke_memory_consolidation_pipeline.py
```

Requires live stack (bus, sql-writer, postgres, llm-gateway, cortex).

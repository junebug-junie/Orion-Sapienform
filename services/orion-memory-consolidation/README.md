# Orion Memory Consolidation

Subscribes to `orion:memory:turn:persisted` (sql-writer post-commit outbox), classifies each chat turn via LLM gateway quick lane logprobs, patches `chat_history_log.spark_meta`, tracks consolidation windows, and runs `memory_graph_suggest` on boundary closure.

## Channels

| Direction | Channel |
|-----------|---------|
| In | `orion:memory:turn:persisted` |
| Out | `orion:chat:history:spark_meta:patch` |
| Out (threshold) | `orion:signals:memory_consolidation` (`signal.memory_consolidation.turn_change`) |

## Turn change appraisal

Each persisted turn (after the first in a window) gets a logprob-calibrated `turn_change_appraisal` patch on `spark_meta`: novelty score, shift kind, confidence, and baseline mode (`prior_turn` or `session_window` fallback). The first turn in a window uses `turn_change_status=skipped` (no baseline, no LLM call). High-confidence novel turns also emit `OrionSignalV1` on `orion:signals:memory_consolidation`.

| Env | Default | Purpose |
|-----|---------|---------|
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Re-appraise vs session window when novelty margin is below this |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Minimum novelty to emit substrate signal |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Prior turns in session-window baseline |
| `CHANNEL_SIGNALS_PREFIX` | `orion:signals` | Bus prefix for organ signal publish |

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

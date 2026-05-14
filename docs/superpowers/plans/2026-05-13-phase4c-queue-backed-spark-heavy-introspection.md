# 2026-05-13 — Phase 4C: Queue-backed Spark Heavy Introspection

See user-provided implementation spec in chat / PR description. Summary:

- PubSub `spark.candidate` unchanged; lightweight telemetry/tissue remain in `handle_candidate`.
- Heavy cortex/orch/LLM introspection moves to Redis Stream `orion:queue:spark:introspection` with consumer group `spark-introspector-workers` via `QueueRabbit`.
- Rollback: `SPARK_INTROSPECTION_QUEUE_ENABLED=false`, `SPARK_INTROSPECTION_INLINE_HEAVY_ENABLED=true`.

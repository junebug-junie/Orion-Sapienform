# PR: Fix spark_telemetry metadata upsert column name

**Branch:** `fix/sql-writer-spark-telemetry-metadata-upsert`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/fix/sql-writer-spark-telemetry-metadata-upsert`  
**Head:** `f9fdd20c` (1 commit)

## Summary

Fixes a one-line SQLAlchemy mapping bug that caused **every** `spark.telemetry` upsert to fail with:

```text
UndefinedColumn: column "metadata_" of relation "spark_telemetry" does not exist
```

The `SparkTelemetrySQL` model maps Python attribute `metadata_` → DB column `metadata`. The ON CONFLICT `set_` dict used the string key `"metadata_"`, so PostgreSQL received `SET metadata_ = ...` instead of `SET metadata = ...`.

Because upserts failed, sql-writer never persisted `spark_telemetry` rows and never ran the `_spark_meta_minimal` backfill into `chat_history_log.spark_meta`. Phi/novelty from spark-introspector was emitted on the bus but never landed in chat history.

```text
Hub chat.history write → chat_history_log (generic spark_meta: mode, trace_verb, turn_effect)
Hub spark.candidate → spark-introspector → spark.telemetry bus
spark.telemetry → sql-writer upsert_spark_telemetry  [BROKEN: metadata_ column]
  → spark_telemetry table (never written)
  → chat_history_log phi/novelty backfill (never ran)
```

**Fix:** Use `SparkTelemetrySQL.__table__.c.metadata` as the `set_` key so compiled SQL targets the real column.

**Non-goals respected:** No Hub changes, no bus/schema changes, no env/docker/settings changes.

## Root cause trace

| Stage | Status before fix |
|-------|-------------------|
| Hub initial `chat_history_log` write | Generic `spark_meta` (mode, trace_verb, turn_effect) — expected |
| spark-introspector emits `spark.telemetry` with phi | Working (logs confirm) |
| sql-writer `upsert_spark_telemetry` | **Failed** on `metadata_` column |
| `chat_history_log.spark_meta` phi backfill | **Never ran** |

## Files changed

| Path | Change |
|------|--------|
| `services/orion-sql-writer/app/spark_telemetry_persist.py` | Use `__table__.c.metadata` in ON CONFLICT `set_` |
| `services/orion-sql-writer/tests/test_spark_telemetry_idempotency.py` | Regression test: compiled SQL must use `metadata =`, not `metadata_ =` |

## Tests

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/fix/sql-writer-spark-telemetry-metadata-upsert
PYTHONPATH=. /mnt/scripts/Orion-Sapienform/orion_dev/bin/python -m pytest \
  services/orion-sql-writer/tests/test_spark_telemetry_idempotency.py -q --tb=short
# 4 passed in 0.62s
```

## Code review

Manual review (review subagents hit API limit):

- **Spec compliance:** ✅ Minimal fix, correct column reference pattern, regression test exercises real `upsert_spark_telemetry` path
- **Quality:** ✅ Approved — no scope creep, test would fail on pre-fix code (verified via SQL compile)

## Rollout

1. Merge PR
2. Rebuild/restart `orion-sql-writer` container
3. Verify on next chat turn:
   - `spark_telemetry` row exists for `correlation_id`
   - `chat_history_log.spark_meta` contains `phi` and/or `novelty`

No migrations. No `.env_example` changes.

## Test plan

- [x] `test_on_conflict_update_uses_metadata_column_not_python_attr` — compiled SQL uses `metadata =`
- [x] Existing idempotency tests still pass (4/4)
- [ ] Post-deploy: confirm `spark_telemetry` row + `chat_history_log.spark_meta.phi` for a live turn

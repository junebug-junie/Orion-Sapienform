-- bus_fallback_log typed timestamps + spark_telemetry correlation_id idempotency
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_bus_fallback_ts_spark_idempotency.sql

ALTER TABLE bus_fallback_log
  ADD COLUMN IF NOT EXISTS created_at_ts TIMESTAMPTZ;

UPDATE bus_fallback_log
SET created_at_ts = CASE
  WHEN created_at IS NULL OR btrim(created_at) = '' THEN NULL
  WHEN created_at ~ '^\d{4}-' THEN created_at::timestamptz
  WHEN created_at ~ '^\d+(\.\d+)?$' THEN to_timestamp(created_at::double precision)
  ELSE NULL
END
WHERE created_at_ts IS NULL;

CREATE INDEX IF NOT EXISTS idx_bus_fallback_log_created_at_ts
  ON bus_fallback_log (created_at_ts);

CREATE INDEX IF NOT EXISTS idx_bus_fallback_log_kind_created_at_ts
  ON bus_fallback_log (kind, created_at_ts);

-- Keep newest spark_telemetry row per correlation_id before unique index.
DELETE FROM spark_telemetry st
WHERE telemetry_id NOT IN (
  SELECT DISTINCT ON (correlation_id) telemetry_id
  FROM spark_telemetry
  WHERE correlation_id IS NOT NULL AND btrim(correlation_id) <> ''
  ORDER BY correlation_id, timestamp DESC NULLS LAST, telemetry_id ASC
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_spark_telemetry_correlation_id
  ON spark_telemetry (correlation_id);

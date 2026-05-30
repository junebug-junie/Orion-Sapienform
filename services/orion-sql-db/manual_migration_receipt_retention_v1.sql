-- Substrate receipt retention v1 (apply before restarting orion-substrate-runtime)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_receipt_retention_v1.sql

ALTER TABLE substrate_reduction_receipts
  ADD COLUMN IF NOT EXISTS receipt_kind text DEFAULT 'success',
  ADD COLUMN IF NOT EXISTS receipt_status text DEFAULT 'ok',
  ADD COLUMN IF NOT EXISTS event_id text,
  ADD COLUMN IF NOT EXISTS delta_id text,
  ADD COLUMN IF NOT EXISTS reducer_name text,
  ADD COLUMN IF NOT EXISTS stream_name text,
  ADD COLUMN IF NOT EXISTS payload_hash text,
  ADD COLUMN IF NOT EXISTS payload_bytes bigint,
  ADD COLUMN IF NOT EXISTS is_full_payload boolean DEFAULT false,
  ADD COLUMN IF NOT EXISTS applied_at timestamptz,
  ADD COLUMN IF NOT EXISTS expires_at timestamptz;

CREATE INDEX IF NOT EXISTS idx_substrate_receipts_expires_at
  ON substrate_reduction_receipts (expires_at);

CREATE INDEX IF NOT EXISTS idx_substrate_receipts_status_created
  ON substrate_reduction_receipts (receipt_status, created_at);

CREATE INDEX IF NOT EXISTS idx_substrate_receipts_delta_id
  ON substrate_reduction_receipts (delta_id);

-- Optional backfill: mark legacy rows for eventual prune (does not delete data)
UPDATE substrate_reduction_receipts
SET
  receipt_kind = 'success',
  receipt_status = 'ok',
  is_full_payload = true,
  expires_at = created_at + interval '48 hours'
WHERE expires_at IS NULL;

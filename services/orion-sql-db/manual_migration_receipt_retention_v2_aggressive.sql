-- Substrate receipt retention v2 — aggressive 30-minute TTL backfill
-- Apply after deploying substrate-runtime with ORION_RECEIPT_RETENTION_SUCCESS_MINUTES=30:
--   psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_receipt_retention_v2_aggressive.sql
--
-- Marks existing success/debug rows as immediately eligible for safe prune (once deltas applied).
-- Does not delete data; orion-substrate-runtime batched pruner removes rows.

UPDATE substrate_reduction_receipts
SET expires_at = created_at + interval '30 minutes'
WHERE receipt_kind IN ('success', 'debug_sample')
   OR (receipt_kind IS NULL AND receipt_status IS NOT DISTINCT FROM 'ok');

UPDATE substrate_reduction_receipts
SET expires_at = created_at + interval '6 hours'
WHERE receipt_kind = 'error';

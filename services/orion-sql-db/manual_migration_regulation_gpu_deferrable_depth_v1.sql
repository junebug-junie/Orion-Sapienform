-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-durable-runs
-- GPU pool state history: how much of each snapshot's queue is Orion's own deferrable self-work
-- (2026-10-23): queued background HOLDS with no parent hold, never a lease already counted in
-- shed_held_depth (orion/gpu_pool/shed.py deferrable_queued). Arousal's GPU-queue input reads it
-- as hunger, not load (orion/regulation/arousal.py; body-economy design, "S2 decision").
-- sql-writer writes the column; durable-runs' regulate node reads it. Additive and rerunnable:
-- older rows keep NULL (unknown, read as 0 deferrable).
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--   < services/orion-sql-db/manual_migration_regulation_gpu_deferrable_depth_v1.sql
ALTER TABLE gpu_pool_state_history ADD COLUMN IF NOT EXISTS deferrable_depth json;

-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-durable-runs
-- GPU pool state history: what an active shed (heat) is holding back in each snapshot's queue
-- (2026-10-20). Arousal's GPU-queue input reads a queue the heat shed is holding as no evidence,
-- not as load (orion/regulation/arousal.py). sql-writer writes the columns; durable-runs' regulate
-- node reads them. Additive and rerunnable: older rows keep NULL (unknown, not zero).
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--   < services/orion-sql-db/manual_migration_regulation_gpu_shed_held_v1.sql
ALTER TABLE gpu_pool_state_history ADD COLUMN IF NOT EXISTS shed_held_depth json;
ALTER TABLE gpu_pool_state_history ADD COLUMN IF NOT EXISTS shed_reason varchar;

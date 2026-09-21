-- World-pulse reading retry counters (Stage 1 + Stage 2). Additive, rerunnable.
-- Apply after the seed_queue, stage2 and general_reading migrations, before
-- deploying the Hub that ships HUB_WORLD_PULSE_READ_MAX_ATTEMPTS.
-- Apply: PGPASSWORD=postgres psql -h 127.0.0.1 -p 55432 -U postgres -d conjourney \
--   -f services/orion-sql-db/manual_migration_world_pulse_read_retry_v1.sql
ALTER TABLE world_pulse_read_seed
    ADD COLUMN IF NOT EXISTS attempts int NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS stage2_attempts int NOT NULL DEFAULT 0;

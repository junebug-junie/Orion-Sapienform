-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Walkway rollups by hour, day, week and month (2026-10-18).
-- Design: docs/superpowers/plans/2026-10-18-walkway-rollups-design.md
--
-- vision_rollup: one row per (camera stream, grain, period). Written by orion-sql-writer
-- (app/vision_rollup_loop.py), deterministic, no LLM. `rollup` holds the structured counts
-- (orion/vision/walkway_rollup.py, schema vision_rollup.v1); `line` is the one plain-English
-- template sentence. Hour rows are written as each hour closes; day rows are built from hour
-- rows, week and month rows from day rows. `complete` is false while the period is still
-- running (today, this week, this month) or a child period has no row; `covered_until` says
-- how far a partial row reaches.
--
-- Readers (all tolerate the table being absent): orion-actions walkway journal
-- (walkway_forecast.py), the day letter (orion/orion_day/gather.py), curiosity study material
-- (orion-hub), the Hub "Walkway history" panel, and orion-substrate-runtime's "seen on" links.
--
-- Safe for the post-pull auto-apply (orion/sql_migration_drift.py auto_apply_refusals): one new
-- table, no DO blocks, no functions other than now(), no index on an existing table.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_rollup_v1.sql
-- Rollback (additive): DROP TABLE vision_rollup;

CREATE TABLE IF NOT EXISTS vision_rollup (
    stream_id      TEXT NOT NULL,
    grain          TEXT NOT NULL CHECK (grain IN ('hour', 'day', 'week', 'month')),
    period_start   TIMESTAMPTZ NOT NULL,
    period_end     TIMESTAMPTZ NOT NULL,
    label          TEXT,
    complete       BOOLEAN NOT NULL DEFAULT false,
    covered_until  TIMESTAMPTZ,
    rollup         JSONB NOT NULL,
    line           TEXT NOT NULL,
    rules_version  INTEGER NOT NULL DEFAULT 1,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (stream_id, grain, period_start)
);

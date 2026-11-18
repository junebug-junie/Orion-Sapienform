-- ORION-MIGRATION-REQUIRED-BY: orion-recall
-- Lived-memory recall status + trace on the recall read log (#3628; spec
-- docs/superpowers/specs/2026-11-13-core-memory-pages-and-concept-growth-design.md, "Recall trace").
-- lived_status says, per recall call, whether lived-memory recall ran and why it injected what it
-- did: ok | below_bar | skipped_short | degraded | timeout | error | off. NULL = lived recall did
-- not run on that call (not a chat-turn continuity call), and NULL on rows older than this column.
-- lived_trace holds ids and Orion's own search text only (rewritten queries, candidate ids that
-- reached the rerank, injected ids, backend), never memory text.
-- Writer: orion-recall (services/orion-recall/app/worker.py, _persist_decision).
--
-- Apply BEFORE deploying orion-recall:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_recall_telemetry_lived_status_v1.sql
-- Additive, nullable, idempotent. Rollback: ALTER TABLE recall_telemetry DROP COLUMN lived_status, DROP COLUMN lived_trace;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS lived_status text;
ALTER TABLE recall_telemetry ADD COLUMN IF NOT EXISTS lived_trace jsonb;

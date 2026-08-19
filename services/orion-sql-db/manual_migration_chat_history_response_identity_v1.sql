-- chat_history_log / aitown_chat_history_log: response_identity column
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_chat_history_response_identity_v1.sql
--
-- Additive only. `user_id` keeps meaning exactly what it always has (the
-- prompt-side / human identity) -- every existing reader of that column is
-- unaffected. `response_identity` is new: the identity that produced
-- `response` on an assistant-authored row -- the served model-card name
-- (e.g. "qwen-36-instruct", "claude-sonnet-5") when one is known, else a
-- human-readable responder name (e.g. "Claude"). Never set on prompt-only
-- rows.
--
-- This file documents the change for manual/out-of-band application. The
-- authoritative, self-healing mechanism is the boot-time
-- `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in
-- services/orion-sql-writer/app/main.py (same convention every other
-- chat_history_log column addition has used), which reapplies on every
-- container start, so this migration does not need to be run by hand
-- against orion-sql-writer's own database. Kept for parity with
-- services/orion-sql-db/manual_migration_aitown_chat_history_log_v1.sql and
-- for any other environment that reads this table without running
-- orion-sql-writer's own boot path.

ALTER TABLE chat_history_log ADD COLUMN IF NOT EXISTS response_identity TEXT;
ALTER TABLE IF EXISTS aitown_chat_history_log ADD COLUMN IF NOT EXISTS response_identity TEXT;

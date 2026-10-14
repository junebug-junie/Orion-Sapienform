-- ORION-MIGRATION-REQUIRED-BY: orion-vision-window, orion-durable-runs
-- Temporal Self seam S1 (company arcs): an append-only history of the room camera's presence
-- state machine (services/orion-vision-window/app/presence.py). substrate_embodied_presence
-- keeps only the latest state; this keeps one row per change so a stretch of company
-- ("someone was here from 09:02 to 11:15") can be replayed.
--
-- Writer: orion-vision-window (VISION_PRESENCE_TRANSITIONS_ENABLED). Without this table each
-- write logs `presence_transition_write_failed` and the state machine carries on.
-- Reader: orion-durable-runs, the Temporal Self chronicle's `presence_transition` source.
-- Without this table every chronicle window read fails (UndefinedTable), so apply BEFORE
-- deploying either service:
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 \
--     < services/orion-sql-db/manual_migration_vision_presence_transition_v1.sql
-- Check: python3 scripts/check_sql_migrations_applied.py --file manual_migration_vision_presence_transition_v1.sql
--
-- Columns:
--   transition_id       sha256(presence_id|occurred_at|to_state)[:32]; retries are no-ops.
--   from_state          NULL on the first row after a producer restart (nothing known before).
--   to_state            present | recent | absent | unknown (no window for STALE_AFTER_MS).
--   occurred_at         the state machine's own state_since (for unknown: the last observation).
--   identity_confirmed  a fresh "probable" face match at that moment (a "possible" match is not enough).
--   label               the enrolled identity label, ONLY when identity_confirmed (never a guess).
--
-- Retention: none, like field_dominance_run. Volume is one row per presence change (tens a
-- day); the chronicle's own copy in temporal_self_event is pruned by orion-durable-runs.
-- Additive and idempotent; re-running is safe.
BEGIN;
CREATE TABLE IF NOT EXISTS vision_presence_transition (
    transition_id      text PRIMARY KEY,
    presence_id        text NOT NULL,
    from_state         text,
    to_state           text NOT NULL,
    occurred_at        timestamptz NOT NULL,
    identity_confirmed boolean NOT NULL DEFAULT false,
    label              text,
    created_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT vision_presence_transition_label_only_when_confirmed
        CHECK (label IS NULL OR identity_confirmed)
);
CREATE INDEX IF NOT EXISTS idx_vision_presence_transition_occurred
    ON vision_presence_transition (occurred_at);
CREATE INDEX IF NOT EXISTS idx_vision_presence_transition_presence
    ON vision_presence_transition (presence_id, occurred_at);
COMMIT;

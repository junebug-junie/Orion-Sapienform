-- Orion's Day: one letter per Denver calendar day about what Orion thought about.
-- Contract: orion/schemas/orion_day.py (OrionDayLetterV1 is one row of this table).
--
-- Single writer: services/orion-durable-runs/app/orion_day_store.py, from the
-- `orion_day.letter` durable run's `persist` node:
--   INSERT ... ON CONFLICT (letter_date) DO NOTHING  -- first run to finish wins.
-- Readers: Hub (orion/orion_day/store.py fetch_letter) sends the email and stamps
-- emailed_at / email_notification_id; curiosity carry-forward injection stamps
-- carry_forward_offered_at / carry_forward_offered_run_id (both in the Hub PR).
--
-- note_md and carry_forward_md are two separate model outputs and must never be
-- merged: the note's prompt carries no instruction about future curiosity.
-- material keeps full texts (the email shows them); sources holds per-source read
-- status plus what the model's budgeted view condensed.
--
-- Grants: orion-hub and orion-durable-runs both connect as `postgres` (their
-- POSTGRES_URI / DATABASE_URL, checked 2026-09-30), which owns this table, so no
-- GRANT is needed for them. Deliberately NO grant to orion_readonly (the FCC
-- sandbox): material carries the chat digest and self-sense answers in one place;
-- widen that on purpose, not by default.
--
-- Apply manually (idempotent):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_orion_day_letter_v1.sql

CREATE TABLE IF NOT EXISTS orion_day_letter (
    letter_date date PRIMARY KEY,
    run_id text NOT NULL,
    window_start timestamptz NOT NULL,
    window_end timestamptz NOT NULL,
    note_md text NOT NULL,
    carry_forward_md text NOT NULL,
    material jsonb NOT NULL,
    sources jsonb NOT NULL DEFAULT '{}'::jsonb,
    journal_entry_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    emailed_at timestamptz,
    email_notification_id text,
    carry_forward_expires_at timestamptz,
    carry_forward_offered_at timestamptz,
    carry_forward_offered_run_id text,
    CHECK (window_end > window_start),
    CHECK (length(btrim(note_md)) > 0),
    CHECK (length(btrim(carry_forward_md)) > 0),
    CHECK (note_md <> carry_forward_md)
);

-- Hub's sender picks up letters not yet emailed.
CREATE INDEX IF NOT EXISTS orion_day_letter_unemailed
    ON orion_day_letter (letter_date) WHERE emailed_at IS NULL;

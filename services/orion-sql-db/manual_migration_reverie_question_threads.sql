-- ORION-MIGRATION-REQUIRED-BY: orion-thought
-- Reveries that ask, Patch 3: threads and read-back
-- (docs/superpowers/specs/2026-11-18-reverie-questions-design.md, "Threads (idea 9) and read-back").
--
-- reverie_question gains three columns, all written by orion-thought only (app/store.py):
--   offer_status  waiting | sent | lapsed | dropped. NULL = the question was never sent and never waited
--                 (a none_reason, a refusal, the cap, routing off). `waiting` = Hub turned an accepted
--                 question away only because the one focus slot was full; the reverie worker offers it
--                 again when the slot is free. `lapsed` = it waited ORION_REVERIE_THREAD_LAPSE_DAYS.
--                 `dropped` = at re-offer time it was a near-duplicate of an open question.
--   routed_at     when the question reached the focus slot (a re-offer is later than created_at; the
--                 3-a-day cap counts this, not created_at).
--   outcome_at    when the reconcile pass filled `outcome` (the curiosity run that took the focus ended).
--
-- reverie_thread: one row per thread_id, its status (open | concluded | exhausted | lapsed), why, and
-- the last question on it. Written by orion-thought's reconcile pass; read by orion-thought (the seed
-- picker never joins a closed thread; a continuing thread is offered first).
--
-- Only orion-thought reads the new columns and the new table. orion-hub, orion-cortex-exec and
-- orion-durable-runs read reverie_question through orion/reverie/questions.py's column list, which this
-- patch does not change, so they do not depend on this file.
--
-- Additive and idempotent (post-pull auto-apply allowlist: add column / create table / create index).
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_reverie_question_threads.sql
-- Rollback: DROP TABLE reverie_thread; ALTER TABLE reverie_question DROP COLUMN offer_status,
--           DROP COLUMN routed_at, DROP COLUMN outcome_at;

ALTER TABLE reverie_question ADD COLUMN IF NOT EXISTS offer_status text;
ALTER TABLE reverie_question ADD COLUMN IF NOT EXISTS routed_at timestamptz;
ALTER TABLE reverie_question ADD COLUMN IF NOT EXISTS outcome_at timestamptz;

-- No index on offer_status: the table grows by at most ORION_REVERIE_MAX_PER_DAY rows a day, and an index on
-- an existing table would need CONCURRENTLY (outside the auto-apply allowlist's single transaction).

CREATE TABLE IF NOT EXISTS reverie_thread (
    thread_id text PRIMARY KEY,
    thread_key text NOT NULL,
    seed_kind text NOT NULL,
    status text NOT NULL DEFAULT 'open',
    status_reason text,
    last_question_id text,
    last_activity_at timestamptz NOT NULL,
    status_at timestamptz NOT NULL DEFAULT now(),
    opened_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reverie_thread_status_idx ON reverie_thread (status, status_at);

-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Curiosity focus: ONE question Orion chose in a conversation with Juniper for its next regular
-- investigate run to look at. Design: docs/superpowers/specs/2026-10-16-orion-queue-curiosity-focus-design.md.
--
-- Writers (all Hub, all in orion/curiosity/focus.py):
--   * the queue_curiosity_focus introspect responder (services/orion-hub/scripts/curiosity_focus_listener.py)
--     INSERTs one 'pending' row, marking the previous pending row 'replaced' in the same
--     transaction. A pending row with origin 'juniper' is never replaced by origin 'self'.
--     focus_id is a uuid5 of (question, why, origin, chat_correlation_id), so a retried tool
--     call returns the same row instead of a second one.
--   * the investigate kickoff TAKEs the pending row (status 'taken', offered_run_id) after every
--     gate passed, and gives it back ('pending') only when the run was cancelled before Orion saw
--     the prompt.
--   * the turn's end stamps 'ran' (turn produced text) or 'failed', turn_ok, and the disposition
--     the run wrote on its :TurnOutcome (focus_disposition). A failed run never re-queues it.
--   * Juniper's Clear button in Hub marks a pending row 'cleared'.
--   * any pending row past expires_at is marked 'expired' (readers treat it as expired before the
--     sweep reaches it).
-- question / why / origin / chat_correlation_id / created_at / expires_at are never updated.
--
-- disposition: 'explored' or 'set_aside:<reason>'. NULL = unknown (the run wrote none, or it
--   could not be read) -- never read as explored. disposition_raw keeps what the run wrote when
--   it was not one of those two shapes.
--
-- Grants: orion-hub connects as `postgres`, which owns this table, so no GRANT is needed.
-- No grant to orion_readonly (the FCC sandbox).
--
-- Apply manually (idempotent):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_curiosity_focus_request_v1.sql

CREATE TABLE IF NOT EXISTS curiosity_focus_request (
    focus_id uuid PRIMARY KEY,
    question text NOT NULL,
    why text NOT NULL,
    origin text NOT NULL,
    chat_correlation_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    status_at timestamptz NOT NULL DEFAULT now(),
    replaced_by uuid,
    offered_at timestamptz,
    offered_run_id text,
    outcome_stamped_at timestamptz,
    turn_ok boolean,
    disposition text,
    disposition_raw text,
    CHECK (length(btrim(question)) > 0),
    CHECK (length(btrim(why)) > 0),
    CHECK (origin IN ('self', 'juniper')),
    CHECK (status IN ('pending', 'replaced', 'cleared', 'expired', 'taken', 'ran', 'failed')),
    CHECK ((offered_at IS NULL) = (offered_run_id IS NULL))
);

-- The one slot: at most one pending focus.
CREATE UNIQUE INDEX IF NOT EXISTS curiosity_focus_request_one_pending
    ON curiosity_focus_request (status) WHERE status = 'pending';
-- The turn's end, the cancel path and the run list find a run's focus.
CREATE INDEX IF NOT EXISTS curiosity_focus_request_offered_run
    ON curiosity_focus_request (offered_run_id) WHERE offered_run_id IS NOT NULL;
-- The read tool lists recent focuses.
CREATE INDEX IF NOT EXISTS curiosity_focus_request_created_at
    ON curiosity_focus_request (created_at);

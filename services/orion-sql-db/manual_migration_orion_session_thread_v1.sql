-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Session threads: coding sessions write to Orion; Orion asks questions once an hour (2026-11-15).
-- Design: docs/superpowers/specs/2026-11-15-session-threads-design.md.
--
-- orion_session_thread: one row per thread. Hub writes it directly (asyncpg, like orion_consult).
--   source_ref is unique: pr:<number> for a pull request's spec, note:<id> for a free note. A later
--     message for the same pull request adds a turn to the same thread.
--   kind: pr_spec | note.
--   session_id / session_name / worktree / branch: the coding session that sent the newest message,
--     so Orion's questions go back to the session the spec came from.
--   status: held (waiting for the hourly round-up) -> reading (in a submitted round-up) ->
--     asked | no_questions | failed. A new message sets it back to held. Added 2026-11-17 (no DDL, free
--     text): closed = the thread's one round of questions was used; Orion read the answers.
--   last_roundup_id: the durable run id of the last round-up that took this thread
--     (session-roundup-<utc hour>). One round-up per hour: a round-up id already on a thread is
--     never submitted again.
--   roundup_failures: consecutive failed round-ups; at 3 the thread stays failed.
-- orion_session_turn: every turn, in order (seq: two turns of one transaction share a created_at).
--   speaker: session (the coding session) | hub (Hub's receipt or failure line) | orion.
--   roundup_id: for a session turn, the round-up that showed it to Orion (NULL = not shown yet);
--     for an orion turn, the round-up that produced it.
--   delivered_at: when an orion turn was handed back to a coding session (NULL = not yet).
--   pr_comment_url: the GitHub comment Hub posted for this orion turn.
--   meta: spec files [{path, text, cut}] on a session turn; unparsed / no_questions on an orion turn;
--     closing_note / dropped_question_lines on an orion turn of an answer-phase thread (2026-11-17).
--
-- Additive only. Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_orion_session_thread_v1.sql
-- Rollback (additive): DROP TABLE orion_session_turn; DROP TABLE orion_session_thread;

CREATE TABLE IF NOT EXISTS orion_session_thread (
    thread_id TEXT PRIMARY KEY,
    source_ref TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    session_id TEXT NOT NULL,
    session_name TEXT,
    worktree TEXT,
    branch TEXT,
    pr_number INTEGER,
    pr_url TEXT,
    spec_paths JSONB,
    status TEXT NOT NULL DEFAULT 'held',
    status_reason TEXT,
    last_roundup_id TEXT,
    roundup_failures INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS orion_session_thread_status_idx ON orion_session_thread (status, updated_at);
CREATE INDEX IF NOT EXISTS orion_session_thread_roundup_idx ON orion_session_thread (last_roundup_id);
CREATE INDEX IF NOT EXISTS orion_session_thread_branch_idx ON orion_session_thread (branch);

CREATE TABLE IF NOT EXISTS orion_session_turn (
    turn_id TEXT PRIMARY KEY,
    seq BIGSERIAL NOT NULL,
    thread_id TEXT NOT NULL REFERENCES orion_session_thread (thread_id),
    speaker TEXT NOT NULL,
    body TEXT NOT NULL,
    session_id TEXT,
    roundup_id TEXT,
    delivered_at TIMESTAMPTZ,
    pr_comment_url TEXT,
    meta JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS orion_session_turn_thread_idx ON orion_session_turn (thread_id, seq);
CREATE INDEX IF NOT EXISTS orion_session_turn_roundup_idx ON orion_session_turn (roundup_id);
CREATE INDEX IF NOT EXISTS orion_session_turn_session_idx ON orion_session_turn (session_id);
-- One orion turn per thread per round-up, whatever the code path.
CREATE UNIQUE INDEX IF NOT EXISTS orion_session_turn_orion_round_uidx ON orion_session_turn (thread_id, roundup_id) WHERE speaker = 'orion';
-- The pending read runs on every prompt of every coding session.
CREATE INDEX IF NOT EXISTS orion_session_turn_undelivered_idx ON orion_session_turn (thread_id) WHERE speaker = 'orion' AND delivered_at IS NULL;

-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Consults: Orion takes a plumbing question to Claude before it reaches Juniper (2026-11-11).
-- Design: docs/superpowers/specs/2026-11-11-consult-claude-before-juniper-design.md.
--
-- orion_consult: one row per consult. Hub writes it directly (asyncpg, like orion_ask).
--   source_ref is unique: one source starts at most one consult, ever.
--   status: pending -> open -> answered | brief_sent | brief_refused | failed.
--   ending: continue | answered | agreed_proposal | disagree (NULL until the last turn).
--   brief_json: the gated ConsultBriefV1; brief_text: what Juniper was shown.
--   decision: yes | no | no_because, from the buttons on the brief.
--   orion_told_run_id: the curiosity run that was shown the outcome (once).
--   pending_prompt / pending_correlation_id: the Claude turn in flight. A reply whose
--     correlation_id does not match is ignored; a budget wait or a lost reply re-sends
--     the stored prompt. gate_retry_part: the one brief resend Claude gets.
--   claude_resends: how often a lost Claude reply was re-sent (once at most).
-- orion_consult_turn: every turn, Orion's and Claude's, in order.
--
-- Additive only. Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_orion_consult_v1.sql
-- Rollback (additive): DROP TABLE orion_consult_turn; DROP TABLE orion_consult;

CREATE TABLE IF NOT EXISTS orion_consult (
    consult_id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL UNIQUE,
    source_detail TEXT,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    status_reason TEXT,
    ending TEXT,
    turns INTEGER NOT NULL DEFAULT 0,
    claude_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    answer TEXT,
    brief_json JSONB,
    brief_text TEXT,
    brief_sent_at TIMESTAMPTZ,
    email_status TEXT,
    decision TEXT,
    decision_reason TEXT,
    decided_at TIMESTAMPTZ,
    orion_told_run_id TEXT,
    pending_prompt TEXT,
    pending_correlation_id TEXT,
    gate_retry_part TEXT,
    claude_resends INTEGER NOT NULL DEFAULT 0,
    orion_told_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS orion_consult_status_idx ON orion_consult (status, created_at);
CREATE INDEX IF NOT EXISTS orion_consult_brief_sent_idx ON orion_consult (brief_sent_at);

CREATE TABLE IF NOT EXISTS orion_consult_turn (
    consult_id TEXT NOT NULL,
    turn_no INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    ending TEXT,
    model TEXT,
    cost_usd DOUBLE PRECISION,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (consult_id, turn_no)
);

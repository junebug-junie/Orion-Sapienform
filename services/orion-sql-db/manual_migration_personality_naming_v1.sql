-- ORION-MIGRATION-REQUIRED-BY: orion-durable-runs, orion-sql-writer
-- Self-naming: Orion names, rejects or merges their own personality dispositions.
-- Spec: docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md (PR #2742).
--
-- orion_disposition_naming: append-only log of every naming run's answer. Writer: orion-sql-writer
-- (bus orion:personality:naming, DispositionNamingV1). Reader: orion-durable-runs, which applies
-- it to personality_disposition (it stays the single writer of personality_*).
-- New personality_disposition columns are nullable / defaulted: additive only.

CREATE TABLE IF NOT EXISTS orion_disposition_naming (
    naming_id        text PRIMARY KEY,
    run_id           text NOT NULL,
    packet_id        text NOT NULL,
    disposition_id   text NOT NULL,
    verdict          text NOT NULL,
    name             text,
    sentence         text,
    merge_with       text,
    evidence_refs    jsonb NOT NULL DEFAULT '[]'::jsonb,
    shown            jsonb NOT NULL,
    hub_check        text NOT NULL,
    refused_reasons  jsonb NOT NULL DEFAULT '[]'::jsonb,
    written_at       timestamptz NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS orion_disposition_naming_disp_idx
    ON orion_disposition_naming (disposition_id, written_at);
CREATE INDEX IF NOT EXISTS orion_disposition_naming_created_idx
    ON orion_disposition_naming (created_at, naming_id);

ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS name text;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS name_sentence text;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS named_at timestamptz;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS naming_id text;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS orion_verdict text;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS merged_into text;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS name_grounding_rank integer;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS name_grounding_margin double precision;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS last_shown_at timestamptz;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS shown_count integer NOT NULL DEFAULT 0;
ALTER TABLE personality_disposition ADD COLUMN IF NOT EXISTS verdict_evidence_count integer;

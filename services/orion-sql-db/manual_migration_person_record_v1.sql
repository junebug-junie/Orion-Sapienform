-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- One record per person (2026-10-22). Design:
-- docs/superpowers/specs/2026-10-22-people-one-record-design.md
--
-- 1. person: one row per person Orion knows, from the camera or from
--    config/person_model/people.yaml. person_id reuses the referent node id
--    when people.yaml has one, else it is minted from the first name.
-- 2. person_link: what points at a person. A camera cluster Juniper named
--    (vision_individual), a named thing that belongs to them (vision_belonging,
--    e.g. a bicycle), a chat referent. One active link per source; a rename or
--    unlink sets revoked_at, nothing is deleted.
-- 3. person_fact: what Juniper told Orion about a person, in her words. Written
--    from her answer to a person question (voice ask_answer), her authored
--    profile (authored_profile) or a Hub edit (hub_edit). Never from a guess.
--
-- Rollback: manual_migration_person_record_v1_rollback.sql

CREATE TABLE IF NOT EXISTS person (
    person_id     TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'other',      -- interlocutor | self | other
    status        TEXT NOT NULL DEFAULT 'active',     -- active | not_a_person
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS person_link (
    link_id       TEXT PRIMARY KEY,                   -- md5(source_kind|source_ref|person_id|evidence_ref)
    person_id     TEXT NOT NULL REFERENCES person (person_id),
    source_kind   TEXT NOT NULL,                      -- vision_individual | vision_belonging | referent
    source_ref    TEXT NOT NULL,                      -- individual_id | referent node id
    decided_by    TEXT NOT NULL,                      -- camera_name | people_yaml | hub_edit
    evidence_ref  TEXT,                               -- the ask that named it
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at    TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS person_link_active_source_idx
    ON person_link (source_kind, source_ref) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS person_link_person_idx
    ON person_link (person_id) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS person_fact (
    fact_id       TEXT PRIMARY KEY,                   -- the ask_id, or md5 for other sources
    person_id     TEXT NOT NULL REFERENCES person (person_id),
    category      TEXT NOT NULL,                      -- orion/person_model/categories.yaml
    value_text    TEXT NOT NULL,                      -- her words, verbatim
    voice         TEXT NOT NULL,                      -- ask_answer | authored_profile | hub_edit
    source_ref    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    removed_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS person_fact_person_idx
    ON person_fact (person_id, category) WHERE removed_at IS NULL;

-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Act downstream use (body economy Organ 3b, weight source (a), 2026-10-23).
-- One row per grammar_acts act: was its output used by a later decision site within 72 h.
-- Single writer: services/orion-sql-writer/app/act_downstream_use.py (periodic rollup).
-- used is NULL when link_status = 'no_link_path' (cannot be measured), never false.
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--   < services/orion-sql-db/manual_migration_act_downstream_use_v1.sql

CREATE TABLE IF NOT EXISTS act_downstream_use (
    act_id        varchar PRIMARY KEY,
    actor         varchar NOT NULL,
    verb          varchar NOT NULL,
    started_at    timestamptz,
    link_status   varchar NOT NULL,
    link_path     varchar,
    output_ref    text,
    used          boolean,
    first_use_at  timestamptz,
    site          varchar,
    evidence_ref  text,
    arc_id        varchar,
    arc_evidence_n integer,
    decision_id   varchar,
    window_end    timestamptz,
    window_closed boolean NOT NULL,
    refreshed_at  timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_act_downstream_use_actor_started ON act_downstream_use (actor, started_at);

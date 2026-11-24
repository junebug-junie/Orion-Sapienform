-- ORION-MIGRATION-REQUIRED-BY: orion-thought
-- Reveries that ask, Patch 1 (docs/superpowers/specs/2026-11-18-reverie-questions-design.md).
-- Contract: orion/schemas/reverie.py (ReverieQuestionV1 is one row of this table).
--
-- One row per reverie: the seed it started from (kind, refs, material), the question it left (or why there
-- was none), why code refused it if it did, and where it was routed. Single writer:
-- services/orion-thought/app/store.py persist_reverie_question (INSERT ... ON CONFLICT (question_id) DO
-- NOTHING, then one UPDATE of routed_to/route_note after the route). `outcome` is filled by Patch 3 when
-- the dig ends. Readers: orion-thought (quiet gate: a row since the stretch began spends it; seed picker:
-- least recently used kind and spent refs; near-duplicate refusal), the chronicle (orion-durable-runs
-- temporal_self_sources.py, source kind reverie_question), Hub's reverie surfaces, the situation brief,
-- Prime's frame and the day letter.
--
-- Additive and idempotent (post-pull auto-apply allowlist: create table / create index only).
-- The orion_readonly grant is a separate file, applied by hand after this:
--   scripts/sql/2026-11-18_grant_orion_readonly_reverie_question_table.sql
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_reverie_question.sql
-- Rollback: DROP TABLE reverie_question;

CREATE TABLE IF NOT EXISTS reverie_question (
    question_id text PRIMARY KEY,
    thread_id text NOT NULL,
    thread_key text NOT NULL,
    seed_kind text NOT NULL,
    seed_refs text[] NOT NULL DEFAULT '{}',
    seed_material text NOT NULL DEFAULT '',
    question text,
    why text,
    refs text[] NOT NULL DEFAULT '{}',
    cites text[] NOT NULL DEFAULT '{}',
    none_reason text,
    refused_reason text,
    refused_detail text,
    routed_to text NOT NULL DEFAULT 'none',
    route_note text,
    outcome jsonb,
    correlation_id text,
    model_id text,
    question_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reverie_question_thread_created_idx ON reverie_question (thread_id, created_at);
CREATE INDEX IF NOT EXISTS reverie_question_seed_kind_created_idx ON reverie_question (seed_kind, created_at);
CREATE INDEX IF NOT EXISTS reverie_question_created_idx ON reverie_question (created_at);

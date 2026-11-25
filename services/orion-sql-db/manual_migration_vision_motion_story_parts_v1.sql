-- ORION-MIGRATION-REQUIRED-BY: orion-hub
-- Walkway story sentences graded in parts (who-card redesign Patch C, 2026-11-24).
-- Design: docs/superpowers/specs/2026-11-24-walkway-who-card-redesign.md, section 2, and
-- "Juniper's answers (2026-11-24)" (answer 3: people, direction, place; length left out).
--
-- vision_motion_story_parts: which parts a story's sentence states, recorded when the sentence
--   is composed (orion-sql-writer app/vision_motion.py sentence_parts, source 'composer'), or
--   derived ONCE from the story's stored facts for stories written before (source 'derived').
--   orion-sql-writer writes it fail-open (it checks the table exists), so only the Hub needs it.
-- vision_motion_story_part_verdict: Juniper's yes / no / can't tell per part (source 'juniper'),
--   or the "people = no" a name correction implies (source 'name_correction', with its ask).
-- vision_motion_story_verdict_source: who wrote each story verdict ('button', 'parts',
--   'name_correction'). A story verdict with no row here was written by the old buttons.
--
-- Safe for the post-pull auto-apply (orion/sql_migration_drift.py auto_apply_refusals): three new
-- tables only (their primary keys are the only indexes); no DO blocks, no functions other than
-- now(), nothing on an existing table.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_motion_story_parts_v1.sql
-- Rollback (additive): DROP TABLE vision_motion_story_verdict_source; DROP TABLE vision_motion_story_part_verdict; DROP TABLE vision_motion_story_parts;

CREATE TABLE IF NOT EXISTS vision_motion_story_parts (
    story_id       TEXT PRIMARY KEY,
    parts          TEXT[] NOT NULL,
    rules_version  INTEGER NOT NULL,
    source         TEXT NOT NULL CHECK (source IN ('composer', 'derived')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS vision_motion_story_part_verdict (
    story_id       TEXT NOT NULL,
    part           TEXT NOT NULL CHECK (part IN ('people', 'direction', 'place')),
    verdict        TEXT NOT NULL CHECK (verdict IN ('yes', 'no', 'cant_tell')),
    rules_version  INTEGER NOT NULL,
    source         TEXT NOT NULL DEFAULT 'juniper' CHECK (source IN ('juniper', 'name_correction')),
    ask_id         TEXT,
    answered_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (story_id, part)
);

CREATE TABLE IF NOT EXISTS vision_motion_story_verdict_source (
    story_id       TEXT PRIMARY KEY,
    source         TEXT NOT NULL CHECK (source IN ('button', 'parts', 'name_correction')),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

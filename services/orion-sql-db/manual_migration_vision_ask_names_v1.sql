-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- Names on visual confirmation cards (2026-10-15, Juniper): she can type who
-- someone is on the card she is already answering.
--
-- 1. orion_ask.answer_detail: what the answer carried besides the answer text.
--    A "same person?" card: {"names": {"<individual_id>": "<name>"}} (one name
--    for both on "Same person", one per face on "Different people").
--    A "who is this?" card with several pictures:
--    {"pictures": [{"image_ref", "verdict": same|different|unsure, "name"}]}.
--    A "who is this?" card closed by a named pair answer: {"resolved_by": "<pair ask_id>"}.
--    NULL for every older row and every other card.
-- 2. vision_individual.label_copied: true when the label was carried across a
--    "same" verdict from another individual, false when Juniper typed it for
--    this one. A typed name replaces a copied one; a copied one never replaces
--    a typed one. Backfilled once, when the column is created: a label is
--    copied when its label_ask_id is not this individual's own "who is this?" ask.
-- 3. vision_ask_picture_verdict: one row per picture on an answered multi-picture
--    "who is this?" card -- "this picture is the same person / someone else
--    (named or not) / not sure". Record only: nothing splits or rewrites an
--    individual from it yet. It is labelled data on how often one appearance
--    cluster holds more than one person. No foreign key: it outlives the prune.
--
-- Every step is idempotent. Requires manual_migration_walkway_camera_v1.sql,
-- manual_migration_walkway_privacy_v2.sql and manual_migration_vision_confirm_asks_v1.sql.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_ask_names_v1.sql
-- Rollback (additive): DROP TABLE vision_ask_picture_verdict; ALTER TABLE vision_individual DROP COLUMN label_copied; ALTER TABLE orion_ask DROP COLUMN answer_detail;

ALTER TABLE orion_ask ADD COLUMN IF NOT EXISTS answer_detail JSONB;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'vision_individual' AND column_name = 'label_copied'
    ) THEN
        ALTER TABLE vision_individual ADD COLUMN label_copied BOOLEAN NOT NULL DEFAULT false;
        UPDATE vision_individual v SET label_copied = true
        WHERE v.label IS NOT NULL AND v.label_ask_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM orion_ask a
              WHERE a.ask_id = v.label_ask_id AND a.source_kind = 'vision_individual'
                AND a.source_ref = v.individual_id);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS vision_ask_picture_verdict (
    ask_id         TEXT NOT NULL,
    image_ref      TEXT NOT NULL,               -- "thumb:<sha256>" as shown on the card
    picture_index  INTEGER NOT NULL,            -- 0 = the first picture (the face the question is about)
    individual_id  TEXT NOT NULL,               -- the individual the card asked about
    verdict        TEXT NOT NULL CHECK (verdict IN ('same', 'different', 'unsure')),
    name           TEXT,                        -- the name Juniper gave this picture, if any
    crop_id        TEXT,                        -- vision_crop_observation row behind the thumbnail, if still there
    answered_at    TIMESTAMPTZ NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ask_id, image_ref)
);
CREATE INDEX IF NOT EXISTS vision_ask_picture_verdict_individual_idx
    ON vision_ask_picture_verdict (individual_id);

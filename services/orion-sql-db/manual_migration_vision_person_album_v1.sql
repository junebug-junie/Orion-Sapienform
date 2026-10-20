-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer, orion-hub
-- Walkway person learning, Phase 1 (2026-10-20): Orion learns who people are
-- from Juniper's answers. Design: docs/superpowers/specs/2026-10-20-walkway-person-learning-design.md
--
-- 1. vision_person_exemplar: the album. One row per (person, crop, source):
--    +1 = this crop is that person, -1 = it is NOT. Built only from Juniper's
--    answers (a cluster she named, a "same person" verdict, a "different"
--    picture, later "Is this X?" yes/no). revoked_at set when the answer
--    behind it no longer holds (a rename, a split).
-- 2. vision_person_bar: per person and channel (face | body | osnet), the
--    learned accept / reject scores (conformal bounds; NULL = too little data,
--    ask only) and the album fingerprint they were fitted on.
-- 3. vision_person_attribution: Orion's guess for an unnamed cluster
--    ("probably X"), band accept | unsure. Never written to vision_individual.label.
-- 4. vision_person_model_eval: one row per held-out eval run.
--
-- Rollback (additive): DROP TABLE vision_person_model_eval, vision_person_attribution, vision_person_bar, vision_person_exemplar;

CREATE TABLE IF NOT EXISTS vision_person_exemplar (
    exemplar_id    TEXT PRIMARY KEY,               -- md5(person_label|crop_id|source)
    person_label   TEXT NOT NULL,
    crop_id        TEXT NOT NULL,                  -- vision_crop_observation.crop_id
    polarity       SMALLINT NOT NULL CHECK (polarity IN (-1, 1)),
    source         TEXT NOT NULL,                  -- cluster_label | pair_same | picture_different | confirm_yes | confirm_no
    ask_id         TEXT,                           -- the answer that put it here
    added_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS vision_person_exemplar_person_idx
    ON vision_person_exemplar (person_label) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS vision_person_bar (
    person_label      TEXT NOT NULL,
    channel           TEXT NOT NULL,               -- face | body | osnet
    accept_at         REAL,                        -- accept when a cluster's score is above this; NULL = no accept band
    reject_below      REAL,                        -- not this person when below; NULL = no reject band
    n_genuine         INTEGER NOT NULL,
    n_impostor        INTEGER NOT NULL,
    album_fingerprint TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    fitted_at         TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (person_label, channel)
);

CREATE TABLE IF NOT EXISTS vision_person_attribution (
    individual_id  TEXT NOT NULL,
    person_label   TEXT NOT NULL,
    channel        TEXT NOT NULL,
    score          REAL NOT NULL,
    band           TEXT NOT NULL CHECK (band IN ('accept', 'unsure')),
    model_version  TEXT NOT NULL,
    decided_at     TIMESTAMPTZ NOT NULL,
    revoked_at     TIMESTAMPTZ,
    PRIMARY KEY (individual_id, person_label, model_version)
);
CREATE INDEX IF NOT EXISTS vision_person_attribution_active_idx
    ON vision_person_attribution (individual_id) WHERE revoked_at IS NULL;

CREATE TABLE IF NOT EXISTS vision_person_model_eval (
    run_id         TEXT PRIMARY KEY,
    model_version  TEXT NOT NULL,
    ran_at         TIMESTAMPTZ NOT NULL,
    metrics        JSONB NOT NULL
);

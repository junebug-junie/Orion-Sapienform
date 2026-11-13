-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer
-- Street continuity: one passage across two cameras (2026-10-20).
-- Design: docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md, sections 3-4.
-- Writer: services/orion-sql-writer/app/vision_handoff.py (+ _loop.py). Read by the Hub
-- story card ("came from the door camera 38 s earlier") and the replay eval.
--
-- 1. vision_handoff: one row per expectation (gap / approach portal), overlap pair,
--    or doorbell ring nobody was seen approaching. status:
--      open | met | early | late | missed | ambiguous | unscorable  (expectations)
--      paired | ambiguous | unpaired                            (overlap pairs)
--      unannounced | announced | unscorable                 (doorbell rings)
--    g = transit_sec x exit_speed - exit_dist (frame widths) is the learned gap; met, early
--    and late rows train the next band.
-- 2. vision_handoff_hypothesis: up to five explanations of a missed handoff (and two of an
--    unannounced ring), each settled true/false from data by its deadline.
-- 3. vision_handoff_street_outbox: StreetEventV1 outbox for orion:vision:street:events.
-- 4. vision_handoff_cursor: landing-time cursors (motion stories).
--
-- Safe for the post-pull auto-apply: CREATE TABLE / CREATE INDEX IF NOT EXISTS only.
-- The writer backs off with a clear log line while this file is not applied.
--
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_vision_handoff_v1.sql
-- Rollback (additive): DROP TABLE vision_handoff_street_outbox, vision_handoff_hypothesis, vision_handoff_cursor, vision_handoff;

CREATE TABLE IF NOT EXISTS vision_handoff (
    handoff_id            TEXT PRIMARY KEY,           -- uuid5(portal | source story [| target story])
    portal_id             TEXT NOT NULL,
    portal_kind           TEXT NOT NULL,              -- gap | approach | overlap
    source_stream         TEXT,
    target_stream         TEXT,
    source_story_id       TEXT,
    target_story_id       TEXT,
    target_event_id       TEXT,                       -- vision_events.event_id (approach / ring)
    subject_kind          TEXT,                       -- motion story kind
    individual_id         TEXT,                       -- source side walker
    target_individual_id  TEXT,
    person_guess          TEXT,                       -- vision_person_attribution accept band only
    departed_at           TIMESTAMPTZ,                -- source story ended_at
    opened_at             TIMESTAMPTZ,
    window_start          TIMESTAMPTZ,
    window_end            TIMESTAMPTZ,
    late_until            TIMESTAMPTZ,
    exit_speed            REAL,                       -- frame widths per second
    exit_dist             REAL,                       -- projected frame widths to the exit edge
    band_source           TEXT,                       -- prior | learned | learned_person_widened
    band_lo               REAL,
    band_hi               REAL,
    model_version         TEXT,
    status                TEXT NOT NULL,
    appearance_score      REAL,
    appearance_channel    TEXT,                       -- face | body | osnet
    transit_sec           REAL,
    g                     REAL,
    p_value               REAL,
    coverage              REAL,
    attention_gate        BOOLEAN,
    salience              REAL,
    steered               TEXT[],                     -- redis keys set for this handoff
    ask_id                TEXT,
    arrival_log_score     REAL,                       -- nats/s of the arrival time under the predicted window
    scored_at             TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS vision_handoff_open_idx ON vision_handoff (status, late_until);
CREATE INDEX IF NOT EXISTS vision_handoff_source_idx ON vision_handoff (source_story_id);
CREATE INDEX IF NOT EXISTS vision_handoff_target_idx ON vision_handoff (target_story_id);
CREATE INDEX IF NOT EXISTS vision_handoff_portal_idx ON vision_handoff (portal_id, status, scored_at);

CREATE TABLE IF NOT EXISTS vision_handoff_hypothesis (
    handoff_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,                       -- detector_miss | late | turned_back | entered_place
                                                      -- | unexplained | out_of_view
    prior        REAL,                                -- counted odds at open (start at 1 each)
    status       TEXT NOT NULL DEFAULT 'open',        -- open | true | false
    settled_by   TEXT,                                -- data | juniper
    evidence     JSONB NOT NULL DEFAULT '{}'::jsonb,
    deadline     TIMESTAMPTZ NOT NULL,
    settled_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (handoff_id, kind)
);
CREATE INDEX IF NOT EXISTS vision_handoff_hypothesis_open_idx ON vision_handoff_hypothesis (status, deadline);

-- One row per StreetEventV1 (outbox): a later event on the same handoff never
-- overwrites an earlier one; published_at is set only after a successful publish.
CREATE TABLE IF NOT EXISTS vision_handoff_street_outbox (
    event_id      TEXT PRIMARY KEY,
    handoff_id    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    event         JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS vision_handoff_street_outbox_pending_idx
    ON vision_handoff_street_outbox (created_at) WHERE published_at IS NULL;

CREATE TABLE IF NOT EXISTS vision_handoff_cursor (
    name             TEXT PRIMARY KEY,                -- stories | ring_events
    last_created_at  TIMESTAMPTZ,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

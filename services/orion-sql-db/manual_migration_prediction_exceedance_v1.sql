-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: every issued threshold probability ("chance the cabinet reaches 34 C within 30
-- min"), and what actually happened in that window, so the warning can be graded (Brier score,
-- calibration) instead of only displayed.
-- Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md ("Grading the exceedance").
--
-- Additive only (post-pull auto-apply allowlist): one new table.

BEGIN;

CREATE TABLE IF NOT EXISTS prediction_exceedance (
    exceedance_id  bigserial PRIMARY KEY,
    series_id      varchar(120) NOT NULL,
    origin_ts      timestamptz  NOT NULL,
    issued_at      timestamptz  NOT NULL,
    name           varchar(60)  NOT NULL,
    threshold      double precision NOT NULL,
    window_start   timestamptz  NOT NULL,   -- first forecast bucket (origin + grain)
    window_end     timestamptz  NOT NULL,   -- exclusive
    probability    double precision,        -- NULL = too few residual paths (unknown, not 0)
    n_paths        integer      NOT NULL,
    served_baseline boolean     NOT NULL,
    -- outcome, written once the window has passed
    scored_at      timestamptz,
    hit_raw        boolean,                 -- any single reading >= threshold (what the 34 C reflex acts on)
    hit_mean       boolean,                 -- any grain-bucket mean >= threshold (what the forecast models)
    max_raw        double precision,
    n_readings     integer,
    UNIQUE (series_id, origin_ts, name)
);

COMMIT;

-- Undo: DROP TABLE prediction_exceedance;

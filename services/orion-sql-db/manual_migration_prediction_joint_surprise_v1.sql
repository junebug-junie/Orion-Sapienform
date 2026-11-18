-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: the joint surprise of a multivariate forecast family, one row per issued bucket and
-- horizon (app/family_a_live.py; spec docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md,
-- "A. Neural forecaster"). Family A = every raw body and room reading at once, 5 and 15 min ahead.
--
-- Each reading's own forecast lives in prediction_forecast (series_id 'family_a:<reading id>', role
-- champion) and the unserved rung in prediction_shadow_forecast, both under the same run_id. This table
-- holds what no single reading row can: the whole body's joint p for that bucket.
--
-- Lifecycle: status 'issued' at issue time (before the target bucket completes; difficulty = the
-- hour-before joint miss size, known then), then 'scored' (d2, both p's, excess) or 'missing' (too few
-- readings arrived) once the target bucket is complete. 'refused' = the hygiene gate refused this issue
-- (refusal holds the reasons); no reading rows exist for it.
--
-- Additive only (post-pull auto-apply allowlist): one new table and two indexes on it.

CREATE TABLE IF NOT EXISTS prediction_joint_surprise (
    joint_id      bigserial PRIMARY KEY,
    family        varchar(32)  NOT NULL,
    run_id        uuid REFERENCES prediction_model_run(run_id),
    horizon_min   integer      NOT NULL,
    issued_at     timestamptz  NOT NULL,
    origin_ts     timestamptz  NOT NULL,
    target_ts     timestamptz  NOT NULL,
    status        varchar(16)  NOT NULL CHECK (status IN ('issued', 'scored', 'missing', 'refused')),
    refusal       jsonb,
    n_forecast    integer      NOT NULL DEFAULT 0,
    difficulty    double precision,
    scored_at     timestamptz,
    n_observed    integer,
    d2            double precision,
    p_static      double precision,
    p_adaptive    double precision,
    p             double precision,
    calibration   varchar(32),
    calm_point    double precision,
    excess        double precision,
    top_shares    jsonb,
    UNIQUE (family, origin_ts, horizon_min)
);
CREATE INDEX IF NOT EXISTS prediction_joint_surprise_issued_idx
    ON prediction_joint_surprise (family, target_ts) WHERE status = 'issued';
CREATE INDEX IF NOT EXISTS prediction_joint_surprise_scored_idx
    ON prediction_joint_surprise (family, horizon_min, target_ts) WHERE status = 'scored';

-- Undo: DROP TABLE prediction_joint_surprise;

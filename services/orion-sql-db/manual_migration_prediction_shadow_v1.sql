-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: shadow forecasts -- every contender model's point forecast at each issue, served or
-- not, scored against the real reading like the served ones, so the gate can decide on live evidence
-- (a backtest win that does not hold live stops serving; a live win stops flickering at p ~ 0.05).
-- Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md ("Shadow scoring").
--
-- Additive only (post-pull auto-apply allowlist): one new table and two partial CONCURRENTLY indexes
-- (no BEGIN/COMMIT: CREATE INDEX CONCURRENTLY cannot run inside a transaction).

CREATE TABLE IF NOT EXISTS prediction_shadow_forecast (
    shadow_id     bigserial PRIMARY KEY,
    series_id     varchar(120) NOT NULL,
    model         varchar(60)  NOT NULL,
    issued_at     timestamptz  NOT NULL,
    origin_ts     timestamptz  NOT NULL,
    target_ts     timestamptz  NOT NULL,
    horizon_min   integer      NOT NULL,
    point         double precision NOT NULL,
    scored_at     timestamptz,
    actual        double precision,
    abs_error     double precision,
    UNIQUE (series_id, model, origin_ts, horizon_min)
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS prediction_shadow_unscored_idx
    ON prediction_shadow_forecast (series_id, target_ts) WHERE scored_at IS NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS prediction_shadow_scored_idx
    ON prediction_shadow_forecast (series_id, target_ts) WHERE scored_at IS NOT NULL;

-- Undo: DROP TABLE prediction_shadow_forecast;

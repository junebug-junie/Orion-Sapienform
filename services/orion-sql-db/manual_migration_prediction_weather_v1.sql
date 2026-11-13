-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: hourly outdoor weather at the situation layer's location (shared Open-Meteo client,
-- orion/weather/openmeteo.py), kept as history so it can be a forecast covariate. Observed hours come
-- from the forecast API's past days and its historical-forecast archive (same models, no level jump);
-- the next 48 h are forecast values, overwritten as they are observed and never read as history.
-- Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md ("Weather + fleet inputs").
--
-- Additive only (post-pull auto-apply allowlist): one new table.

BEGIN;

CREATE TABLE IF NOT EXISTS prediction_weather_hourly (
    hour_ts      timestamptz PRIMARY KEY,
    temp_f       double precision,
    is_forecast  boolean     NOT NULL,      -- true = a forecast value for an hour not yet observed
    source       varchar(20) NOT NULL,      -- forecast (future) | recent (past 7 d) | history (backfill)
    fetched_at   timestamptz NOT NULL
);

COMMIT;

-- Undo: DROP TABLE prediction_weather_hourly;

-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: Juniper reachable. One forecast per hour of "Juniper sends Orion a message in
-- [origin, origin + 1 h)", settled against chat_message (role = 'user', outside orion_* sessions).
-- p = served model (activity logistic: state, Denver clock rules, typing to Claude Code), p_state =
-- state-only, p_baseline = recency-weighted base rate. features = the served model's inputs at issue.
-- Spec: docs/superpowers/specs/2026-11-10-juniper-reachable-forecast-design.md.
--
-- Additive only (post-pull auto-apply allowlist): one new table and one index on it.

CREATE TABLE IF NOT EXISTS prediction_reachable_forecast (
  forecast_id   text PRIMARY KEY,
  origin        timestamptz NOT NULL UNIQUE,
  window_end    timestamptz NOT NULL,
  issued_at     timestamptz NOT NULL,
  model         text NOT NULL,
  p             real NOT NULL,
  p_state       real NOT NULL,
  p_baseline    real NOT NULL,
  served        text NOT NULL,
  hours_since   real,
  state_bucket  smallint NOT NULL,
  local_hour    smallint NOT NULL,
  n_messages    integer NOT NULL,
  features      jsonb NOT NULL,
  settled_at    timestamptz,
  happened      boolean,
  n_in_window   integer
);
CREATE INDEX IF NOT EXISTS prediction_reachable_forecast_pending ON prediction_reachable_forecast (window_end)
  WHERE settled_at IS NULL;

-- Undo: DROP TABLE prediction_reachable_forecast;

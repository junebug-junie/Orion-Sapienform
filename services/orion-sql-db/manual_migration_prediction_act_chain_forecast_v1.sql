-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: act chain (prediction-error suite, family B). One forecast per 15-min bucket of
-- "which act does Orion do next" (state = main verb of grammar_acts in the bucket, or idle), issued in
-- the first 3 min of the bucket from rows landed before it, settled 20 min after the bucket ends.
-- probs = every model's distribution {model: {state: p}}; k = backoff strengths chosen by LOO.
-- surprise / excess are the served model's (-log p(observed); minus the forecast's entropy).
-- Raw rows are kept (>= 90 days, spec retention rule): nothing deletes from this table.
-- Spec: docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md ("B. Markov chains").
--
-- Additive only (post-pull auto-apply allowlist): one new table and one index on it.

CREATE TABLE IF NOT EXISTS prediction_act_chain_forecast (
  forecast_id   text PRIMARY KEY,
  origin        timestamptz NOT NULL UNIQUE,
  window_end    timestamptz NOT NULL,
  issued_at     timestamptz NOT NULL,
  model         text NOT NULL,
  served        text NOT NULL,
  prev_state    text NOT NULL,
  block         text NOT NULL,
  n_train       integer NOT NULL,
  k             jsonb NOT NULL,
  probs         jsonb NOT NULL,
  settled_at    timestamptz,
  observed      text,
  n_acts        integer,
  surprise      real,
  excess        real
);
CREATE INDEX IF NOT EXISTS prediction_act_chain_forecast_pending ON prediction_act_chain_forecast (window_end)
  WHERE settled_at IS NULL;

-- Undo: DROP TABLE prediction_act_chain_forecast;

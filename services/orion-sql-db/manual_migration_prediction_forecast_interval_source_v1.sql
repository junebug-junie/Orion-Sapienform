-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: where each forecast's range came from (app/adaptive.py). "adaptive" = live-score
-- pool scaled by the last hour's miss size; "backtest:<reason>" = the backtest pool (horizon past
-- 15 min, thin pool, stale difficulty, a failed model handed to the baseline, adaptive disabled).
-- The surprise calm check filters on it; the adaptive pool excludes "backtest:model_fallback" rows.
--
-- Additive only (post-pull auto-apply allowlist): one nullable column on an existing table.

ALTER TABLE prediction_forecast ADD COLUMN IF NOT EXISTS interval_source varchar(64);

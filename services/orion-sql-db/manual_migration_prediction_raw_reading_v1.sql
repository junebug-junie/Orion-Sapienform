-- ORION-MIGRATION-REQUIRED-BY: orion-substrate-runtime, orion-hub
-- Rung 6 replay corpus (orion/substrate/prediction_expectation.py).
--
-- One row per raw per-tick reading of an EWMA-scored prediction-error domain
-- (execution, chat), recorded by orion-substrate-runtime BEFORE it becomes
-- surprise. The Hub's PredictionExpectationTuner replays these under candidate
-- smoothing-rate / variance-floor settings to let Orion revise how surprised
-- they expect to be. Raw readings do not depend on the setting, so a replay is
-- an exact counterfactual.
--
-- Written when SUBSTRATE_PREDICTION_RAW_READINGS_ENABLED=true; pruned by the
-- writer at SUBSTRATE_PREDICTION_RAW_READING_RETENTION_HOURS (default 192).
-- Expected volume: ~2k rows/day (execution ~70/h, chat per turn). Purely
-- additive; dropping it costs only the history.

create table if not exists substrate_prediction_raw_reading (
    domain text not null,
    observed_at timestamptz not null,
    raw_value double precision not null,
    recorded_at timestamptz not null default now(),
    primary key (domain, observed_at)
);
create index if not exists idx_prediction_raw_reading_observed
    on substrate_prediction_raw_reading(observed_at);

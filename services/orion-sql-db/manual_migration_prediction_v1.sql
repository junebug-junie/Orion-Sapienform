-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: model runs (backtest + champion), served forecasts with their live scores,
-- logged interventions, switchback experiments, and effect estimates.
-- Spec: docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md
--
-- Apply BEFORE starting orion-prediction (it refuses to start without these tables):
--   docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
--     < services/orion-sql-db/manual_migration_prediction_v1.sql
--
-- Additive only: new tables, indexes, and read-only grants for Orion's sandbox role.

BEGIN;

-- One selection writes one row per backtested model (kind=model) plus two stitched rows: the served
-- champion_blend and the baseline_blend it is compared against live. Forecast rows point at a blend.
CREATE TABLE IF NOT EXISTS prediction_model_run (
    run_id                 uuid PRIMARY KEY,
    selection_id           uuid         NOT NULL,
    series_id              varchar(120) NOT NULL,
    kind                   varchar(16)  NOT NULL CHECK (kind IN ('model', 'champion_blend', 'baseline_blend')),
    model                  varchar(60)  NOT NULL,
    fitted_at              timestamptz  NOT NULL,
    train_start            timestamptz,
    train_end              timestamptz,
    n_obs                  integer      NOT NULL,
    grain_min              integer      NOT NULL,
    usable_seasonalities   jsonb        NOT NULL DEFAULT '[]',
    refused_seasonalities  jsonb        NOT NULL DEFAULT '[]',
    n_windows              integer      NOT NULL,
    metrics                jsonb        NOT NULL,   -- horizon_min -> {n, mae, rmse, bias, rel_mae, passed, coverage80, coverage95, width80}
    assignment             jsonb        NOT NULL DEFAULT '{}',  -- blends: horizon_min -> model
    served_baseline        boolean      NOT NULL DEFAULT false,
    backtest_seconds       double precision NOT NULL,
    residual_paths         jsonb        NOT NULL,   -- (n_windows x h) list of lists, null = no actual
    errors                 jsonb        NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS prediction_model_run_series_idx ON prediction_model_run (series_id, fitted_at DESC);

CREATE TABLE IF NOT EXISTS prediction_forecast (
    forecast_id   bigserial PRIMARY KEY,
    run_id        uuid         NOT NULL REFERENCES prediction_model_run(run_id),
    series_id     varchar(120) NOT NULL,
    model         varchar(60)  NOT NULL,
    role          varchar(16)  NOT NULL CHECK (role IN ('champion', 'baseline')),
    issued_at     timestamptz  NOT NULL,
    origin_ts     timestamptz  NOT NULL,
    target_ts     timestamptz  NOT NULL,
    horizon_min   integer      NOT NULL,
    point         double precision NOT NULL,
    quantiles     jsonb        NOT NULL,
    -- live score, written by the scoring reducer once target_ts's bucket is complete
    scored_at     timestamptz,
    score_status  varchar(16)  CHECK (score_status IN ('scored', 'missing')),
    actual        double precision,
    abs_error     double precision,
    in_80         boolean,
    in_95         boolean,
    UNIQUE (series_id, role, origin_ts, horizon_min)
);
CREATE INDEX IF NOT EXISTS prediction_forecast_unscored_idx ON prediction_forecast (series_id, target_ts) WHERE scored_at IS NULL;
CREATE INDEX IF NOT EXISTS prediction_forecast_scored_idx ON prediction_forecast (series_id, role, target_ts DESC) WHERE scored_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS prediction_experiment (
    experiment_id  uuid PRIMARY KEY,
    action         varchar(120) NOT NULL,
    series_id      varchar(120) NOT NULL,
    created_at     timestamptz  NOT NULL,
    burn_in_min    integer      NOT NULL,
    design         jsonb        NOT NULL,
    guard          jsonb        NOT NULL,   -- experiment_guard probability at design time
    status         varchar(16)  NOT NULL CHECK (status IN ('designed', 'analyzed', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS prediction_intervention (
    intervention_id  uuid PRIMARY KEY,
    action           varchar(120) NOT NULL,
    started_at       timestamptz  NOT NULL,
    ended_at         timestamptz  NOT NULL,
    arm              varchar(8)   CHECK (arm IN ('on', 'off')),
    experiment_id    uuid REFERENCES prediction_experiment(experiment_id),
    source           varchar(80)  NOT NULL,
    notes            text         NOT NULL DEFAULT '',
    created_at       timestamptz  NOT NULL,
    CHECK (ended_at > started_at)
);
CREATE INDEX IF NOT EXISTS prediction_intervention_action_idx ON prediction_intervention (action, started_at);

CREATE TABLE IF NOT EXISTS prediction_effect_estimate (
    estimate_id      uuid PRIMARY KEY,
    action           varchar(120) NOT NULL,
    series_id        varchar(120) NOT NULL,
    claim_kind       varchar(32)  NOT NULL CHECK (claim_kind IN ('observational_effect', 'experimental_effect')),
    intervention_id  uuid REFERENCES prediction_intervention(intervention_id),
    experiment_id    uuid REFERENCES prediction_experiment(experiment_id),
    effect           double precision,
    ci_lo            double precision,
    ci_hi            double precision,
    p_value          double precision,
    n                integer      NOT NULL,
    details          jsonb        NOT NULL,
    created_at       timestamptz  NOT NULL
);
CREATE INDEX IF NOT EXISTS prediction_effect_action_idx ON prediction_effect_estimate (action, created_at DESC);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'orion_readonly') THEN
        GRANT SELECT ON prediction_model_run, prediction_forecast, prediction_experiment,
                        prediction_intervention, prediction_effect_estimate TO orion_readonly;
    END IF;
END $$;

COMMIT;

-- Undo (drops every forecast, score and effect estimate):
--   DROP TABLE prediction_effect_estimate, prediction_intervention, prediction_experiment,
--              prediction_forecast, prediction_model_run;

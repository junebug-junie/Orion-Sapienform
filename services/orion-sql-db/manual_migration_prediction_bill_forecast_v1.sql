-- ORION-MIGRATION-REQUIRED-BY: orion-prediction
-- orion-prediction: the power-bill bet (app/bill_forecast.py). Once a day, Orion's own projection of
-- the current Rocky Mountain Power bill: period-end kWh and pre-tax $ (orion.energy.reconcile
-- .project_period), each with an 80 % range (coverage; NULL with range_reason when no honest range).
-- origin = 'live' (made that day) or 'replay' (backfilled from stored meter history as of a past day;
-- never graded by the self-monitor). Settled when energy_bill_actual lands for the period:
-- kwh_in_80 / usd_in_80 (usd on the bill's pre-tax basis; NULL when the bill has no tax line).
--
-- Additive only (post-pull auto-apply allowlist): one new table and one index on it.

CREATE TABLE IF NOT EXISTS prediction_bill_forecast (
  bet_id            text PRIMARY KEY,
  usage_point_id    text NOT NULL,
  bet_date          date NOT NULL,
  origin            text NOT NULL,
  as_of             timestamptz NOT NULL,
  model             text NOT NULL,
  period_start      date NOT NULL,
  period_end        date NOT NULL,
  covered_through   timestamptz NOT NULL,
  days_into_cycle   double precision NOT NULL,
  remaining_days    double precision NOT NULL,
  observed_kwh      double precision NOT NULL,
  projected_kwh     double precision NOT NULL,
  projected_usd     double precision NOT NULL,
  coverage          double precision,
  kwh_lo80          double precision,
  kwh_hi80          double precision,
  usd_lo80          double precision,
  usd_hi80          double precision,
  range_reason      text,
  range_inputs      jsonb NOT NULL,
  tariff_version    text NOT NULL,
  created_at        timestamptz NOT NULL DEFAULT now(),
  settled_at        timestamptz,
  bill_period_start date,
  bill_period_end   date,
  billed_kwh        double precision,
  billed_usd        double precision,
  billed_basis      text,
  kwh_in_80         boolean,
  usd_in_80         boolean,
  UNIQUE (usage_point_id, bet_date)
);
CREATE INDEX IF NOT EXISTS prediction_bill_forecast_settled ON prediction_bill_forecast (settled_at);

-- Undo: DROP TABLE prediction_bill_forecast;

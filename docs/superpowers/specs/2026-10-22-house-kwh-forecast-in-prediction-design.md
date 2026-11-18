# House kWh forecast moves to orion-prediction; orion-energy keeps the price

Status: DESIGN (2026-10-22). No code yet. Follows PR #2843 (tariff r3, meter-read cycles).

## Arsonist summary

Move the *kWh forecast* to orion-prediction. Do not move the *pricing*. Only orion-energy knows
RMP's tariff rules (season split, proration, dated riders). orion-prediction knows how to
forecast, backtest, grade and give ranges.

Be honest about what the move fixes. Today's "$400 vs $427" miss came from the tariff, the
missing $18 tax, and the wrong cycle end, all fixed in #2843. The forecast itself was not the
problem: the run rate came within 2 % of the cycle's metered kWh from day 10 on. It missed by
5-14 % only in the first week. The move buys three things:

- a range instead of one number,
- grading against every bill,
- a forecast that splits out Orion's own servers, which use about a third of the house.

It does not buy weather skill yet. Daily kWh barely tracks temperature in this shoulder season.

## Evidence (live, 2026-10-22)

| fact | value | source |
|---|---|---|
| Unbroken house history | 2026-09-17 -> now (~35 days); plus Jul 29 - Aug 7 | `energy_usage_interval` |
| Daily kWh | mean 101, sd 15 (26 full days with weather) | same |
| Daily kWh vs mean temp | corr 0.16 | + `prediction_weather_hourly` (starts 2026-09-26) |
| Early Aug level | 41-59 kWh/day vs ~100 now | level shift (away? servers off?) |
| Servers, 14-day mean | athena 416 W + circe 948 W + hecate 112 W = 1.48 kW = 35 kWh/day | `orion_biometrics_summary`, `orion_biometrics_cluster` |
| Run-rate miss, Sep 18 cycle | day 3: -14 %, day 7: -6 %, day 10: +2 %, day 21: +0.6 % | replayed from `energy_usage_interval` |
| Data lag | ~35 h behind real time | `energy_importer_status.usage_lag_hours` |

## Current architecture

- `orion/energy/reconcile.py::project_period`: linear run rate (cycle kWh so far x period / elapsed),
  priced with `Tariff.itemize`. Feeds `energy.stakes.snapshot.v1.orion_projected_total_usd`
  (pre-tax) and forecast reconcile.
- orion-prediction: one target (`athena.cabinet_temp_c`), 5-min grain, horizons up to 24 h.
  Inputs count as stale after 3 grains + settle (`engine.py STALE_GRAINS`). It issues from the
  latest complete bucket before `now - settle`. It selects a champion model per horizon from 40
  backtest origins. Intervals are split-conformal per step. It publishes
  `prediction.forecast.v1` (per-step point + quantiles). Source kinds are biometrics, cooling,
  cluster, and weather.

## What orion-prediction cannot do today (the real work)

1. **No source for house usage.** It needs a source kind `energy_usage` that reads
   `energy_usage_interval`.
2. **Lagged input reads as stale.** Under `STALE_GRAINS` the 35 h lag is always stale. It needs a
   per-series `input_lag` (e.g. 72 h). The forecast origin must be the last *observed* bucket,
   not `now - settle`.
3. **Day buckets are UTC.** A local day ends at 18:00 UTC (MDT), and a bill counts local days.
   It needs a per-series `bucket_tz`.
4. **The horizon is too long for the history.** The cycle ends up to ~33 days after the origin
   (31-day cycle + lag). 40 backtest origins at a 33-day horizon need ~73 days of unbroken data,
   and we have ~35. It needs to declare long horizons while backtesting with whatever origins
   exist, and it must report `n_origins` and serve the baseline when there are too few
   (min 8).
5. **Per-day quantiles cannot be summed.** The bill needs the quantiles of the *remaining-cycle
   total*. Summing per-day p90s overstates the range. The service already simulates residual
   paths for exceedance, so it can give quantiles of a sum from the same paths.

## Proposed design

### orion-prediction

Registry entry (`config/prediction/series.v1.yaml`):

```yaml
- series_id: house.energy_kwh
  role: target
  source: {kind: energy_usage}          # sum of energy_usage_interval.energy_kwh
  unit: kWh
  grain: 1d
  bucket_tz: America/Denver            # NEW: a day is a local day, like the bill
  agg: sum
  max_fill: 0min                       # a missing hour makes the day missing, never guessed
  input_lag: 72h                       # NEW: RMP lags ~35 h; older than this is stale
  train_window: 60d
  seasonalities: [{name: weekly, period: 7d, fourier_order: 2}]
  horizons: [1d, 3d, 7d, 14d, 21d, 28d, 35d]
  min_backtest_origins: 8              # NEW: fewer -> serve baseline, say so
  covariates: []                       # phase 2 adds servers.kwh after its gate
  consumers: [orion-energy-stakes, prediction-scoreboard]
```

Baselines stay the engine's (naive, seasonal_naive at 7d). The incumbent run rate becomes an
explicit baseline, `cycle_mean`: the mean daily kWh so far this cycle. The bar a model must beat
is therefore at least what orion-energy does today.

New endpoint (HTTP, not bus: no schema change on a `forbid` model that hardware-watch consumes):

```text
GET /series/house.energy_kwh/sum?from=<local date>&until=<local date>
-> {issued_at, origin_ts, model, served_baseline, n_origins,
    point_kwh, quantiles: {"0.1": .., "0.5": .., "0.9": ..}, days_forecast}
```

It sums the stored residual paths over `[from, until)`, so the quantiles are of the total.

### orion-energy

- `project_period` takes observed kWh (unchanged) plus the remaining days from
  `/sum?from=<covered day>&until=<cycle end>`. It prices p10/p50/p90 through `Tariff.itemize`
  (pre-tax and with tax).
- If the endpoint is unreachable, stale, or `served_baseline` with too few origins, it uses the
  run rate and names it. The method is always published. This keeps the stakes honest; it is
  not a silent fallback.
- Additive fields on `EnergyStakesSnapshotV1`: `projection_method`
  (`prediction | run_rate`), `orion_projected_low_usd`, `orion_projected_high_usd`,
  `orion_projected_bill_usd` (p50 with tax: the number Juniper compares with the bill).

### Hub

The energy strip shows "Orion projects $402 - $418 - $436 (bill, with tax), method: prediction".

## Metric quality gate

**`house.energy_kwh` (target)**

1. Provenance: `energy_usage_interval`, written by sql-writer from `energy.usage.observed.v1`,
   which comes from the RMP Green Button XML (`orion/energy/espi.py`).
2. Independence: it is the target, not an input.
3. Theory: it is the billed quantity itself. The meter matched the bill within 0.9 %.
4. Live sanity: it varies (41-138 kWh/day) and can rest (the Aug level). It is not flat, null,
   or saturated.
5. Existing mechanism: the run rate in `project_period`. It stays as a named baseline and as the
   fallback.
6. Reversibility: one registry entry, an endpoint, and additive stakes fields. Cheap to remove.

**`servers.kwh` (phase-2 covariate): NOT wired yet**

- Provenance is known (iLO/PDU/BMC watts).
- It is not independent of the target: it is a *component* of the house total. So it enters as a
  decomposition (house = servers + rest; forecast rest; add planned server load), not as a
  regressor.
- Theory: energy conservation.
- It is blocked on one check. Over a full cycle, the hourly house kWh minus the server kWh must
  stay ≥ 0 and must not equal a constant. If the meter's hours and the iLO clock misalign, it
  goes negative.

**Weather: FAILS the live check today.** Corr 0.16 over 26 shoulder-season days. Not wired. Re-check
in January (heating) and July (AC: the hourly heat-balance eval found +0.33 kWh/h per °F above 75).

## Missing questions (for Juniper)

1. **Is the house heated electrically** (heat pump, baseboard, space heaters) or by gas? If it is
   electric, winter breaks the run rate and makes weather worth wiring in January. If it is gas,
   winter barely moves the bill.
2. What was early August (41-59 kWh/day)? Away, servers off, or both? Mark it as an `exclude`
   change event if it was unusual.
3. Do you want the Hub to lead with the with-tax bill number? This design assumes yes.

## Files likely to touch

- `services/orion-prediction/app/{series.py,store.py,frame.py,engine.py,main.py}`, `config/prediction/series.v1.yaml`, tests/evals
- `orion/energy/{reconcile.py,stakes.py}`, `orion/schemas/energy.py` (additive)
- `services/orion-sql-writer/app/models/energy.py` + a `manual_migration_energy_stakes_projection_v1.sql` (additive columns)
- `services/orion-hub/{scripts/energy_routes.py,static/js/energy-strip.js,templates/index.html}`
- `services/orion-energy/{.env_example,settings.py,README.md}` (`ENERGY_PREDICTION_URL`)

## Rollout order (consumer-first: the stakes model is `extra="forbid"`)

1. Schema fields + sql-writer columns + migration; deploy sql-writer and hub first.
2. orion-prediction: source kind, `bucket_tz`, `input_lag`, `min_backtest_origins`, `cycle_mean`
   baseline, `/sum`; deploy.
3. orion-energy reads `/sum`; deploy.

## Non-goals

- Moving the tariff or any pricing into orion-prediction.
- Weather or server covariates in phase 1.
- Using the range to *act* (stakes pressure still compares p50 with RMP's forecast). Spend gates
  are unchanged.
- An hourly house forecast.

## Acceptance checks

- `run_bill_projection_eval.py` (new, orion-energy evals): for each closed bill, replay the
  projection at day 3/7/14/21 of the cycle from stored usage, priced with the bill's tariff, for
  both methods. It reports absolute $ error. **The switch to `prediction` is accepted only if
  its median error beats the run rate at day 3 and day 7, with no worse at day 14/21, across
  ≥ 2 closed bills.** Until then `projection_method` stays `run_rate` and the prediction is shown
  beside it.
- Calibration: across graded days, the p10-p90 range of the remaining-cycle total contains the
  actual 70-90 % of the time (from `prediction_forecast` scoring).
- A stale or missing forecast publishes `projection_method=run_rate`, never a blank total.
- The day buckets sum to the bill's local days: a test with a DST-change cycle (Nov 1).
- Live: `/sum` returns `n_origins ≥ 8` before the method can switch. With the current history,
  expect that around early December.

## Recommended next patch

Ship `run_bill_projection_eval.py` alone, in orion-energy. It needs no move. It measures today's
run rate against both closed bills and gives every later phase its bar. The first bill cannot be
replayed (its usage is missing), so the second closed bill (read ~Nov 18) is when the gate can
first say anything.

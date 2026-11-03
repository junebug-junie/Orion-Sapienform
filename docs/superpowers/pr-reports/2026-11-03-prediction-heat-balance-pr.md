## Summary

- Fixed why the office room forecast stopped re-choosing its models every 6 hours. A model that only
  had test results on half the test windows was promoted by its live record, and the promoted model
  then left the served forecast without enough error samples. Live promotion now needs the same
  window coverage as backtest promotion (`app/backtest.py`, GATE_VERSION 4).
- The room temperature is now an input to the cabinet forecast. The existing covariate model is
  unchanged; a new model, `direct_arx_overlap`, uses the room too and trains only on the hours
  where the room sensor existed. The gate decides when it serves (earliest ~2026-11-07).
- New `GET|POST /effects/heat-per-watt`: how many degrees a held load of 100 W adds to the cabinet,
  per machine, in two parts -- "local" (cabinet over its intake air) and "total" (the room warms too).
  It converts an experiment's measured watts into a predicted cabinet change, and the
  experiment answer at `GET /effects/{action}` now carries that prediction beside it.
- The real cabinet fixture is re-exported to 2026-11-03 with the room column; two new evals/tests.

## Outcome moved

- `office.temp_c` reselection: refused every 6 h since 2026-11-02 ("too few blend residual paths at
  [5] min ... champion [8, 16, 16, 16, 16]") -> passes on the same real data (regression test fails on
  the old gate).
- Orion can now ask "how much does this action heat the cabinet?" and get a per-machine answer with an
  interval, not one confounded correlation. Live: the walkway focus experiment measured +0.21 C
  (-0.16..+0.58); the heat-per-watt prediction from the watts it moved is +0.18 C (-0.05..+0.41).

## Current architecture

orion-prediction backtests a model zoo per target, gates each horizon against the better baseline
(paired test + window share, or a live shadow verdict), and serves stitched blends with conformal
intervals. `office.temp_c` was a young second target, not a cabinet input. Effects had three levels
(scenario, intervention, switchback experiment); nothing related watts to cabinet heat inside the
service. An offline fit (`scripts/analysis/fit_heat_sensitivity.py` -> `config/physiology/heat_sensitivity.yaml`)
estimated whole-cabinet heat per compute watt for the body-economy budget, before the room sensor existed.

## Architecture touched

- orion-prediction only. HTTP API: two new routes. No bus channel, no schema registry entry, no DB
  migration (rows go into the existing `prediction_effect_estimate`).
- Registry contract: new optional series field `overlap_covariates` (validated subset of `covariates`).

## Files changed

- `services/orion-prediction/app/backtest.py`: live promotion needs window coverage; GATE_VERSION 4.
- `services/orion-prediction/app/series.py`: `overlap_covariates`; a series cannot be its own covariate.
- `services/orion-prediction/app/direct.py`, `app/models.py`: `direct_arx` column subset; `direct_arx_overlap`.
- `services/orion-prediction/app/frame.py`: covariate's own exclusions masked; straddling buckets masked.
- `services/orion-prediction/app/heat_balance.py` (new): local + total fits, refusals, conversion, storage.
- `services/orion-prediction/app/effects.py`, `app/main.py`: heat-per-watt routes; `best_effect` citation.
- `services/orion-prediction/app/experiment_runner.py`: `manipulation_watts` extracted for reuse.
- `config/prediction/series.v1.yaml`: room joins cabinet covariates as an overlap covariate.
- `services/orion-prediction/evals/export_cabinet_fixture.py`, `evals/fixtures/cabinet_5min.csv.gz`: room column, to 11-03.
- `services/orion-prediction/evals/run_cabinet_replay_eval.py`: `--train-from`, `--without-overlap`; spec/column alignment.
- `services/orion-prediction/evals/run_heat_per_watt_eval.py` (new), CI step in `.github/workflows/orion-prediction-tests.yml`.
- Tests: `tests/test_young_series.py`, `tests/test_heat_balance.py` (new); `test_direct.py`, `test_frame.py`, `test_series.py`.
- Docs: service README, spec `docs/superpowers/specs/2026-10-15-orion-prediction-service-design.md`.

## Root cause of the office selection failure

`direct_ar` needs about 3 days of training (288 complete rows at its 24 h lead, after its one-day-lag
feature drops the first day). The backtest gives its oldest windows 2 days. On a 4-5 day old series,
it therefore fails the oldest half of the windows ("only 287 complete training rows at lead 1"). That
alone was handled: the backtest route refuses a model with residuals on under 90 % of the baseline's
windows. But the live route (`verdict == "better"` from shadow forecasts) skipped that check, so
`direct_ar` was promoted at 5 min with 8 of 16 windows; the stitched champion column then had 8 < 10
paths, `check_paths` refused the whole reselection, and the old all-naive selection stayed served.
The raw data's single gap was not involved. Reproduced on the exported real data at the 11-02 21:10 cutoff.

The 0.68 live coverage is a different effect of youth, not this bug: the interval pool is 10-18
overlapping windows over 30-54 h; below 39 paths both the 80 % and 95 % levels clip to the pool's
min/max (live c80 = c95 at every horizon); and the room moves up to 2.5 C/h (11-03 00-03 UTC: -6 C in
3 h, outside anything in the pool). Live 60-min coverage was 0.54, 6 h was 0.79. It closes as the pool
reaches 40 windows. Explained, not fixed, and the alert is left on (it is correct).

## Room as a cabinet covariate: how the history mismatch is handled

The cabinet's training window starts 2026-10-23 (adopted break); the room starts 2026-10-29 00:10
(after its settling exclusion). Adding the room to `direct_arx` would fail every window on the
10 % covariate-gap check, or, interpolated, invent days of room air. So the room is an
`overlap_covariate`: `direct_arx` keeps exactly its old columns (replay results identical with and
without the room), and `direct_arx_overlap` trims its training to the rows where every covariate
reads, refusing windows with too little overlap. The gate compares it with the baseline only on
windows both forecast, and refuses it below 90 % window share on both routes. Nothing is forced.

## Metric quality gate: which watts

1. **Provenance.** `athena.chassis_watts` = athena iLO `ilo_power_watts` (`orion/telemetry/biometrics_pipeline.py`,
   BMC at the PSU). `circe.chassis_watts` = Panduit PDU outlets polled by athena for circe
   (`services/orion-biometrics/app/main.py:_proxy_measurements`; same value as `pdu_watts`).
   `hecate.chassis_watts` = hecate BMC read remotely. `gpu_watts_total` (nvidia-smi sum) and
   `cpu_watts_total` (RAPL packages) are contained in chassis. Hub `_WATTS_SQL` and `orion_energy_hourly`
   sum the same per-node chassis watts (+ cooling).
2. **Independence.** GPU and CPU watts are parts of chassis (not independent of it). circe `pdu_watts`
   is circe chassis. The fleet sum forces one slope on three meters. athena vs circe: different
   meters and machines, hourly corr -0.09.
3. **Theory.** Air heat balance: outlet minus inlet = P / (m_dot c_p), all electrical power ends as
   heat, so whole-node (chassis) watts, per machine. Check: cabinet on room + watts gives a room
   coefficient 0.97 (0.94..0.99), so "cabinet minus room" is the right local quantity.
4. **Live sanity** (hourly 10-29..11-03). athena 396-784 W (sd 132); circe 582-1353 W (sd 203);
   hecate 0 W except ~2 h (auto-refused: flat); AC 117-120 W in fan mode (auto-refused); athena GPU
   sd 6 W (flat). Why the sum dropped R^2 to 0.18: circe's watts vary more than athena's and the local
   gap does not see them. Why the level R^2 0.89 is not the answer: it is one step. At 2026-10-30
   02:20 UTC the gap fell 1.1 C in 10 minutes with athena flat at ~740 W (cabinet humidity rose 22.8 ->
   24.6 %: cooler air came in; no change event logged); athena then dropped 300 W at 04:00 and the gap
   moved ~0.1 C. The level fit reads that as +0.43 C/100 W; the change-based fit says +0.07.
5. **Existing mechanism.** `fit_heat_sensitivity.py` / `heat_sensitivity.yaml` (whole cabinet, compute
   watts, 09-26..10-27, athena 0.64, circe 0.31 C/100 W) predates the room sensor and is not reachable
   from the service or an experiment. The new total fit agrees with it; the new local fit is the
   split it could not make. It stays as the budget organ's input; not replaced.
6. **Reversibility.** One module and two routes; rows under new action names in an existing table; no
   schema, no training default.

**Winner:** per-machine `chassis_watts` (athena, circe), each with its own slope; hecate and the AC are
candidates that enter automatically once they vary. Rejected: fleet sum, GPU-only, CPU-only,
`orion_energy_hourly`.

## Heat per watt, live (2026-11-03, 7-day window)

| C per 100 W held ~2 h | local (cabinet - room) | total (room free, outdoor controlled) |
|---|---|---|
| athena | +0.07 (95 % -0.00..+0.14), 123 h | +0.54 (-0.02..+1.09), 166 h |
| circe | -0.01 (-0.04..+0.02) | +0.28 (+0.16..+0.40) |

Most of a machine's effect on the cabinet comes back through the room. Walkway focus experiment
9e6f9ab7 (+7.8 W athena, +50 W circe measured): predicted +0.18 C (-0.05..+0.41) total, -0.00 C local;
measured +0.21 C (-0.16..+0.58). Method: hourly means, lags 0-2 h summed, AR(1) feasible GLS +
Newey-West. The "95 %" interval covers ~89 % in simulation (plain OLS + HAC: ~82 %).

## Schema / bus / API changes

- Added: `GET /effects/heat-per-watt` (read-only), `POST /effects/heat-per-watt` (stores rows);
  `heat_balance_prediction` field on experimental answers from `GET /effects/{action}`; registry field
  `overlap_covariates`; model `direct_arx_overlap`.
- Removed: none. Renamed: none.
- Behavior changed: live promotion needs window coverage (GATE_VERSION 4 -> one reselect per target on
  deploy); covariate series mask their own exclusions; straddling buckets are masked.
- Compatibility notes: no DB migration; `prediction_effect_estimate` rows use new action prefixes
  `heat_per_watt:` / `heat_per_watt_total:`.

## Env/config changes

- Added keys: none. Removed keys: none. Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env change).
- skipped keys requiring operator action: none.

## Tests run

```text
docker (orion-prediction image + pytest), services/orion-prediction:
python -m pytest tests -q -p no:cacheprovider                 -> 244 passed
tests/test_young_series.py on the OLD backtest.py             -> FAILS (assert 'direct_ar' != 'direct_ar')  [regression proof]
host: tests/test_no_private_eval_artifacts.py, tests/test_no_personal_terms.py -> pass
scripts/check_metric_lineage.py --gate, check_definition_drift.py --gate, check_env_key_single_source.py -> PASS
```

## Evals run

```text
evals/run_causal_calibration_eval.py                           -> PASS
evals/run_heat_per_watt_eval.py (200 draws)                    -> PASS: planted coverage 0.895, mean 0.427 vs 0.43,
   circe covers 0 0.94, null claim rate 0.065 (refused 0.81), hecate never fitted, total coverage 0.88 / 0.905
evals/run_heat_per_watt_eval.py --live --days 7 (read-only)    -> numbers in the table above
evals/run_cabinet_replay_eval.py (registry train_from, CI mode) -> PASS
evals/run_cabinet_replay_eval.py --train-from 2026-10-23T19:25:21Z vs --without-overlap -> PASS both;
   every existing model's row identical; direct_arx_overlap: residuals on 8/40 windows (room history to
   11-03 00:00), rel MAE 1.09 / 1.01 / 1.21 / 0.65 (p 0.01) / 1.68 at 5m/15m/1h/6h/24h on those windows,
   not served (window share); assignment unchanged {5: naive, 15: direct_arx, 60: naive, 360: mstl_ets, 1440: naive}
```

## Docker/build/smoke checks

```text
Live read-only smoke from the worktree against the real DB (store writes blocked):
- office.temp_c backtest + live verdicts (repro): direct_ar 10/18 windows, errors "only 287 complete training rows"
- Effects.heat_per_watt(athena.cabinet_temp_c, 7 d, experiment 9e6f9ab7) -> fit + prediction above
No image build: requirements unchanged (statsmodels already pinned). Throwaway test image removed.
```

## Review findings fixed

(filled in after review)

## Restart required

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build   # from clean primary main after merge
curl -fsS -X POST localhost:8132/effects/heat-per-watt -H 'content-type: application/json' -d '{"days": 7}'
```

## Risks / concerns

- Severity: medium. Concern: the "95 %" heat-per-watt interval covers ~89 % in simulation on 5-day
  windows. Mitigation: stated in the response summary, README, and eval; widens toward nominal with
  more hours.
- Severity: medium. Concern: observational. The total fit has no control for people in the office,
  doors, or sun; the local fit is robust to steps but is not the whole effect. Mitigation:
  `claim_kind` is always `observational_effect`; the experiment stays the causal answer and is
  shown first.
- Severity: low. Concern: an unlogged physical change at 2026-10-30 02:20 UTC (gap -1.1 C) is not a
  change event. Mitigation: reported; Juniper may want to log it as a `break`.
- Severity: low. Concern: `office.temp_c` live coverage stays out of band until ~40 windows exist.
  Mitigation: explained above; the alert is correct and left on.

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

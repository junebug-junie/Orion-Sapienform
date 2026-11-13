## Summary

- **Shared weather seam.** `orion/weather/openmeteo.py` is now the one Open-Meteo client. The situation layer's `_fetch_weather` uses it with the same request as before, and orion-prediction uses it too.
- **Stored weather history.** orion-prediction keeps hourly outdoor temperature in `prediction_weather_hourly`:
  - Refreshed every hour: the past 7 days plus the next 48 h.
  - Gaps are backfilled from the historical-forecast API, which uses the same models as the live feed.
  - Only observed hours are ever read as covariate values. Stored forecasts are never read as history.
- **New covariates for the cabinet target:**
  - `circe.chassis_watts`, from the fleet table.
  - `hecate.chassis_watts`, from the fleet table; counted as 0 before its first reading, since hecate wasn't running yet.
  - `athena.central_ac_proxy`: outdoor °F above 75, 3 h earlier. It stands in for the central AC switching on.
- **Derived-input rules** live in one place, `app.frame.covariate_series`: `lag`, `above`, `zero_before_first`, plus holding each hourly weather value across its hour.
- **New eval:** `evals/run_house_heat_balance_eval.py`, read-only against the live DB. It re-checks the central-AC proxy against the house meter.

## Outcome moved

**Before:** the cabinet forecast saw only athena's power and the portable AC. Weather was fetched for chat turns but never kept.

**After:** every heat source in the cabinet is an input, and so is the central AC's on/off driver.

**Measured:**

| measurement | result |
|---|---|
| live smoke, 14 windows (10-19 to 10-21) | `direct_arx` at 0.73–0.89x persistence, not yet significant |
| 40-window replay ending 10-20 (inside the post-T10 regime) | about 1.0x, no significant gain; the gate serves the baseline |
| central-AC proxy, this fall | 0 throughout (outside below 75 °F since about 10-11) |

The change-event breaks adopt on 10-25 and 10-27. After that, training uses only post-change data, which is the cleaner test.

## Current architecture

- The situation layer had its own `urlopen` call to Open-Meteo and stored no history.
- The forecaster's covariates were `athena.chassis_watts` and `athena.cabinet_cooling_watts`, read raw.

## Architecture touched

- `orion/weather/` (new)
- `orion/situational/context.py`
- `services/orion-prediction`:
  - new source kinds `cluster` and `weather`
  - new series options: `lag`, `above`, `zero_before_first`
  - weather refresh in the engine
  - `covariate_series`, used by the engine and the scenario path
  - settings
- New migration.
- CI path triggers.

## Files changed

- `orion/weather/openmeteo.py` (new), `orion/situational/context.py`: the shared client.
- `orion/situational/tests/test_weather_shared_client.py` (new), `services/orion-cortex-exec/tests/test_situation_provider.py`: patch the shared client's `urlopen`.
- `services/orion-sql-db/manual_migration_prediction_weather_v1.sql`: the weather table. Auto-apply clean.
- `services/orion-prediction/app/weather.py` (new), plus `app/{frame,series,store,engine,effects,settings,main}.py`.
- `config/prediction/series.v1.yaml`: three covariate series and the target's covariate list.
- `services/orion-prediction/{.env_example,docker-compose.yml,README.md}`.
- `services/orion-prediction/tests/test_inputs.py` (new), `tests/test_series.py`.
- `services/orion-prediction/evals/{export_cabinet_fixture.py,run_cabinet_replay_eval.py,run_house_heat_balance_eval.py}`, `evals/fixtures/cabinet_5min.csv.gz` (re-exported with circe and hecate power; deliberately no weather).
- `.github/workflows/{orion-prediction-tests,orion-gpu-pool-tests,session-scope-tests}.yml`: trigger on `orion/weather/**`.
- The spec.

## Schema / bus / API changes

- Added:
  - table `prediction_weather_hourly`
  - `/health` → `weather` (refresh state, backfill, errors)
  - series registry source kinds `cluster` and `weather`, and options `lag`, `above`, `zero_before_first`
- Removed: none.
- Renamed: none.
- Behavior changed: `Effects.scenario` builds its covariate from the same view training uses (real buckets only, derived rules applied, exclusions masked).
- Compatibility notes: none on the bus.

## Env/config changes

- Added keys: `PREDICTION_WEATHER_ENABLED=true`, `PREDICTION_WEATHER_REFRESH_SEC=3600`, `ORION_SITUATION_WEATHER_LAT`, `ORION_SITUATION_WEATHER_LON`.
  - The location keys use the same names as the situation layer's.
  - They are empty in the template, and empty means weather is off.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes. The location values were copied from orion-hub's local `.env` (gitignored; never committed).
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-prediction: pytest tests -q -> 121 passed
orion/situational/tests -> 237 passed
services/orion-cortex-exec/tests/test_situation_provider.py -> 22 passed
migration: auto_apply_refusals -> [], REQUIRED-BY OK; tests/test_no_personal_terms.py -> passed
```

## Evals run

```text
run_cabinet_replay_eval.py (40 windows, real history incl. circe/hecate) -> PASS; direct_arx ~1.0x, baseline served
live smoke (branch image; real telemetry read-only, throwaway DB for writes): weather backfill 792 h in 1.4 s, all five
  covariates complete over 2 days, selection ran with no direct_arx errors
run_house_heat_balance_eval.py: run after deploy (needs the live weather table)
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-prediction build -> built; smoke above
```

## Review findings fixed

- Finding: the service crashed at boot when LAT/LON were empty (the template default).
  - Fix: `env_ignore_empty=True`, so empty means unset.
  - Evidence: `test_empty_weather_location_means_off_not_a_boot_crash`.
- Finding: cortex-exec CI broke, because 3 tests patched `situation.urlopen`.
  - Fix: they patch the shared client's `urlopen`; their URL and parameter assertions are kept.
  - Evidence: 22 passed.
- Finding: during a weather outage, stored forecasts were read as observed history.
  - Fix: covariate reads use observed hours only.
- Finding: a failed backfill also lost the forecast refresh.
  - Fix: the forecast is stored first, and the backfill sits in its own try.
  - Evidence: `test_a_failed_backfill_still_stores_the_forecast`.
- Finding: the archive (ERA5) and live data come from different models, a level jump.
  - Fix: backfill from the historical-forecast API. It matched the live feed exactly for the same hours; ERA5 read 2–4 °F lower.
- Finding: interior gaps were never backfilled.
  - Fix: `missing_weather_hours` covers holes, not just the head of history.
  - Evidence: the backfill test punches a hole.
- Finding: the hecate first-reading scan cost about 0.7 s per covariate load.
  - Fix: cached once found; a None is re-checked hourly.
- Finding: the weather refresh blocked the tick.
  - Fix: it runs after the series loop.
- Finding: the eval crashed if the hecate column was empty.
  - Fix: guarded.
- Finding: the test coordinates were near the real location.
  - Fix: changed to 0.0/0.0.
- Finding: the scenario path's gap handling changed without being documented.
  - Fix: documented.

## Restart required

From the clean primary `main` checkout after merge. The post-pull hook applies the migration first, then:

```bash
scripts/safe_docker_build.sh orion-prediction up -d --build
```

Then:

```bash
python3 services/orion-prediction/evals/run_house_heat_balance_eval.py
```

## Risks / concerns

- Severity: low.
  - Concern: the central-AC proxy carries no signal until it is warm outside again.
  - Mitigation: the heat-balance eval decides whether to keep it.
- Severity: low.
  - Concern: the +0.33 kWh/h per °F fit was computed with ERA5 temperatures.
  - Mitigation: re-run the eval after deploy, on the live-model history.
- Severity: info.
  - Concern: no significant forecast gain yet.
  - Mitigation: the gate keeps serving the baseline; re-check after the 10-25 and 10-27 break adoptions.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2829

🤖 Generated with [Claude Code](https://claude.com/claude-code)

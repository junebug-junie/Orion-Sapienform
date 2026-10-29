## Summary

- New **Climate and power — overlay** strip in Hub → Biometrics → Cabinet / Office. It shows office, cabinet and outdoor temperature on one °C plot. Orion's total wall watts sit in a panel underneath on the same time axis.
- Filter chips above the chart turn each series on or off. The chips are also the legend, and each series keeps its color when others are hidden.
- Axis ticks on both panels. One crosshair spans both panels, with a dot on each line and a tooltip listing every value, or "no reading" in a gap.
- New `GET /api/cabinet/climate/overlay?window=24h|3d|7d` returns all four series on a shared bucket grid: 5, 15 or 30 min. Outdoor is hourly.
- Bug fix: the **Office — pan-1 SHT40** charts had no hover. The hover code moved from `cabinet-sensors.js` into a shared `chart-hover.js`, so the office charts now use the same crosshair and tooltip.

## Outcome moved

Juniper can now see how cabinet heat follows outdoor temperature, office temperature and Orion's power draw on one time axis. Before, these lived in 3 separate places, and the weather was not in the Hub at all.

## Current architecture

The office strip (`office-env.js`) drew plain SVG lines with no hover. The cabinet charts (`cabinet-sensors.js`) had a private hover helper. Outdoor weather existed only in `prediction_weather_hourly`, written by orion-prediction. Orion power existed only as hourly kWh on the Orion energy tab.

## Architecture touched

orion-hub only: one new read-only route, one new static module, one shared hover module, and a template block. No bus, schema or env changes.

## Files changed

- `services/orion-hub/scripts/cabinet_climate_routes.py`: new overlay API. Each series is queried separately, so one failed table empties only that series.
- `services/orion-hub/scripts/api_routes.py`: mounts the router.
- `services/orion-hub/static/js/cabinet-climate-overlay.js` (+ `.test.js`): chart, filters, ticks, hover.
- `services/orion-hub/static/js/chart-hover.js` (+ `.test.js`, renamed from `cabinet-sensors.test.js`): shared hover helpers.
- `services/orion-hub/static/js/cabinet-sensors.js`: uses the shared hover.
- `services/orion-hub/static/js/office-env.js`: adds hover to the temp/humidity charts.
- `services/orion-hub/templates/index.html`: strip markup and script tags (`chart-hover.js` loads first).
- `services/orion-hub/tests/test_cabinet_climate_routes.py`, `test_cabinet_climate_overlay_panel.py`: route and wiring tests.

## Schema / bus / API changes

- Added: `GET /api/cabinet/climate/overlay`. Read-only Hub HTTP.
- Removed / Renamed: none.
- Behavior changed: none for existing routes.
- Compatibility notes: none.

## Env/config changes

- None. Reuses `OFFICE_ENV_NODE` and `CABINET_AMBIENT_HISTORY_NODE`. No `.env_example` change, so no sync was needed.

## Metric gate (Orion watts series)

This is a display aggregate, not a new cognition signal.
- **Provenance:** per-node `chassis_watts` in `orion_biometrics_cluster`, plus `home_cooling_sample.cooling_watts`. These are the same rows and filters as `orion-sql-writer/app/orion_energy_rollup.py`.
- **Independence:** it is a sum of existing signals, and nothing consumes it.
- **Live check:** 24h, about 1,430-2,300 W. It drops when GPU load drops, so it is not flat.
- **Correction:** Hecate reports exactly 0.0 W in every row while powered off. A node at 0 W is therefore not counted as a source.

## Tests run

```text
pytest test_cabinet_climate_routes.py test_cabinet_climate_overlay_panel.py test_office_env_panel.py test_office_env_routes.py test_cabinet_sensors_panel.py -> all passed
node --test (all 35 hub static/js test files) -> 368 pass, 0 fail
```

## Evals run

```text
None. This is a read-only display route with no quality dimension to score. The live query check below stands in.
```

## Docker/build/smoke checks

```text
Route run against live Postgres (24h): office 14-16, cabinet 289, outdoor 23-24, watts 289 buckets, errors {}.
Latest: office 26.1 °C, cabinet 28.2 °C, outdoor 14.3 °C, Orion 1,430 W from 3 sources.
7d query is about 1.5 s.
Headless Chromium render with live data: overlay hover tooltip lists all 4 series. Office chart hover shows crosshair + tooltip. Turning chips off collapses the watts panel and keeps colors fixed.
No container was rebuilt from this branch. A deploy from main is needed.
```

## Review findings fixed

- Finding: Hecate's flat 0 W counted as a live source ("4 sources" when 3 report).
  - Fix: `chassis_watts > 0` in the per-node branch.
  - Evidence: live latest bucket now reads `sources: 3`. A regression test was added.
- Finding: a failed window switch drew old data on the new window's axis.
  - Fix: render the domain from `payload.window`.
- Finding: `date_bin` floors the first bucket before the window start, so it drew into the label gutter.
  - Fix: widen `tMin` to the first point.
- Finding: `cabinet-sensors.js` threw if `chart-hover.js` was missing.
  - Fix: it now degrades to no hover.
- Not fixed (cosmetic): `preserveAspectRatio="none"` makes the office chart's dots slightly elliptical.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run it from the clean primary main checkout after merge.

## Risks / concerns

- Severity: low. Concern: the office sensor has only about 1.5 h of history (installed 2026-10-28 22:16 UTC), so its line is short for now. Mitigation: none needed. It fills in over time.
- Severity: low. Concern: outdoor is hourly model data (Open-Meteo), not a sensor. The strip text says so.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)

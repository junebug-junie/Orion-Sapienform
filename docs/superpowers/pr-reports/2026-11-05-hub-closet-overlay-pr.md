## Summary

- The server-closet temperature (the SCD40 above the server cabinet) now draws on the Hub's **Climate and power — overlay** chart, next to the office, cabinet and outdoor lines. It has its own on/off chip.
- Temperature only. The closet CO2 stays on its own strip.
- If the `closet_env_sample` table does not exist yet, the closet line is just missing. It is not shown as an error.
- The Athena Nano tile grid (environment, UV, magnetic, particulate, lidar, IMU) now has a **Cabinet sensors — Athena Nanos** heading. The particulate tile still reads `absent`, because the PMSA003I now lives on the office pan-1 array.

## Outcome moved

You can now see heat building up above the server cabinet on the same chart as the cabinet temperature, the office temperature and Orion's wall watts.

## Current architecture

`/api/cabinet/climate/overlay` (`scripts/cabinet_climate_routes.py`) served four series: office, cabinet, outdoor and watts. The closet SCD40 had its own strip (`closet_env_routes.py`), but no line on the overlay. `#cabinetSensorGrid` was an unlabeled box.

## Architecture touched

Hub only. No bus, schema or env changes.

## Files changed

- `services/orion-hub/scripts/cabinet_climate_routes.py`: adds the `closet_temp_c` series (`_CLOSET_SQL`, same node setting as the closet strip). A missing closet table reads as empty rows.
- `services/orion-hub/static/js/cabinet-climate-overlay.js`: adds the closet series, label "closet", colour `#e09a00`.
- `services/orion-hub/templates/index.html`: adds the closet chip, updates the overlay text, and puts a heading on the cabinet tile block.
- `services/orion-hub/README.md`: documents the closet line and the heading.
- `services/orion-hub/tests/test_cabinet_climate_routes.py`, `tests/test_cabinet_climate_overlay_panel.py`: cover the closet series, the missing-table case, chip/JS colour parity and the heading.

## Schema / bus / API changes

- Added: `closet_temp_c` key in `/api/cabinet/climate/overlay` `series`, `errors` and `series_grain_sec`.
- Removed / Renamed: none.
- Compatibility notes: the change only adds a key, and the only consumer is this chart.

## Env/config changes

None.

## Metric gate (closet_temp_c on the overlay)

1. Provenance: the pan-1 closet Nano SCD40 publishes `closet.env.sample.v1`, sql-writer stores it in `closet_env_sample.closet_temp_c`, and the overlay reads it with `_CLOSET_SQL`.
2. Independence: this is a separate physical sensor from the cabinet BME680 and the office SHT40. It is related to them only through real room heat.
3. Anchor: the sensor measures air temperature directly at the cabinet exhaust area.
4. Live data: 128 rows from 00:44 to 01:52Z on 11-05, range 26.2–28.7 °C. The real 5-minute overlay query returned buckets moving between 26.8 and 27.3 °C, so the data is not flat.
5. Existing mechanism: the overlay already did this job for the other temperatures. This patch reuses it.
6. Reversibility: it is one series key and one chip, so it is easy to remove.

## Colour choice

The palette validator (dataviz skill) checked the four temperature lines that share one plot. The yellow in the palette's band (`#c98500`) is too close to the cabinet orange: normal-vision ΔE is 10.6, below the minimum of 15. Magenta, red and green also fail against orange or aqua. `#e09a00` passes the colour-blind and normal-vision checks. It fails only the lightness band (0.736 against a maximum of 0.67), so it looks a little brighter than the other lines. The chips and the tooltip also name each line, so you do not have to tell the lines apart by colour alone.

## Tests run

```text
pytest tests/test_cabinet_climate_routes.py tests/test_cabinet_climate_overlay_panel.py tests/test_closet_env_routes.py tests/test_office_env_panel.py -q  -> 37 passed
node --test static/js/cabinet-climate-overlay.test.js static/js/chart-hover.test.js -> 14 pass, 0 fail
```

## Evals run

This change is a UI read path, and the Hub has no eval for it. The real overlay SQL was run against live Postgres (see the metric gate).

## Docker/build/smoke checks

UNVERIFIED in the browser until Hub is redeployed from main.

## Review findings fixed

The code-review subagent found no material defects.
- The overlay JS loops over every series, so the closet line works in the tooltip, the chips and the y-range without special code.
- Wrapping `#cabinetSensorGrid` does not change how the tiles render.
- The shared asyncpg connection stays usable after the missing-table error, because no transaction is open.
- The tests check real behaviour, not a coincidence.

Two cosmetic notes stay unfixed, and both are acceptable:
- The red "No snapshot" box now sits inside the new bordered block.
- When the closet table is missing, the status line does not say so. The table exists live, so this does not happen today.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run this from the clean primary `main` checkout after the merge, or let the post-pull rebuild do it.

## Risks / concerns

- Severity: low
- Concern: the closet line is a little brighter than the other lines.
- Mitigation: the chips and the tooltip name each line.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3128

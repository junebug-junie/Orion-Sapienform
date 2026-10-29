## Summary

The cabinet temperature sensor never moved. Juniper confirmed this on 10-29. On 10-16/17 she brought up the vision services. So the ~2 °C rise in the cabinet reading after 2026-10-17 00:35 UTC is real heat. Earlier work called it a sensor move and subtracted 2 °C from every reading. This patch removes that subtraction everywhere.

- The heat price now compares the raw cabinet reading with the 31 °C line. Before, it lowered the reading by 2 °C first, so the room looked 2 °C cooler than it was.
- The heat fit (`scripts/analysis/fit_heat_sensitivity.py`) no longer has a fixed 2 °C offset. The 10-17 change is now a free step: the fit measures its size as heat.
- `config/physiology/heat_sensitivity.yaml` is refit with the same command and rewritten by `--write-config`. It has no `measurement_offsets` list.
- The offset code is removed, not just switched off. A config that still carries `measurement_offsets` is refused, and heat reads `no_reading` with the reason.
- The seed row text, the READMEs, the design doc, the findings doc and three PR reports are corrected. The live database row needs one UPDATE (below). It is not written by this patch.

## Outcome moved

- The live heat price stops reading 2 °C too cool. Live, now: the deployed node says "cabinet 27.7 C (sensor offset -2.0 C, cabinet sensor geometry changed (vision bring-up)) -> 25.7 C". The new code says "cabinet 27.7 C".
- Read-only live smoke over the last 6 h with the new code: heat was **binding on 82 of 171 ticks**. That means Orion's own load is what pushed the cabinet over the line on those ticks. Strain was driven by heat on 98 ticks. The body economy is in shadow, so nothing acts on this yet.
- Replay of the live rules over 10-07..10-23 with the raw reading: R1 PASS for both forecasters. R3 PASS for oracle (longest unflagged ceiling run 3 h). R3 FAIL for persistence (6 h run, oscillation 0.50). 155 of 384 persistence hours are `exogenous_bind` ("the room is too hot, not my load"). With the offset, the 10-27 run had R3 failing with 9 h (persistence) and 17 h (oracle) runs.

## Current architecture

- `prediction_change_event` row `6d1f5c0a-…1003` (seeded by #2780) is a `break` at 2026-10-17 00:35 titled "cabinet sensor geometry changed (vision bring-up)".
- orion-prediction treats every `break` the same way: a level shift, then train only after it. It never reads the title and never treats a break as a measurement offset. That behaviour is still right for real heat, so its code does not change.
- The heat fit found that break by the word "sensor" in its title and subtracted a fixed 2 °C from the reading after it. `--write-config` copied that as `measurement_offsets` into the yaml.
- `orion/physiology/heat_config.py` parsed `measurement_offsets`, and `orion/physiology/budget.py` subtracted the offset before comparing with the soft line. The replay and smoke scripts did the same.
- The 34 °C hardware reflex never used the offset (see "34 °C reflex" below).

## Architecture touched

- Heat fit: every change-event break is a free step. The `*_offset_fixed` models are gone. `proxy_no_steps` and `house_meter_ac_no_steps` keep the "heat only through watts" bound.
- Live heat price (durable-runs `physiology` node through `orion/physiology/budget.py`): raw reading.
- Config contract `heat_sensitivity.v1`: `measurement_offsets` is no longer accepted.
- No bus, schema, channel or env change.

## Files changed

- `scripts/analysis/fit_heat_sensitivity.py`: remove `SENSOR_OFFSET_C`, `fixed_offsets`, the title match and the offsets in the config document. Rename the no-step variants.
- `config/physiology/heat_sensitivity.yaml`: refit, written by `--write-config`.
- `orion/physiology/heat_config.py`: remove `MeasurementOffset`, `offset_at`, `CABINET_SERIES`. Refuse a non-empty `measurement_offsets`.
- `orion/physiology/budget.py`: heat uses the raw reading. The `why` text no longer names an offset.
- `orion/physiology/tests/test_budget.py`: fixture without offsets; levels moved down 2 °C where a test meant the corrected value; new tests for the raw read, the refusal, the checked-in yaml, and the live 10-27 reading binding.
- `scripts/analysis/replay_body_economy.py`, `scripts/analysis/smoke_body_economy_prices.py`: raw reading.
- `scripts/analysis/tests/test_body_economy_replay_and_fit.py`: synthetic data now has real heat after the break; tests that the fit estimates it, that no fixed-offset path is left, and that the config has no offsets.
- `services/orion-sql-db/manual_migration_prediction_change_events_v1.sql`: corrected comment and seed text (fresh databases only; see SQL below).
- `services/orion-prediction/app/events.py`, `services/orion-prediction/README.md`, `services/orion-durable-runs/README.md`: corrected text.
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`, `docs/superpowers/specs/2026-10-23-body-economy-heat-and-replay-findings.md`, and the PR reports for #2780 (2026-10-20), #2868 (2026-10-23) and Patch B (2026-10-24): dated correction notes.

## Why a free step for the 10-17 change, not "explained by watts"

- The 10-17 change is real heat. The honest question is how much of it the measured watts explain.
- Without a step (`proxy_no_steps`), the fit must put the whole rise on athena's watts. It then gives athena +3.68 °C per 100 W held (CI +1.54..+6.42). That is about 6 times the short-run value, and it still leaves +2.99 °C of the 09-26..10-10 → 10-18..10-22 rise unexplained. athena's typical watt swing is only 59 W. So measured watts cannot carry this rise.
- With a free step, the fit measures the 10-17 heat at **+3.13 °C (CI +1.25..+4.14)** and leaves no residual. The heat comes from something the compute-watt columns do not see (for example the new hardware's own draw or changed airflow).
- Limit: the T10 swap (10-15) and the vision bring-up (10-17) are 1.5 days apart. The data cannot split them cleanly. Together they are about 4 °C in both the old and new fits (old: T10 +1.94 + fixed 2.00; new: T10 +0.95 + vision +3.13).
- The live config does not depend on this choice. It uses the hourly first-difference model, which never used the offset.

## Coefficients before and after

Same command both times: `scripts/analysis/fit_heat_sensitivity.py --start 2026-09-26T00:00:00Z --end 2026-10-27T00:00:00Z --reps 1000 --seed 20261023`. "Old code, today's data" is the unmodified script run on the same extract as the new one, to separate the code change from data drift.

Live config (short-run, °C per 100 W step within 2 h):

| host | yaml on main (fit 10-27) | old code, today's data | new code, today's data (written) |
|---|---|---|---|
| athena | +0.6863 (CI +0.2336..+1.0596), n 707, baseline 223.6 W | +0.643 (CI +0.197..+1.035) | **+0.6425 (CI +0.1970..+1.0345), n 661, baseline 223.2 W** |
| circe | +0.2985 (CI +0.1448..+0.4491), n 707, baseline 263.1 W | +0.311 (CI +0.145..+0.476) | **+0.3111 (CI +0.1452..+0.4756), n 661, baseline 262.0 W** |

The short-run numbers are identical between old and new code on the same data. The shift from main's yaml is data drift since 10-27 (n 707 → 661, baselines moved), not this fix. Cause of the drift: UNVERIFIED (new change events since 10-27 are one part; the baseline change means some watt rows in the window also differ).

Day-level primary model (°C per 100 W held), old code vs new code, same data:

| | athena | circe | T10 step | 10-17 step |
|---|---|---|---|---|
| old primary (offset fixed 2.0) | +0.474 (−1.221..+1.644) | +0.103 (−0.244..+0.549) | +1.94 (+0.62..+3.16) | fixed 2.00 |
| new primary (free step) | +0.527 (−1.207..+1.684) | +0.123 (−0.229..+0.583) | +0.95 (+0.11..+2.70) | +3.13 (+1.25..+4.14) |
| old no-steps (offset fixed) | +2.232 (+0.496..+4.082) | +0.127 | — | fixed 2.00 |
| new no-steps | +3.675 (+1.544..+6.416) | +0.184 | — | — |

## 34 °C reflex

The 34 °C hardware reflex never used this offset. `services/orion-hardware-watch/app/watcher.py` `_cabinet()` reads the raw `cabinet_temp_c` through `services/orion-hardware-watch/app/store.py` `temp_points()` (a plain `SELECT … measurements->>'cabinet_temp_c' FROM orion_biometrics_summary`). Nothing in `services/orion-hardware-watch/app/` imports `orion.physiology`. The gpu-pool and orion-thought thermal lines do not import `heat_config` either.

## SQL for Juniper (live row)

The seed migration uses `ON CONFLICT DO NOTHING`, so a deploy does **not** fix the existing row. The post-pull applier also skips it (the table already exists). Run this once by hand:

```bash
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "UPDATE prediction_change_event SET title = 'vision services brought up (real heat)', notes = 'cabinet reads ~2 C higher after: real heat, not a sensor move (the sensor never moved; Juniper 2026-10-29). lidar 116 -> 94 mm, magnetic 36 -> 33.5 uT: something near the sensor changed, likely the new vision hardware.' WHERE event_id = '6d1f5c0a-6a11-4c6e-9d2b-0a1e5b7c1003';"
```

It changes only the text. orion-prediction's selection fingerprint uses the times, not the title, so no forecaster reselect happens.

## Schema / bus / API changes

- Added: none.
- Removed: `measurement_offsets` in `heat_sensitivity.v1` (now refused when non-empty). Python: `MeasurementOffset`, `offset_at`, `CABINET_SERIES`, `HeatSensitivityConfig.offsets`.
- Renamed: fit models `proxy_offset_fixed_no_steps` → `proxy_no_steps`, `house_meter_ac_offset_fixed_no_steps` → `house_meter_ac_no_steps`. `sensor_step_estimated` and `house_meter_ac_offset_fixed` are gone (the first is now `primary`).
- Behavior changed: the heat price reads the raw cabinet temperature.
- Compatibility notes: the yaml is baked into the durable-runs image, so code and yaml change together on rebuild.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
pytest -q orion/physiology/tests scripts/analysis/tests/test_body_economy_replay_and_fit.py   180 passed
pytest -q services/orion-durable-runs/tests/test_physiology_node.py  15 passed, 4 skipped
services/orion-prediction tests: 156 passed, 8 failed locally; the same failures happen on main in this
  venv (prophet is not installed here). Only a docstring changed in orion-prediction. CI installs it.
py_compile on every changed script; git diff --check clean.
```

## Evals run

```text
Refit (above), replay_body_economy.py 10-07..10-23 with the new config (Patch B gate numbers above).
smoke_body_economy_prices.py --hours 6 (live, read-only): heat binding 82 / slack 89; all checks ok.
```

## Docker/build/smoke checks

```text
No Docker build run in this patch. The live path is UNVERIFIED until durable-runs is rebuilt from main.
```

## Review findings fixed

Code review ran in a subagent on commit d06f56f54.

- Finding (must-fix): `replay_body_economy.py` `run_wants()` still called the removed `corrected_hours()`. Any `--wants` run with a heat config would crash with `NameError`.
  - Fix: pass the raw hours, the same way `run_patch_b` does. Added `test_wants_replay_runs_with_a_heat_config_on_the_raw_reading`.
  - Evidence: with the old line put back, the new test fails with `NameError: name 'corrected_hours' is not defined`. With the fix, it passes. pyflakes reports no undefined names.
- Finding: the SQL comment and the orion-prediction README point to this PR report, and the report was not committed yet.
  - Fix: this report is committed with the exact UPDATE.
  - Evidence: the file is in this PR.
- Finding: two budget tests do not catch the old bug, because the fixture has no offsets.
  - Fix: their docstrings now say so. The real guards are the parser-refusal test and the checked-in-yaml test.
  - Evidence: `test_budget.py` docstrings.
- Finding: the raw-reading fit test used a 6-day window that ends before the break, so it passed without testing anything.
  - Fix: 12 days. The test now asserts that rows exist on both sides of the break.
  - Evidence: 180 passed.
- Finding: the yaml numbers moved because the input data changed, not because of the fix.
  - Fix: the report already showed old code and new code on the same extract. Both give the same short-run numbers. The data drift is named in Risks.
  - Evidence: "Coefficients before and after" table.
- Finding: the heat price will bind more after deploy. Strain will also rise.
  - Fix: named in Risks.
  - Evidence: see Risks.

## Restart required

durable-runs bakes `config/physiology/heat_sensitivity.yaml` into its image (Dockerfile `COPY`), so it needs a rebuild, not a restart. After merge, from the clean primary main checkout:

```bash
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Then check: `curl -s localhost:8124/physiology/state` — the heat `why` should start "cabinet NN.N C;" with no "sensor offset".

orion-prediction: no rebuild needed (docstring and README only).

## Risks / concerns

- Severity: medium. Concern: the heat price will bind much more often (82 of 171 ticks in the last 6 h), so `strain` will rise after deploy. This is not a regression: the reading is now true. Mitigation: shadow only; nothing that decides reads the price or strain today. Check that before Patch C wires bids to it. The separate heat gates (the 34 °C reflex, gpu-pool and thought thermal lines) never imported `heat_config`, so they do not change.
- Severity: low. Concern: the T10 and vision steps cannot be split cleanly (1.5 days apart). Mitigation: reported as a pair; the live config does not use either step.
- Severity: low. Concern: in the closed window 09-26..10-27, the fit input lost 46 hourly rows between 10-27 and 10-29 (n 707 → 661), and the baseline watts moved. So the short-run numbers moved for a reason that is not this fix. Cause UNVERIFIED (retention, backfill or missing 15-minute bins). Follow-up: find out why rows left a closed window.

## PR link

PR_LINK_PLACEHOLDER

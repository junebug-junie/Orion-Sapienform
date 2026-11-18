## Summary

- The Hub's Ambient audio card now shows the office sound meter first: a dBA chart with the hover readout, plus tiles for now, typical, quiet floor, loudest, time at 60 dBA or more, age and status.
- The card says plainly that this meter is in Juniper's office, not in the cabinet or the closet. The cabinet mic section below it is labelled "uncalibrated, not dB".
- The Particulate tile is gone from "Cabinet sensors — Athena Nanos". That sensor is on pan-1's office Nano, so the tile only ever read "absent". The office strip already shows PM.
- Orion's situation brief (unified chat) gets a new line with the office level in dBA, the last 10 minutes' typical level, how that compares with the office's last 24 hours, and a dB scale. The cabinet mic line stays as it was.

## Outcome moved

Orion can now say how loud the office is in decibels. Before, Orion only knew "the cabinet is louder or quieter than usual" from an uncalibrated mic. Juniper can now see the office level next to the cabinet mic on one card.

## Current architecture

- Office meter: DFRobot SEN0232 on pan-1's office Nano → orion-office-env (`office_sound_dba`) → `office_env_sample.payload_json` → Hub `/api/office/env/*` → Office strip tile and chart (PR #3154). No Orion consumer.
- Ambient audio card: cabinet USB mic only (`cabinet-sensors.js`).
- Situation brief: cabinet mic as louder/quieter than usual, no numbers.

## Architecture touched

- Hub template and static JS (`index.html`, `office-env.js`, `cabinet-sensors.js`).
- Shared situation builder (`orion/situational/`), schema `orion/schemas/situation.py`.
- No bus, channel, env key, or SQL change.

## Files changed

- `services/orion-hub/templates/index.html`: office sound block above the cabinet mic tiles and charts in the Ambient audio card.
- `services/orion-hub/static/js/office-env.js`: `OFFICE_SOUND` strip; `historyTiles` and `ownWindowButtons` options on `createStrip`; pure `soundSummary` / `summaryText`.
- `services/orion-hub/static/js/office-env.test.js`: tests for the summary and the strip config.
- `services/orion-hub/static/js/cabinet-sensors.js`: Particulate tile removed.
- `services/orion-hub/tests/test_cabinet_sensors_panel.py`: pins the tile removal and the office block's place and note.
- `orion/situational/office_sound_reader.py`: one fail-open query for the newest reading and the 24 h band.
- `orion/situational/context.py`: builder, settings, cache staleness gate, prompt line.
- `orion/schemas/situation.py`: `OfficeSoundContextV1`, `SituationBriefV1.office_sound`.
- `orion/situational/tests/test_situation_office_sound.py`: tests.

## Schema / bus / API changes

- Added: `OfficeSoundContextV1`; `SituationBriefV1.office_sound` (defaults to unavailable).
- Removed: none. Renamed: none.
- Behavior changed: one more line in the situation prompt fragment when the meter is fresh.
- Compatibility notes: additive, with a default. No other service validates `SituationBriefV1`.

## Env/config changes

- Added keys: none. Hub reuses its existing `OFFICE_ENV_NODE` and `OFFICE_ENV_STALE_AFTER_SEC`.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Skipped keys requiring operator action: none.

## Metric quality gate (office_sound_dba reaching Orion's prompt)

1. Provenance: `services/orion-office-env/app/sample.py` `_sound_reading` (mV × 0.05 = dBA, SEN0232 range 30–130).
2. Independence: a separate sensor in a separate room from the cabinet mic. No shared upstream.
3. Theory anchor: SEN0232 is a calibrated dBA sound level meter (±1.5 dB per DFRobot). Juniper's reference meter read ~74 dB while it read 74–77.
4. Live data (2026-11-07 02:20 UTC): 50 rows in the first ~33 min, range 37.4–90.2 dBA, newest 74–75. It moves. The low 37.4 values are from the earlier wrong-lead wiring (memory note on PR #3154), so the 24 h band is polluted until they age out (~2026-11-08 02:00 UTC). A quiet room should read ~35–45; not yet seen live.
5. Existing mechanism: the cabinet mic reader is the model; the engine and row floor are reused from it.
6. Reversibility: cheap. One brief field with a default and one prompt line.

## Tests run

```text
pytest orion/situational/tests -q                         306 passed, 1 skipped
pytest services/orion-hub/tests/test_cabinet_sensors_panel.py test_office_env_routes.py test_cabinet_ambient_api.py   88 passed
node --test office-env.test.js (after review fixes)         16 pass
node --test services/orion-hub/static/js/*.test.js        411 pass, 0 fail
check_metric_lineage.py --gate / --prompt-semantics       PASS / PASS
check_definition_drift.py --gate                          PASS
check_inner_state_registry.py                             OK
tests/test_metric_prompt_semantics.py tests/test_metric_rest_range.py   67 passed
```

Live reader check against Postgres: `fetch_office_sound(node="pan-1")` returned 75.0 dBA, recent median 75.25, band None (fewer than 60 rows so far). An unknown node returned None.

## Evals run

No eval harness covers the situation prompt line. The unit tests render the real fragment.

## Docker/build/smoke checks

```text
Not run in this branch. Hub UI check after deploy: open Biometrics -> Cabinet, confirm the office block draws and the hover reads dBA.
```

## Review findings fixed

Review ran in a subagent. No correctness bugs found. Fixed:

- Finding: the newest row with no sound value (an out-of-range pin read) made the brief drop the office line, even though a reading 30 s older existed.
  - Fix: the `latest` query now takes the newest row that has a sound value. The 90 s staleness gate still limits its age.
  - Evidence: live read after the fix returned 78.1 dBA, band 37.5 / 76.1 / 78.2, 10-min median 77.4.
- Finding: after a failed history fetch, the summary tiles kept the previous window's numbers under the newly selected button.
  - Fix: the tiles reset to "—" on a history error.
- Finding: the new click guard also stopped a click on the active button from refreshing the existing Office and Closet strips.
  - Fix: the guard now applies only to the strip that shares the Ambient buttons.

Not fixed:

- The Ambient card and the Office strip each call `/api/office/env/*` every 30 s. These calls are cheap, so this is accepted.
- `cabinet_particulate_activity` has no input on the Athena Nanos. The field digester and the situation code also read it, so removing it is a separate change. It only shows as a chip when it has a value, so the Hub shows nothing for it.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

Run from the clean primary main checkout after merge (the post-pull rebuild does this).

## Risks / concerns

- Severity: low. Concern: the 24 h band includes ~30 min of wrong-lead 37.4 dBA readings until about 2026-11-08 02:00 UTC, so "quieter than usual" is less likely for that first day. Mitigation: it ages out on its own.
- Severity: low. Concern: on 3d/7d windows, typical, floor and time at 60 dBA come from bucket-averaged points; loudest uses the raw max. Mitigation: 24h is close to raw. On longer windows, read these tiles as approximate.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

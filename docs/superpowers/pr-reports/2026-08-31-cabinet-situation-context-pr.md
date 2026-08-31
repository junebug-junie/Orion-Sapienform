# PR report: Orion's own cabinet sensors in the chat-turn situation brief

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/2008
Branch: `feat/cabinet-situation-context`
Status: **DONE**

## Summary

- Adds Orion's own physical cabinet sensors (Athena Nano ESP32 node: BME680 climate, LTR390 UV/ALS, magnetometer, PMSA003I particulate, VL53L1X lidar, BNO085 IMU/vibration) as a new `CabinetContextV1` section in the unified chat-turn situation brief.
- `LabContextV1` was always a stub (`_build_lab_context` unconditionally returns `available=False`) -- Juniper: "we had stubbed these as lab, but it is much richer!" `LabContextV1` stays untouched (a distinct, still-unwired GPU-cluster concept); this is a new, separate section.
- Reuses the exact same shared `orion.telemetry.cabinet_sensors` / `cabinet_snapshot_merge` helpers `services/orion-hub/scripts/cabinet_sensors_routes.py`'s own `/api/cabinet/sensors/latest` route already uses -- not an import of that route module (`orion/` is shared code services import FROM, never the reverse).
- ON by default in orion-hub (the only process with the `/run/orion-sensors` host bind mount) -- carries no private-home content, same call already made for curiosity/reverie. Off by default everywhere else.
- Rendered omitted-when-unavailable (like curiosity/reverie), not as an always-on placeholder (like weather/lab/perception) -- an always-on line reproduced this file's own documented 2026-08-26 budget regression, caught live while writing this feature and fixed before the first commit.
- A backgrounded `/code-review` pass (4 parallel sub-agents) found 9 issues; 7 material findings fixed in a follow-up commit, 2 accepted as documented trade-offs.

## Outcome moved

Orion's chat-turn situation brief can now carry a real, live reading of its own physical housing instead of the permanently-stubbed "Lab: unavailable/stub; do not infer." line.

## Current architecture

`orion/situational/context.py`'s `build_situation_for_ctx()` assembles a `SituationBriefV1` from a fixed set of `_build_*` providers, each fail-open and TTL-cached, rendered to a bounded-length prompt fragment by `_build_prompt_fragment()`. `_build_lab_context()` has always been a no-op stub. Separately, `services/orion-hub/scripts/cabinet_sensors_routes.py` already reads live cabinet sensor snapshots from `/run/orion-sensors/latest.json` (host-bind-mounted into orion-hub via `network_mode: host`) for Hub's own `/api/cabinet/sensors/*` operator routes -- but that data never reached the chat-turn prompt.

## Architecture touched

- `orion/schemas/situation.py`: new `CabinetContextV1`, new `cabinet` field on `SituationBriefV1`.
- `orion/situational/context.py`: new `_fetch_cabinet_context`/`_build_cabinet_context` provider (TTL-cached, fail-open), new `SituationSettings` fields, `settings_from_runtime`/`hub_settings_to_runtime_namespace` wiring, new `_build_prompt_fragment` render block, new shared `_recency_phrase()` helper.
- `orion/telemetry/cabinet_snapshot_merge.py`: new shared `device_label_from_sources()` helper.
- `orion/schemas/registry.py`: `CabinetContextV1` added to `_REGISTRY`.
- `services/orion-hub/app/settings.py` + `.env_example`: new `ORION_SITUATION_CABINET_ENABLED`/`ORION_SITUATION_CABINET_TTL_SECONDS` keys. No new sensor-path keys -- reuses Hub's existing `CABINET_SENSORS_PATH`/`CABINET_SENSORS_B_PATH`/`CABINET_SENSORS_STALE_AFTER_SEC`.

## Files changed

- `orion/schemas/situation.py`: new `CabinetContextV1` schema + `SituationBriefV1.cabinet` field.
- `orion/situational/context.py`: new provider, settings wiring, prompt rendering, review fixes.
- `orion/telemetry/cabinet_snapshot_merge.py`: new `device_label_from_sources()` shared helper.
- `orion/schemas/registry.py`: registered `CabinetContextV1`.
- `services/orion-hub/app/settings.py`: two new settings fields.
- `services/orion-hub/.env_example`: two new documented keys.
- `orion/situational/tests/test_situation_cabinet_context.py` (new): 35 tests.
- `orion/situational/tests/test_hub_settings_adapter.py`: 2 new assertions.

## Schema / bus / API changes

- Added: `CabinetContextV1` (orion/schemas/situation.py), `SituationBriefV1.cabinet` field (additive, defaults `available=False`).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing sections.
- Compatibility notes: purely additive; old payloads still validate. Not bus-published under its own `kind`.

## Env/config changes

- Added keys: `ORION_SITUATION_CABINET_ENABLED` (default `true`), `ORION_SITUATION_CABINET_TTL_SECONDS` (default `30`).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys`: yes -- confirmed `ORION_SITUATION_CABINET_ENABLED='true'`/`ORION_SITUATION_CABINET_TTL_SECONDS='30'` landed in the primary checkout's `services/orion-hub/.env`. (`ORION_SITUATION_*` is not in the script's default `SYNC_PREFIXES` allowlist, same as the existing curiosity/reverie keys -- `--all-keys` was required.)
- skipped keys requiring operator action: none.

## Tests run

```
PYTHONPATH=<worktree> .venv/bin/python -m pytest orion/situational/ tests/test_cabinet_sensors.py \
  services/orion-cortex-exec/tests/test_situation_curiosity_reverie_context.py \
  services/orion-cortex-exec/tests/test_situation_affect_context.py \
  services/orion-cortex-exec/tests/test_situation_perception_context.py \
  services/orion-cortex-exec/tests/test_situation_input_modality.py -q
176 passed
```

## Evals run

No dedicated eval harness exists for `orion/situational/` -- gate tests above are the only coverage lane for this module (matches curiosity/reverie precedent).

## Docker/build/smoke checks

Not run -- pure Python library change plus new env keys read by existing settings plumbing. `orion-hub`'s app code is bind-mounted (not baked into the image), so a plain restart is sufficient -- no `--build` needed.

## Review findings fixed

- Finding: Data race on `_CABINET_TRACKER`'s EWMA state via concurrent `asyncio.to_thread` calls.
  - Fix: dedicated `_CABINET_TRACKER_LOCK` around the `compute_cabinet_pressures` call.
  - Evidence: `test_tracker_mutation_holds_the_dedicated_lock`.
- Finding: Malformed empty prompt line (`"...: ."`) when `available=True` but no rendered field populated.
  - Fix: `if parts:` guard, omit the line.
  - Evidence: `test_available_with_no_renderable_field_omits_line_not_malformed`.
- Finding: Missing `uv_activity` in the "notable" list; unverified 0.6 threshold borrowed from an unrelated context (CLAUDE.md 0A step 3/4).
  - Fix: removed the interpretive "Notably: elevated X" narration entirely; raw activity numbers stay on the schema, undisplayed until calibrated.
  - Evidence: `test_activity_pressures_are_not_narrated`.
- Finding: `x or DEFAULT` silently discarding an explicit `0.0`/`""` override, twice.
  - Fix: plain `getattr` defaults, no `or` fallback.
  - Evidence: `test_cabinet_stale_after_sec_zero_is_not_silently_replaced`, `test_hub_adapter_cabinet_stale_after_sec_zero_is_not_silently_replaced`, `test_hub_adapter_cabinet_sensors_path_empty_string_is_not_silently_replaced`.
- Finding: only 6 of the schema's measurement fields were ever rendered.
  - Fix: widened the rendered `parts` list (added pressure_hpa/uv_raw/pm25_ug_m3).
  - Evidence: `test_widened_measurement_fields_render`.
- Finding: device-label derivation duplicated verbatim from `cabinet_sensors_routes.py`.
  - Fix: extracted `device_label_from_sources()` to `orion/telemetry/cabinet_snapshot_merge.py`.
  - Evidence: covered by `test_fetch_live_snapshot_populates_measurements_and_pressures`.
- Finding: `age_min`/"just now"/"N min ago" duplicated a 3rd time (perception/affect/cabinet).
  - Fix: extracted `_recency_phrase()`, used by all three.
  - Evidence: existing perception/affect render tests still pass unchanged.
- Finding (accepted, no change): a 3rd independent `CabinetSensorTracker` instance means the same physical event can score differently across consumers.
  - Rationale: sharing state across processes/purposes would let one consumer's read cadence silently perturb another's baseline; already documented in the tracker's own comment.
- Finding (accepted, no change): cautions can still be dropped if every section renders maximally at once at the production 7200-char cap.
  - Rationale: pre-existing property of the truncate-then-reserve-cautions design, not introduced by this feature; regression test confirms this section alone does not trip it even at the tight historical 1200-char cap.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml restart hub-app
```

## Risks / concerns

- Severity: low
  - Concern: `*_activity` fields are populated but not yet narrated -- interpretive language deferred per the metric-quality gate rather than shipped with an uncalibrated threshold.
  - Mitigation: follow-up task to pull real cabinet sensor history and calibrate a per-domain threshold before adding narration back.
- Severity: low
  - Concern: a 3rd independent `CabinetSensorTracker` baseline means the same physical event can read differently across consumers (biometrics, Hub's operator routes, this chat-brief consumer).
  - Mitigation: documented trade-off, consistent with the precedent Hub's own routes already set.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2008

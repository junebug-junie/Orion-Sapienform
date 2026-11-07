## Summary

- Orion's office sensor array now has a sound level meter (DFRobot SEN0232 / Robu RB-Dfr-792) on the office Nano's A0 pin.
- Firmware sends `sound.level_mv` (mean pin voltage per 1 s frame). Host converts to dBA (volts x 50) and range-checks 30..130 dBA.
- New measurement `office_sound_dba` reaches the Hub office strip (tile + chart) and `payload_json` in `office_env_sample`.
- The meter is read off the I2C bus, so it works with the multiplexer unplugged.

## Outcome moved

Orion can hear how loud the office is. Before: no sound sensing on pan-1.

## Current architecture

Office Nano (SHT40, CCS811, PMSA003I, TSL2591 behind a TCA9548A) -> USB serial NDJSON frame -> `orion-office-env` on pan-1 -> `orion:office:env:sample` -> `orion-sql-writer` -> `office_env_sample` -> Hub office strip. Closet Nano runs the same firmware.

## Architecture touched

- `firmware/pan-1-office-nano`: analog read of A0, pull-down presence test, 10 ms sampling, mean per frame.
- `orion/schemas/telemetry/office_env.py`: frame `sound` block, measurement `office_sound_dba`.
- `services/orion-office-env/app/sample.py`: volts-to-dBA conversion and range gate.
- `services/orion-hub`: route payload field + stats key, template tile + chart, JS tile + chart.
- No new bus channel, no registry change (same payload kind, new optional field), no env keys, no SQL migration (Hub and writer already carry `payload_json`).

## Files changed

- `firmware/pan-1-office-nano/pan-1-office-nano.ino`: sound sampling, `init_sound`, `add_sound`.
- `firmware/pan-1-office-nano/README.md`: wiring, frame example, object table.
- `orion/schemas/telemetry/office_env.py`: `OfficeEnvFrameSoundV1`, `office_sound_dba`.
- `services/orion-office-env/app/sample.py`, `README.md`: conversion, stale rule, deploy order.
- `services/orion-hub/scripts/office_env_routes.py`, `templates/index.html`, `static/js/office-env.js`, `README.md`: display.
- Tests: `orion-office-env/tests/test_sample.py`, `test_main.py`; Hub `test_office_env_routes.py`, `test_office_env_panel.py`, `office-env.test.js`; `orion-sql-writer/tests/test_office_env_sample_sql_shape.py`.

## Schema / bus / API changes

- Added: frame `sound.level_mv`; measurement `office_sound_dba` (30..130, optional).
- Removed / renamed: none.
- Behavior changed: a frame with only a valid `sound` reading is fresh (not stale).
- Compatibility: the key is omitted unless the meter reads, so old consumers keep working until the meter publishes. Models are `extra="forbid"`: rebuild `orion-sql-writer` before the producer sends the key.

## Env/config changes

None. No `.env_example` changed; `sync_local_env_from_example.py` not needed.

## Metric quality gate (office_sound_dba)

1. **Provenance.** Producer: `add_sound()` in the firmware (`analogReadMilliVolts(A0)`, mean per frame) -> `_sound_reading()` in `sample.py`. Traced to code, not a schema comment.
2. **Independence.** Acoustic pressure at the desk. None of SHT40, CCS811, PMSA003I, TSL2591 shares an upstream sensor or a transform with it. Not redundant.
3. **Theory anchor.** DFRobot SEN0232 datasheet: A-weighted SPL, factory calibrated, output 0.6..2.6 V = 30..130 dBA (dBA = V x 50), +-1.5 dB, 125 ms time weighting. [wiki](https://wiki.dfrobot.com/sen0232/).
4. **Live-data sanity check: UNVERIFIED.** The meter is wired (A0, VBUS, GND) but the new firmware is not flashed yet, and the multiplexer was unplugged during the session. Rest-state note: the meter floors at 30 dBA (0.6 V), so a silent room reads exactly 30.0, not lower; a quiet office (30-40 dBA) should sit above it. To check after flashing: confirm the values vary with a clap and settle in a quiet room, and that the closet Nano sends no `sound`.
5. **Existing mechanism.** The repo has cabinet ambient audio (`scripts/orion_ambient_audio_reader.py`, `orion/situational/cabinet_sound_reader.py`, field glossary entries): a microphone on Athena's cabinet, a different room and instrument. It does not measure the office and is not calibrated SPL. This adds a second, independent sound source and does not replace it. Nothing consumes `office_sound_dba` beyond the Hub and the stored payload yet; wiring it into situational context is a separate patch.
6. **Reversibility.** Optional key, no column, no migration. Remove by deleting the key; nothing is baked into a schema default.

## Tests run

```text
office-env:   46 passed
hub:          34 passed (office routes + panel); node --test office-env.test.js: 13 passed
sql-writer:   14 passed (office sample shape)
tests/test_office_env_bus_catalog.py: 3 passed
check_metric_lineage.py: exit 0; check_metric_dead_wiring.py: exit 0; check_env_template_parity.py: PASS
```

## Evals run

The office-env service has replay evals only for outage and air warm-up; none cover the meter. Eval gap: add a sound replay eval once there is a recorded capture from the flashed board.

## Docker/build/smoke checks

```text
arduino-cli compile --fqbn arduino:esp32:nano_nora firmware/pan-1-office-nano: OK (332457 bytes, 10%)
```
Built with ArduinoJson 6.21.5 and the documented libraries in a scratch library dir. Not flashed. No Docker build run yet.

## Review findings fixed

- Finding (should): the pull-down presence test did not pull down. arduino-esp32 re-runs `pinMode(pin, ANALOG)` on every analog read, which switches the pull off.
  - Fix: discharge via pull-down then read within microseconds (5 ms settle), at boot, on the retry path and once per frame, so an unplugged meter is also dropped. Core source read at `esp32-hal-adc.c` `__analogReadRaw` -> `__adcAttachPin` -> `pinMode(pin, ANALOG)`.
  - Evidence: compiles; hardware behaviour still UNVERIFIED (unplug test).
- Finding (should): a sound-only frame is fresh and `sensor.online` is true.
  - Fix: kept on purpose (works with the multiplexer unplugged); `online` docstring now says so.
- Finding (should): deploy order. Checked `rg OfficeEnvSampleV1|office.env.sample` in services: only `orion-sql-writer` validates the sample model (prediction does not); Hub reads the DB. README order stands: sql-writer first, Nano last.
- Nits: mean-of-dBA vs energy average is now noted in the firmware README; `_NEW_COLUMNS` in the Hub test means "never a bare SQL column" and still holds.

## Restart required

```bash
# 1. athena, after merge: rebuild the consumer first
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# 2. pan-1: pull main, rebuild the office reader
# 3. flash the office Nano last (see firmware README, DFU -S E8F60AD53FB4)
```

## Risks / concerns

- Severity: medium. Presence is a discharge probe, not a real identity check. An unplugged meter should read ~0 and be dropped within a frame, but this is UNVERIFIED on hardware until the unplug test is run. Mitigation: probe every frame.
- Severity: low. A 5 V meter on VBUS: output is under 3.3 V, so the pin is safe.

## PR link

TBD

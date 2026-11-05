## Summary

- Onboards two new office sensors on the pan-1 office Nano: **PMSA003I** particulates (PM1 / PM2.5 / PM10) and **TSL2591** light on the windowsill.
- Onboards a **second Nano** on pan-1 with an **SCD40** (real CO2, temperature, humidity) in the server closet, above the server cabinet. It publishes its own `closet.env.sample.v1`, not an office sample.
- One firmware runs on both Nanos (same pattern as Athena's two cabinet Nanos). Each sensor's multiplexer channel is now **found at start-up**, not hard-coded, so moving a cable needs no reflash.
- sql-writer stores the new office columns and a new `closet_env_sample` table. Hub shows PM and light tiles and a closet strip.
- Adds the missing CI job for `orion-office-env`; until now no workflow ran its tests.

## Outcome moved

Before: pan-1 sensed temperature, humidity and eCO2/TVOC in the office only. After: Orion also stores office particulates and daylight, and the server closet's real CO2, temperature and humidity. Raw samples only: no pressure, alert, or cognition reads them yet (each needs the metric quality gate on real data first).

Live hardware evidence (2026-11-05, pan-1, new firmware on both boards):

```text
office: {"environment":{"temp_c":26.60,"humidity_pct":24.46},"air":{"eco2_ppm":422,...},
         "particulate":{"pm1_ug_m3":0,"pm25_ug_m3":0,"pm10_ug_m3":0},
         "light":{"full":2547,"ir":600,"gain_x":428,"saturated":false,"lux":14.19}}
closet: {"co2":{"co2_ppm":414,"temp_c":27.79,"humidity_pct":24.59}}
```

Multiplexer scans (TCA9548A 0x70): office — CCS811 ch2, SHT40 ch3, PMSA003I ch4 (plus an unidentified `0x78` on ch4), TSL2591 ch7. Closet — SCD40 ch7.

## Current architecture

One Nano ESP32 (SHT40 + CCS811, channels hard-coded) on pan-1 USB → `orion-office-env` → `orion:office:env:sample` → sql-writer `office_env_sample` → Hub office strip. Flashing assumed a single Nano.

## Architecture touched

- Firmware `firmware/pan-1-office-nano` (both boards).
- `orion-office-env`: one image, two containers (`ENV_ROOM=office|closet`), each pinned to its own `/dev/serial/by-id/` path.
- Contracts: office frame + sample extended; new closet sample, channel, registry entries.
- `orion-sql-writer`: boot DDL for new office columns; new closet table and route.
- `orion-hub`: office tiles and a closet strip; tolerates the not-yet-migrated DB.

## Files changed

- `firmware/pan-1-office-nano/pan-1-office-nano.ino`: PMSA003I, TSL2591 (auto-gain, saturation at the 100 ms full scale 36863), SCD40; channel discovery; SCD40 power-up wait.
- `firmware/pan-1-office-nano/README.md`: both boards, addresses, two-Nano DFU flashing by serial.
- `orion/schemas/telemetry/office_env.py`: frame `particulate` / `light` / `co2`; sample PM + lux fields, `light_saturated`.
- `orion/schemas/telemetry/closet_env.py` (new): `ClosetEnvSampleV1`.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`: closet schema (both registries) and channel.
- `config/biometrics/node_catalog.yaml`: pan-1 hardware + `orion-closet-env`.
- `config/metrics/metric_definitions.lock.json`: re-locked (new closet bus metric).
- `services/orion-office-env/app/{sample,main,settings,smoke}.py`, `docker-compose.yml`, `.env_example`, `README.md`, tests.
- `services/orion-sql-writer/app/{main,settings,worker}.py`, `app/models/{office_env_sample,closet_env_sample,__init__}.py`, `.env_example`, `README.md`, tests.
- `services/orion-hub/scripts/{office_env_routes,closet_env_routes,api_routes}.py`, `static/js/office-env.js`, `templates/index.html`, `README.md`, tests.
- `.github/workflows/orion-office-env-tests.yml` (new), `tests/test_office_env_bus_catalog.py`.

## Schema / bus / API changes

- Added: `orion:closet:env:sample` / `closet.env.sample.v1` / `ClosetEnvSampleV1`; office measurements `office_pm1_ug_m3`, `office_pm25_ug_m3`, `office_pm10_ug_m3`, `office_lux`; state `light_saturated`; frame objects `particulate`, `light`, `co2`; Hub `/api/closet/env/latest|history`.
- Removed: none.
- Renamed: heartbeat detail `office_sensor` → `sensor` (+ new `room`); no consumer outside the service.
- Behavior changed: office payloads are published with `exclude_none`, so absent readings are omitted instead of sent as `null`.
- Compatibility notes: office measurements/state are `extra="forbid"`. Because absent fields are omitted, a payload without PM/lux stays valid for the old sql-writer. A payload *with* PM/lux readings is not, so sql-writer must deploy before pan-1 (see Restart required).

## Env/config changes

- Added keys: `orion-office-env`: `CLOSET_SERIAL_DEVICE_HOST`, `CLOSET_ENV_SAMPLE_CHANNEL`, `CLOSET_SENSOR_ID`. `ENV_ROOM` is set per container by compose.
- Removed / renamed keys: none.
- `.env_example` updated: `orion-office-env`, `orion-sql-writer` (subscribe list + route map).
- local `.env` synced: athena — no office-env `.env`; sql-writer's list values show as "diverged" and need no edit (route is a merged code default, channel is force-subscribed). pan-1 — sync at deploy (below).
- skipped keys requiring operator action: `CLOSET_SERIAL_DEVICE_HOST` on pan-1 = `/dev/serial/by-id/usb-Arduino_Nano_ESP32_E8F60ABE42B4-if01`.

## Tests run

```text
orion-office-env tests+evals .................. 44 passed (also clean venv with only CI deps: 43+3 before the last test)
orion-sql-writer CI unit set ................... 1203 passed, 0 failed, 183 skipped (tests folder)
orion-hub focused (office/closet/overlay) ...... 51-58 passed; node office-env.test.js 13 pass
tests/test_office_env_bus_catalog.py + registry agreement ... 5 passed
orion-static-gates.yml, every step run locally ............ 40/40 PASS
```

## Evals run

```text
services/orion-office-env/evals (outage + air warm-up replays) ... pass (in the 44 above)
```

No eval yet for PM / light / closet values: there is no real history to replay. Follow-up once a week of rows exists.

## Docker/build/smoke checks

```text
docker compose config (both containers) ... office -> /dev/office-nano + ENV_ROOM=office;
  closet -> by-id E8F60ABE42B4 -> /dev/closet-nano, ENV_ROOM=closet, SERVICE_NAME=orion-closet-env
firmware compile (arduino:esp32 2.0.18) ... OK, 322869 bytes
live serial frames from both boards ....... see Outcome moved
running office container (old image) on new firmware ... fresh samples, temp/humidity intact
```

Image build of the new service code: UNVERIFIED (Dockerfile unchanged; it copies `orion/` whole).

## Review findings fixed

- Finding: TSL2591 saturation checked against 0xFFFF, but the 100 ms full scale is 36863, so clipped lux looked valid and `saturated` never fired.
  - Fix: compare against `TSL_MAX_COUNT_100MS = 36863`.
  - Evidence: firmware recompiled and reflashed; live frames show auto-ranging (25x → 428x).
- Finding: `full == 0` / `ir > full` produced NaN or garbage lux.
  - Fix: lux omitted unless measurable and finite.
- Finding: new office payloads carried `null` PM/lux keys, so the old `extra="forbid"` sql-writer would reject every office row if pan-1 deployed first.
  - Fix: publish with `exclude_none`; `light_saturated` is `None` unless true.
  - Evidence: `test_office_payload_stays_valid_for_a_pre_pm_light_consumer`.
- Finding: one `up` for both containers fails when the closet path is the placeholder or the Nano is unplugged.
  - Fix: README deploys each container by name.
- Finding: SCD40 could get commands under 1 s after power-up.
  - Fix: wait until `millis() >= 1000` before `init_scd40()`.

## Restart required

athena: merge, then the post-pull rebuild redeploys `orion-sql-writer` and `orion-hub`. Check the sql-writer boot log for `office_env_sample required columns ensured`, then the closet table:

```bash
PGPASSWORD=postgres psql -h localhost -p 55432 -U postgres -d conjourney -c "\d closet_env_sample"
```

pan-1 (by hand, after sql-writer is up), from the clean primary main checkout:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && python3 scripts/sync_local_env_from_example.py --all-keys orion-office-env
sed -i 's|^CLOSET_SERIAL_DEVICE_HOST=.*|CLOSET_SERIAL_DEVICE_HOST=/dev/serial/by-id/usb-Arduino_Nano_ESP32_E8F60ABE42B4-if01|' services/orion-office-env/.env
scripts/safe_docker_build.sh orion-office-env up -d --build orion-office-env
scripts/safe_docker_build.sh orion-office-env up -d --build orion-closet-env
```

Firmware is already flashed on both boards.

## Risks / concerns

- Severity: medium. Concern: closet CO2 read 215 ppm at first start and was 414 ppm after about an hour. That is plausible now but unproven. Mitigation: re-check after a day; if it sits under ~420 ppm, do a forced recalibration outdoors (README "Closet CO2 calibration").
- Severity: low. Concern: the CCS811 restarts its 20-minute warm-up on every Nano reflash/reset. Mitigation: expected; host withholds the readings.
- Severity: low. Concern: an unidentified I2C device at `0x78` shares mux channel 4 with the PMSA003I. Mitigation: no address conflict; ask Juniper what it is.
- Severity: low. Concern: `metric_definitions.lock.json` may conflict if main re-locks first. Mitigation: re-run `check_definition_drift.py --update`.

## PR link

(filled in after `gh pr create`)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

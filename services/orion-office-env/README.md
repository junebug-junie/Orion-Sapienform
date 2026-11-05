# orion-office-env

Reads the two pan-1 sensor Nanos (Arduino Nano ESP32, USB) and publishes them to the bus.
It runs on pan-1 only, as two containers from one image:

| Container | `ENV_ROOM` | Board | Sensors | Publishes |
| :--- | :--- | :--- | :--- | :--- |
| `orion-pan-1-orion-office-env` | `office` | office Nano | SHT40, CCS811, PMSA003I, TSL2591 (windowsill) | `office.env.sample.v1` on `orion:office:env:sample` |
| `orion-pan-1-orion-closet-env` (`SERVICE_NAME=orion-closet-env`) | `closet` | closet Nano | SCD40, server closet above the server cabinet | `closet.env.sample.v1` on `orion:closet:env:sample` |

Each container is pinned to its own `/dev/serial/by-id/` path (`OFFICE_SERIAL_DEVICE_HOST`,
`CLOSET_SERIAL_DEVICE_HOST`). Never point both at a glob: with two Nanos, "first match" can
swap the rooms (same rule as Athena's two cabinet Nanos,
`docs/superpowers/specs/2026-08-29-athena-cabinet-dual-nano-design.md`).

It is separate from `orion-biometrics` on purpose: biometrics is an NVIDIA/Athena host
service, and the cabinet `cabinet_*` names describe the cabinet, not the office.

## What it does

1. A thread reads NDJSON lines from the Nano (`orion.office_env_frame.v1`, ~1 per second).
2. Every `OFFICE_PUBLISH_INTERVAL_SEC` (30 s, same as the cabinet's biometrics tick) it
   publishes the newest frame on its room's channel.
3. A frame older than `OFFICE_STALE_AFTER_SEC` (10 s), no frame at all, or a frame with no
   usable reading from either sensor yields a sample with `state.stale=true` and **no
   measurements**. Absent is never zero and an old value is never repeated. The sensors
   are independent: one failing does not drop the others' readings.
4. CCS811 air quality (`office_eco2_ppm`, `office_tvoc_ppb`): for the first 20 minutes after
   each sensor start (datasheet conditioning period) the readings are withheld and
   `state.air_warming=true`. Values outside the datasheet ranges (eCO2 400..29206 ppm, TVOC
   0..32768 ppb) are dropped, not clamped. eCO2 is estimated from VOCs, not measured CO2.
5. PMSA003I particulates (`office_pm1_ug_m3`, `office_pm25_ug_m3`, `office_pm10_ug_m3`,
   CF=1, ug/m3): values outside the datasheet range 0..1000 are dropped. The firmware holds
   them back for the first 30 s after each sensor start.
6. TSL2591 light (`office_lux`): when it overflows at its lowest gain, `office_lux` is absent
   and `state.light_saturated=true` (direct sun: too bright to measure, not dark).
7. Closet SCD40 (`closet_co2_ppm`, `closet_temp_c`, `closet_humidity_pct`): real CO2, not an
   estimate. The closet sample uses only the frame's `co2` object.
8. It always publishes a `SystemHealthV1` heartbeat on `orion:system:health`
   (details: `room`, `sensor`, `serial_open`, `frames_ok`, `lines_rejected`).

```text
office Nano (USB on pan-1) → orion-office-env (ENV_ROOM=office) → orion:office:env:sample
  → orion-sql-writer → Postgres office_env_sample → Hub Biometrics → Cabinet / Office
closet Nano (USB on pan-1) → orion-closet-env (ENV_ROOM=closet) → orion:closet:env:sample
  → orion-sql-writer → Postgres closet_env_sample → Hub Biometrics → Cabinet / Office
```

## What it does not do

- Pressures, baselines, or any substrate / cognition wiring. Raw samples only. A pressure
  needs the metric quality gate (CLAUDE.md 0A) first.
- Any pressure, alert, or cognition reading of eCO2/TVOC, particulates, light, or closet CO2.
  The CCS811 has no baseline save/restore and a 48-hour first-use burn-in (started
  2026-11-03). The closet SCD40 read 215 -> 376 ppm in its first 8 minutes (2026-11-04),
  below outdoor air (~420 ppm), so it is not trusted yet: see "Closet CO2 calibration".
  Every one of these needs the metric quality gate on real data first.
- The sound meter (planned): it will get its own optional frame key and sample fields.
- Flash the Nano. See `firmware/pan-1-office-nano/README.md`.

## Contracts

| Channel | Kind | Schema | Consumer |
| :--- | :--- | :--- | :--- |
| `orion:office:env:sample` | `office.env.sample.v1` | `OfficeEnvSampleV1` | `orion-sql-writer` |
| `orion:closet:env:sample` | `closet.env.sample.v1` | `ClosetEnvSampleV1` | `orion-sql-writer` |

Serial frame (host-local, not a bus payload): `OfficeEnvFrameV1` in
`orion/schemas/telemetry/office_env.py` (both boards). Closet sample:
`orion/schemas/telemetry/closet_env.py`.

## Env

Provenance: `.env_example` → `docker-compose.yml` → `app/settings.py`.

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `ORION_BUS_URL` | `redis://100.92.216.81:6379/0` | Always the Tailscale bus address. |
| `OFFICE_SERIAL_DEVICE_HOST` | (set it) | Host `/dev/serial/by-id/...` path of the Nano. Compose-only. |
| `OFFICE_SERIAL_GID` | `20` | Group that owns the device on the host (`stat -c %g`). Compose-only. |
| `OFFICE_SERIAL_BAUD` | `115200` | Must match the firmware. |
| `OFFICE_PUBLISH_INTERVAL_SEC` | `30` | Bus cadence. |
| `OFFICE_STALE_AFTER_SEC` | `10` | Frame age after which a sample is stale. |
| `OFFICE_ENV_SAMPLE_CHANNEL` | `orion:office:env:sample` | Publish channel. |
| `OFFICE_SENSOR_ID` | `sht40-office` | Sensor id carried in the office sample. |
| `CLOSET_SERIAL_DEVICE_HOST` | (set it) | Host by-id path of the closet Nano. Must differ from the office one. Compose-only. |
| `CLOSET_ENV_SAMPLE_CHANNEL` | `orion:closet:env:sample` | Closet publish channel. |
| `CLOSET_SENSOR_ID` | `scd40-closet` | Sensor id carried in the closet sample. |
| `ENV_ROOM` | `office` | Set per container by compose (`office` / `closet`); not an `.env` key. |

## Deploy (pan-1, by hand)

This service is excluded from the post-merge rebuild and the bring-up sweeps
(`mesh-utilities/common/exclude_services.txt`), and is not on circe's allowlist. pan-1 has
no rebuild hook. Deploy from the clean primary `main` checkout:

```bash
cd /mnt/scripts/Orion-Sapienform
cp services/orion-office-env/.env_example services/orion-office-env/.env   # first time; then set OFFICE_SERIAL_DEVICE_HOST
ls /dev/serial/by-id/
scripts/safe_docker_build.sh orion-office-env up -d --build
```

**Deploy order for a schema change (e.g. the CCS811 fields, 2026-11-03):** the bus schema
rejects unknown fields. Merge first, let `orion-sql-writer` and `orion-hub` rebuild from main,
and check that sql-writer's boot DDL added the `office_env_sample` columns. Only then rebuild
this service on pan-1. If you rebuild pan-1 first, every sample fails validation in the old
sql-writer and **no office rows are stored, temperature included**. Flashing new firmware
early is safe: the old service ignores frame keys it does not know.

Order for a first bring-up: flash the Nano → serial smoke → start the service → check the bus.

```bash
# 1. serial smoke. Exit 0 = every expected sensor seen in 10 s; 1 = silence/junk; 2 = port;
#    office: 3 no SHT40, 4 no CCS811, 5 no PMSA003I, 6 no TSL2591; closet: 7 no SCD40.
#    Run it >= 40 s after a Nano reset (the PMSA003I is held back for 30 s).
#    Stop the running container first: only one process can hold the serial port.
docker compose --env-file .env --env-file services/orion-office-env/.env \
  -f services/orion-office-env/docker-compose.yml run --rm orion-office-env python -m app.smoke
docker compose --env-file .env --env-file services/orion-office-env/.env \
  -f services/orion-office-env/docker-compose.yml run --rm orion-closet-env python -m app.smoke
# 2. heartbeat + publish log lines
docker logs --tail 20 orion-pan-1-orion-office-env
docker logs --tail 20 orion-pan-1-orion-closet-env
```

**Deploy order for the PMSA003I / TSL2591 / closet patch (2026-11-04):** same rule. Merge,
let `orion-sql-writer` rebuild on athena, and check that its boot DDL added
`office_pm1_ug_m3`, `office_pm25_ug_m3`, `office_pm10_ug_m3`, `office_lux`, `light_saturated`
and that `create_all` made `closet_env_sample`. Then on pan-1: pull main, set
`CLOSET_SERIAL_DEVICE_HOST` in `services/orion-office-env/.env`, flash the office Nano, and
`scripts/safe_docker_build.sh orion-office-env up -d --build` (starts both containers).

## Closet CO2 calibration

The SCD40 read 215 ppm at start and 376 ppm after 8 minutes (2026-11-04), which is below
outdoor air. Check again after it has run for a few hours. If it still reads under ~420 ppm,
it needs a forced recalibration: take the closet Nano outside (or to an open window) for
5 minutes, and send the SCD4x `performForcedRecalibration(420)` command. That is not built
here; automatic self-calibration (on by default) only corrects over about a week, and only if
the sensor sees fresh air during that week.

## Tests

```bash
cd services/orion-office-env && PYTHONPATH=../..:. pytest tests evals -q
```

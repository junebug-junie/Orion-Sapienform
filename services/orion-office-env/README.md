# orion-office-env

Reads the **office** environment node (Arduino Nano ESP32 + SHT40 temperature/humidity +
CCS811 air quality, plugged into
**pan-1** by USB) and publishes it to the bus as `office.env.sample.v1`. It runs on pan-1 only.

It is separate from `orion-biometrics` on purpose: biometrics is an NVIDIA/Athena host
service, and the cabinet `cabinet_*` names describe the cabinet, not the office.

## What it does

1. A thread reads NDJSON lines from the Nano (`orion.office_env_frame.v1`, ~1 per second).
2. Every `OFFICE_PUBLISH_INTERVAL_SEC` (30 s, same as the cabinet's biometrics tick) it
   publishes the newest frame on `orion:office:env:sample`.
3. A frame older than `OFFICE_STALE_AFTER_SEC` (10 s), no frame at all, or a frame with no
   usable reading from either sensor yields a sample with `state.stale=true` and **no
   measurements**. Absent is never zero and an old value is never repeated. The two sensors
   are independent: one failing does not drop the other's readings.
4. CCS811 air quality (`office_eco2_ppm`, `office_tvoc_ppb`): for the first 20 minutes after
   each sensor start (datasheet conditioning period) the readings are withheld and
   `state.air_warming=true`. Values outside the datasheet ranges (eCO2 400..29206 ppm, TVOC
   0..32768 ppb) are dropped, not clamped. eCO2 is estimated from VOCs, not measured CO2.
5. It always publishes a `SystemHealthV1` heartbeat on `orion:system:health`
   (details: `office_sensor`, `serial_open`, `frames_ok`, `lines_rejected`).

```text
Nano ESP32 + SHT40 (USB on pan-1)
  → orion-office-env            (this service; pyserial, publishes every 30 s)
  → orion:office:env:sample
  → orion-sql-writer            → Postgres office_env_sample
  → Hub Biometrics → Cabinet / Office → Office strip
```

## What it does not do

- Pressures, baselines, or any substrate / cognition wiring. Raw samples only. A pressure
  needs the metric quality gate (CLAUDE.md 0A) first.
- Any pressure, alert, or cognition reading of eCO2/TVOC. The CCS811 has no baseline
  save/restore and a 48-hour first-use burn-in (started 2026-11-03); its values need the
  metric quality gate on real data first.
- The sound meter (planned): it will get its own optional frame key and sample fields.
- Flash the Nano. See `firmware/pan-1-office-nano/README.md`.

## Contracts

| Channel | Kind | Schema | Consumer |
| :--- | :--- | :--- | :--- |
| `orion:office:env:sample` | `office.env.sample.v1` | `OfficeEnvSampleV1` | `orion-sql-writer` |

Serial frame (host-local, not a bus payload): `OfficeEnvFrameV1` in
`orion/schemas/telemetry/office_env.py`.

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
| `OFFICE_SENSOR_ID` | `sht40-office` | Sensor id carried in the sample. |

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
# 1. serial smoke (exit 0 = real SHT40 + CCS811 frames; 1 = silence/junk; 2 = port;
#    3 = frames but no SHT40 reading; 4 = SHT40 fine but no CCS811 reading)
docker compose --env-file .env --env-file services/orion-office-env/.env \
  -f services/orion-office-env/docker-compose.yml run --rm orion-office-env python -m app.smoke
# 2. heartbeat + publish log lines
docker logs --tail 20 orion-pan-1-orion-office-env
```

## Tests

```bash
cd services/orion-office-env && PYTHONPATH=../..:. pytest tests evals -q
```

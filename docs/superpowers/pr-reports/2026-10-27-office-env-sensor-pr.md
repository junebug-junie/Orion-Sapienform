# feat: office temperature/humidity sensor (pan-1 Nano ESP32 + SHT40)

## Summary

- New service `orion-office-env` (runs on pan-1): reads the Nano's USB-serial NDJSON and publishes `office.env.sample.v1` on `orion:office:env:sample` every 30 s.
- New contract: `OfficeEnvSampleV1` (bus) and `OfficeEnvFrameV1` (host-local serial frame), registered, with a channel entry.
- sql-writer stores samples in new table `office_env_sample`.
- Hub: Biometrics subtab renamed "Cabinet / Office"; new Office strip (live tiles + temp/humidity charts, 24h/3d/7d).
- Firmware sketch `firmware/pan-1-office-nano/` (SHT40, heater off).
- `orion-office-env` added to `mesh-utilities/common/exclude_services.txt`. Circe needs no entry (positive allowlist). pan-1 has no rebuild hook; deploy by hand.

## Outcome moved

Orion gets a second physical sensing site (the office, outside the cabinet). Raw readings only. No pressures, no substrate or cognition wiring.

## Current architecture

Physical sensing was Athena-only: Nano -> host reader -> `/run/orion-sensors/latest.json` -> `orion-biometrics` (NVIDIA image) -> `orion_biometrics_summary`. A pan-1 sensor cannot use that path (file bind-mount is host-local; biometrics needs `nvidia-smi`).

## Architecture touched

`orion-office-env` (new), `orion-sql-writer`, `orion-hub`, `orion/schemas`, `orion/bus/channels.yaml`, `config/metrics/metric_definitions.lock.json`, `mesh-utilities/common/exclude_services.txt`, `firmware/`.

## Files changed

- `orion/schemas/telemetry/office_env.py`, `orion/schemas/registry.py`, `orion/bus/channels.yaml`: contract.
- `services/orion-office-env/**`: service, tests, replay eval, README.
- `services/orion-sql-writer/app/{models/office_env_sample.py,models/__init__.py,settings.py,worker.py}`, `.env_example`, README, test.
- `services/orion-hub/{scripts/office_env_routes.py,scripts/api_routes.py,app/settings.py,static/js/office-env.js,templates/index.html,docker-compose.yml,.env_example,README.md}` + tests.
- `config/metrics/metric_definitions.lock.json`: re-locked (one added channel, MEDIUM).
- `mesh-utilities/common/exclude_services.txt`: exclude entry.
- `firmware/pan-1-office-nano/**`.

## Schema / bus / API changes

- Added: channel `orion:office:env:sample` (`OfficeEnvSampleV1`, `office.env.sample.v1`); Hub `GET /api/office/env/latest`, `GET /api/office/env/history?window=24h|3d|7d`; table `office_env_sample` (via `create_all`, no migration file).
- Removed / Renamed: none (UI label only: "Cabinet" -> "Cabinet / Office"; element id unchanged).
- Compatibility: purely additive.

## Env/config changes

- Added keys: new `services/orion-office-env/.env_example` (14 keys); Hub `OFFICE_ENV_NODE`, `OFFICE_ENV_STALE_AFTER_SEC`, `OFFICE_ENV_HISTORY_MAX_POINTS`; sql-writer channel list gains `orion:office:env:sample` (code also force-appends it).
- `.env_example` updated: yes (office-env, hub, sql-writer).
- local `.env` synced: **NOT DONE.** pan-1 has no `.env` for hub/sql-writer (they run on Athena); `sync_local_env_from_example.py` reports "no .env found". Athena's hub `.env` gets the three new keys on its next sync (all have code defaults). `services/orion-office-env/.env` must be created on pan-1 at deploy (see README).
- skipped keys requiring operator action: `OFFICE_SERIAL_DEVICE_HOST` (no safe default).

## Metric quality gate (new channels `office_temp_c`, `office_humidity_pct`)

1. Provenance: SHT40 (Sensirion) via `Adafruit_SHT4x::getEvent` in `firmware/pan-1-office-nano/pan-1-office-nano.ino` -> frame -> `build_sample` (`services/orion-office-env/app/sample.py`).
2. Independence: both come from one chip; RH is temperature-dependent physically. Not independent of each other. Neither is independent of anything in a model because **neither is wired into any model**.
3. Theory anchor: SHT40 datasheet accuracy (+-0.2 C, +-1.8 %RH typical). Stored and charted only; no detector built.
4. Live-data sanity: **UNVERIFIED.** The Nano is not attached; no real data has been seen. Rest/saturation/degeneracy check must be done on first real data before anything consumes these.
5. Existing mechanism: cabinet BME680 temp/humidity exists on Athena (`cabinet_temp_c`); different location, so not a duplicate.
6. Reversibility: cheap. One table, one channel, one service; nothing downstream consumes it.

## Tests run

```text
orion-office-env: pytest tests evals            -> 22 passed (after review fixes)
orion-sql-writer: test_office_env_sample_sql_shape + test_home_cooling_sample_sql_shape -> 13 passed
orion-hub: office routes/panel + cooling routes + cabinet sensors panel -> 70 passed
hub JS: node --test office-env.test.js          -> 6 passed
tests/test_office_env_bus_catalog.py            -> passed
scripts/check_definition_drift.py --gate        -> PASS (after --update)
scripts/check_env_template_parity.py            -> PASS
```

Tests ran in `python:3.12` containers because pan-1 has no pydantic/pip/venv.

## Evals run

`services/orion-office-env/evals/test_outage_replay_eval.py`: replays noise + missing-sensor frame + 47 s silence + recovery; asserts stale samples carry no readings and no old value is repeated.

## Docker/build/smoke checks

```text
docker build -f services/orion-office-env/Dockerfile .   -> built
scripts/safe_docker_build.sh orion-office-env config     -> FAILED: pan-1 has no `docker compose` plugin
Live path (serial -> bus -> Postgres -> Hub)             -> UNVERIFIED (no Nano attached)
```

## Review findings fixed

- Finding: Hub showed an old reading as a live number when the publisher was dead.
  - Fix: `/latest` drops readings when `publisher_stale`; JS only prints numbers when `ok`.
  - Evidence: `test_latest_publisher_dead_hides_the_old_reading`.
- Finding: chart line joined across outages on wide (7d) buckets.
  - Fix: downsample per contiguous segment, flag `gap: true`; JS breaks runs on it.
  - Evidence: `test_downsample_never_averages_across_an_outage`, JS gap test.
- Finding: reader thread could die silently.
  - Fix: import error logged; publish loop restarts a dead reader and reports `reader_alive`.
- Finding: replayed identical frames refreshed freshness.
  - Fix: `FrameStore.put` ignores a frame with the same `seq` + `uptime_ms`.
  - Evidence: `test_repeated_identical_frame_does_not_refresh_the_clock`.
- Finding: DB error on `/latest` looked like "no samples"; office chart shared the cabinet's max-points setting.
  - Fix: `error` field + JS keeps last values; own `OFFICE_ENV_HISTORY_MAX_POINTS`.

## Restart required

Athena (hub + sql-writer pick up the new route/channel/table; sql-writer must restart to subscribe and create the table):

```bash
cd /mnt/scripts/Orion-Sapienform   # on Athena, after merge + pull
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

pan-1 (after installing the compose plugin, flashing the Nano, and creating `.env`): see `services/orion-office-env/README.md`.

## Risks / concerns

- Severity: high (deploy). Concern: pan-1 has no `docker compose`; `safe_docker_build.sh` cannot run there. Mitigation: `sudo apt install docker-compose-plugin` (operator step).
- Severity: medium. Concern: nothing in this patch has seen real hardware. Mitigation: serial smoke (`python -m app.smoke`) is the first step after flashing.
- Severity: low. Concern: no retention on `office_env_sample` (~2,880 rows/day).

## PR link

(pending)

## Update 2026-10-28: bench test on pan-1

- Nano flashed from a container (`arduino-cli`, `arduino:esp32:nano_nora`). ArduinoJson pinned to 6.21.5.
- First scan found a device at `0x70` and nothing at `0x44`. It is a TCA9548A multiplexer. The SHT40 is on channel 3.
- Firmware now selects the multiplexer channel (`MUX_CHANNEL`) before each read.
- Serial smoke: `PASS temp_c=24.455 humidity_pct=33.13`. 11 frames, 0 rejected lines.
- Bus publish, SQL write and Hub strip are still UNVERIFIED. The metric gate live-data check is still open until stored data exists.

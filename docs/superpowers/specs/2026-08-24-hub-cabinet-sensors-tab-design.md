# Hub Cabinet Sensors Tab — Design

**Date:** 2026-08-24  
**Status:** Approved for planning  
**Branch intent:** `docs/cabinet-hub-tab` → `feat/hub-cabinet-sensors-tab`

## Arsonist summary

Add a top-level Hub **Cabinet** tab that polls the Athena Nano sensor snapshot while visible and shows raw channels plus derived `cabinet_*` pressures. Reuse the existing host snapshot and telemetry helpers; do not invent a second cognition path.

## Decisions locked

| Decision | Choice |
|---|---|
| Surface | Top-level Hub nav tab `#cabinet` |
| Content | Raw Nano frame **and** derived pressures |
| Refresh | Poll ~1s **only while tab visible** |
| Data path | Hub reads `/run/orion-sensors/*.json` directly (bind-mount `:ro`) |

## Current architecture

```text
Nano ESP32 NDJSON (~1 Hz)
  → scripts/orion_cabinet_sensor_reader.py (systemd)
  → /run/orion-sensors/latest.json (+ boot.json)
  → orion-biometrics (bind-mount :ro)
       → BiometricsSampleV1.sensors
       → cabinet_* measurements / pressures
       → grammar / field-digester
```

Hub today:

- Small **Biometrics** strip on the Hub tab (cluster strain / homeostasis / stability via WebSocket cache).
- No Cabinet tab.
- No Hub mount of `/run/orion-sensors`.
- No Hub API for cabinet snapshots.

Existing contracts to reuse:

- Frame schema: `orion/schemas/telemetry/cabinet_sensor_frame.py` (`orion.sensor_frame.v1`)
- Measurement/pressure helpers: `orion/telemetry/cabinet_sensors.py`
- Host snapshot loader pattern: `services/orion-biometrics/app/cabinet_snapshot.py`
- Operator diagnose script: `scripts/diagnose_athena_cabinet_sensors.sh`
- Prior node design: `docs/superpowers/specs/2026-08-23-athena-cabinet-sensor-node-design.md`

## Missing questions (resolved)

1. Top-level tab vs biometrics strip expand → **top-level tab**
2. Raw vs pressures vs both → **both**
3. Transport → **visible-tab poll ~1s**
4. Owner of snapshot read → **Hub direct read** (not biometrics proxy, not WS-only)

## Proposed schema / API changes

### HTTP API (Hub)

`GET /api/cabinet/sensors/latest`

Response shape (v1):

```json
{
  "ok": true,
  "age_sec": 0.8,
  "snapshot": {
    "status": "ok",
    "device": "/dev/serial/by-id/usb-Arduino_Nano_ESP32_...",
    "received_at": "2026-08-24T...",
    "frame": { "schema": "orion.sensor_frame.v1", "seq": 12, "...": "..." }
  },
  "boot": {
    "schema": "orion.sensor_boot.v1",
    "i2c": { "primary": "A4/A5", "addresses": ["0x30", "0x53"] },
    "sensors": { "mmc5603": { "ok": true, "addr": "0x30" }, "...": "..." }
  },
  "measurements": {
    "cabinet_magnetic_ut": 78.01,
    "cabinet_als_raw": 67.0
  },
  "pressures": {
    "cabinet_em_activity": 0.12,
    "cabinet_uv_activity": 0.04,
    "cabinet_sensor_staleness": 0.0
  }
}
```

Rules:

- Missing / unreadable `latest.json` → `ok=false`, `snapshot=null`, empty `measurements`/`pressures` (not zero-filled).
- Missing `boot.json` → `boot=null` (tab still usable from `latest.json` alone).
- Stale snapshot (reader status or age) → `ok=false` with snapshot still returned so UI can show last-seen values + stale badge.
- Absent frame sub-objects stay absent; UI labels them `absent`.
- Pressures computed with the same helpers biometrics uses (`extract_cabinet_measurements` + `compute_cabinet_pressures`). Baseline tracker may be process-local for Hub; document that Hub pressures are **operator-debug approximations** if baseline state is not shared with biometrics. Prefer reading already-persisted measurement/pressure keys only when a shared source exists; otherwise instantiate a thin Hub-side tracker and label the strip “activity (Hub)”.

No bus-schema registry changes. This is a Hub debug/operator API, not a new Redis stream payload.

### Config / Docker

| Key | Default | Notes |
|---|---|---|
| `CABINET_SENSORS_PATH` | `/run/orion-sensors/latest.json` | Same default as biometrics |
| `CABINET_BOOT_PATH` | `/run/orion-sensors/boot.json` | Optional boot diagnostic |

Update in the same changeset:

- `services/orion-hub/.env_example`
- `services/orion-hub/settings.py`
- `services/orion-hub/docker-compose.yml` bind: `/run/orion-sensors:/run/orion-sensors:ro`
- local `.env` via `python scripts/sync_local_env_from_example.py`

## UI design

### Nav

- New primary nav link: **Cabinet** → `#cabinet`
- Same tab styling/hash routing as Organ Signals / Self

### Panel layout

1. **Status strip**
   - Reader status (`ok` / `stale` / `missing`)
   - Device path
   - Frame `seq`, `uptime_ms`
   - Age from `received_at`
   - I2C addresses from boot when present

2. **Sensor grid (primary)**
   - Environment (BME680)
   - UV / ALS (LTR390)
   - Magnetic (MMC5603)
   - Particulate (PMSA003I)
   - Lidar (VL53L1X)
   - IMU (BNO085)
   - Each tile: live values **or** explicit `absent`

3. **Pressure strip (secondary)**
   - `cabinet_climate_activity`
   - `cabinet_particulate_activity`
   - `cabinet_em_activity`
   - `cabinet_uv_activity`
   - `cabinet_vibration_activity`
   - `cabinet_proximity_activity`
   - `cabinet_sensor_staleness`

4. **Polling behavior**
   - Start poller when `#cabinet` becomes visible
   - Interval ~1000 ms
   - Stop/cleanup on tab hide (match Hub poll-only-while-visible rule)
   - Transient fetch failures must not blank previously good content (stale badge instead)

Preserve existing Hub visual language (dark panels, compact labels). No new chart library for v1.

## Files likely to touch

```text
docs/superpowers/specs/2026-08-24-hub-cabinet-sensors-tab-design.md
services/orion-hub/templates/index.html
services/orion-hub/static/js/app.js                    # tab routing / panel show
services/orion-hub/static/js/cabinet-sensors.js        # new panel module
services/orion-hub/scripts/cabinet_sensors_routes.py   # new API
services/orion-hub/scripts/api_routes.py               # include router
services/orion-hub/app/settings.py
services/orion-hub/.env_example
services/orion-hub/docker-compose.yml
services/orion-hub/tests/test_cabinet_sensors_api.py
services/orion-hub/tests/test_hub_ui_polish.py         # nav/tab presence
services/orion-hub/README.md                           # short operator note
```

Optional reuse: factor shared snapshot-load logic next to biometrics’ `cabinet_snapshot.py` only if duplication becomes painful; first patch may copy the thin load path into Hub to stay service-bounded.

## Non-goals

- WebSocket push for cabinet frames
- Historical charts / retention
- Changes to Nano firmware
- Changes to `orion-biometrics` cognition pipeline
- Replacing `scripts/diagnose_athena_cabinet_sensors.sh`
- Zero-filling missing sensors
- Multi-node cabinet (Athena-only for v1)

## Error handling

| Condition | API | UI |
|---|---|---|
| Snapshot file missing | `ok=false`, `snapshot=null` | “No snapshot — is `orion-cabinet-sensors.service` running?” |
| Snapshot unreadable JSON | `ok=false` | Same + last error text |
| Snapshot stale | `ok=false`, snapshot returned | Stale badge; show last values |
| Boot missing | `boot=null` | Hide I2C boot section |
| Sensor key absent in frame | omitted in measurements | Tile shows `absent` |
| Poll error while previously live | HTTP error | Keep last good render + “poll error” badge |

## Acceptance checks

1. Cabinet tab visible in Hub primary nav and opens `#cabinet`.
2. With a live `/run/orion-sensors/latest.json`, magnetic/UV (etc.) tiles show present values within ~2s of tab open.
3. Sensors omitted from the Nano frame render as `absent`, never as `0`.
4. Pressure strip shows keys only when derivable; otherwise absent/empty strip state.
5. Leaving the tab stops the poller (no background interval).
6. With snapshot removed/missing, UI shows explicit no-snapshot state.
7. Focused tests: API missing/stale/fresh fixtures; Hub UI polish asserts tab + asset wiring.
8. Env/compose mount documented; restart Hub required after mount/env change.

## Recommended next patch

Thin implementation slice:

1. Hub settings + compose bind-mount + env keys
2. `GET /api/cabinet/sensors/latest` + unit tests
3. `#cabinet` panel + `cabinet-sensors.js` poller
4. README note + Hub restart instructions

## Risks / concerns

- **Severity: medium — Hub pressure baselines diverge from biometrics.** Mitigation: label Hub activity strip as operator debug; do not claim it is the live field value unless shared baseline state is wired later.
- **Severity: low — Hub without mount shows perpetual no-snapshot.** Mitigation: compose + README + API error message name the missing path/service.
- **Severity: low — boot.json optional.** Mitigation: tab must work from `latest.json` alone.

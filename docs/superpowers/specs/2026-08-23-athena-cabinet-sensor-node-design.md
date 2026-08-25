# Athena cabinet sensory node (Nano ESP32) — design

Date: 2026-08-23  
Status: approved for implementation planning  
Worktree: `/mnt/scripts/Orion-Sapienform-cabinet-sensor-node` on `feat/cabinet-sensor-node`

## Arsonist summary

Give Athena a physical cabinet sense: a Nano ESP32 on USB publishes raw sensor NDJSON; a tiny host reader snapshots it; `orion-biometrics` folds it into the existing biometrics → grammar → substrate → field-digester path as **new** measurements, pressures, and field channels on node `athena`. No Arduino→field shortcut. No overload of host CPU/GPU/thermal/fan pressures. No invented absolute “badness” thresholds in v1. USB mic on Athena is out of scope for this patch — see `docs/superpowers/specs/2026-08-24-athena-ambient-audio-levels-design.md` for the separate host-side ambient audio levels path (not Nano firmware).

## Decisions locked

| Topic | Choice |
|---|---|
| Architecture | Host systemd serial reader → `/run/orion-sensors/latest.json` → biometrics (ro bind) → grammar → substrate → field-digester |
| Field node | `athena` |
| Pressures (v1) | Baseline-relative only (EWMA / delta / volatility / anomaly ∈ [0,1]) |
| Absolute comfort/AQI bands | Out of scope for v1 |
| Existing WIP | Continue; drop absolute climate/PM pressure normalizers from WIP |
| Firmware | Write + flash in-repo (Nano never installed on Athena before) |
| Arduino audio (MAX9814) | Removed — not on this board |
| USB mic on Athena | Out of scope — see `2026-08-24-athena-ambient-audio-levels-design.md` (host reader, not Nano) |
| Perturbation mode | `mode="replace"` for all current-reading cabinet hints |

## Current architecture (grounded)

Existing path (do not bypass):

```text
orion-biometrics sample/summary
  → app/grammar_emit.py (GrammarEventV1 atoms)
  → orion/substrate/biometrics_loop (grammar_extract → pressure_hints → StateDeltaV1)
  → orion-field-digester app/ingest/state_deltas.py (Perturbation, mode="replace" for biometrics pressures)
  → NODE_CHANNELS on the node
```

Relevant surfaces today:

- `orion/schemas/telemetry/biometrics.py` — `BiometricsSampleV1`, `BiometricsSummaryV1.measurements` (absent ≠ zero)
- `orion/telemetry/biometrics_pipeline.py` — host pressures + measurements
- `services/orion-biometrics/app/grammar_emit.py` — host pressure signal atoms
- `orion/substrate/biometrics_loop/grammar_extract.py` — `pressure_hints`
- `services/orion-field-digester/app/ingest/state_deltas.py` — replace-mode biometrics fan-out
- `services/orion-field-digester/app/tensor/channels.py` — `NODE_CHANNELS`
- `config/field/field_channel_glossary.v1.yaml`
- `scripts/smoke_field_digester_biometrics.sh` — thin Hub API smoke (extend, do not replace)

Hardware observed on Athena: `/dev/serial/by-id/usb-Arduino_Nano_ESP32_E8F60ABE5E98-if01` → `ttyACM0`. Port currently root-owned `660`; udev/group fix required. No `/run/orion-sensors` yet. No cabinet firmware on main.

WIP already present on this branch (partial): frame schema, `cabinet_sensors` helpers, `BiometricsSampleV1.sensors`, pipeline merge that keeps host `peak_pressure` / `constraint` host-only. Missing: firmware, systemd reader, grammar atoms, field channels, automation smokes. WIP absolute thermal/humidity/PM pressure scales are **rejected for v1** and must be removed or left unused.

## End-to-end data flow

```text
Nano ESP32 firmware (~1 Hz NDJSON)
  → host systemd reader (validate, received_at, atomic latest.json, stale/error)
  → /run/orion-sensors/latest.json
  → orion-biometrics (ro bind): sample.sensors
  → summary.measurements (native units, absent ≠ zero)
  → summary.pressures (cabinet_* only; baseline-relative 0..1)
  → grammar atoms (new roles; existing host atoms unchanged)
  → biometrics substrate StateDeltaV1.pressure_hints
  → field-digester state_deltas.py Perturbation(mode="replace")
  → new NODE_CHANNELS on node athena
```

Hard rules:

1. Physical sensor telemetry belongs to `orion-biometrics`. No parallel Arduino → field-digester path.
2. Malformed/partial serial lines never overwrite the last good snapshot.
3. Stale/missing Nano → absent measurements / stale status — never zero-filled.
4. Existing host pressures and field channels unchanged in meaning and producers.
5. Host `peak_pressure` / `constraint` remain host-only; cabinet pressures do not enter that max.

## Firmware contract

Location: `firmware/athena-cabinet-nano/` (Arduino sketch for Arduino Nano ESP32 / ABX00083).

Emit one versioned NDJSON object per line at ~1 Hz. On sensor init/read failure, **omit the entire sub-object** — never send zeros for a dead sensor.

```json
{
  "schema": "orion.sensor_frame.v1",
  "seq": 4812,
  "uptime_ms": 992813,
  "environment": {
    "temp_c": 24.6,
    "humidity_pct": 31.8,
    "pressure_hpa": 857.2,
    "gas_resistance_ohm": 138421
  },
  "uv": {"raw": 17, "als_raw": 1292},
  "magnetic": {
    "x_ut": 31.2,
    "y_ut": -8.4,
    "z_ut": 42.1,
    "magnitude_ut": 53.0
  },
  "particulate": {
    "pm1_ug_m3": 2,
    "pm25_ug_m3": 4,
    "pm10_ug_m3": 5
  },
  "lidar": {"distance_mm": 438, "status": 0},
  "imu": {
    "accel_x": 0.01,
    "accel_y": -0.02,
    "accel_z": 9.79,
    "yaw_deg": 12.4,
    "pitch_deg": 0.8,
    "roll_deg": -1.1
  }
}
```

| Sensor | Bus / notes |
|---|---|
| BME680 | I2C — environment |
| LTR390 | I2C — UV + ALS raw counts |
| LIS3MDL | I2C — µT + magnitude |
| PMSA003I | I2C — µg/m³ |
| VL53L1X | I2C — mm; consumers trust distance only when `status == 0` |
| BNO085 | **UART-RVC only** — not on ESP32 I2C |
| MAX9814 | **Not used** — removed from this design |

Audio is not part of the Nano frame. A USB microphone plugged into Athena is a separate host-side concern — implemented in `docs/superpowers/specs/2026-08-24-athena-ambient-audio-levels-design.md` (host ALSA reader → biometrics `ambient_audio`, not Nano `sensors`).

Host-local transport validation: `orion/schemas/telemetry/cabinet_sensor_frame.py` (not a bus-registry payload). Align WIP field names to this document (`schema: orion.sensor_frame.v1`, no `audio` block, lidar trust rule above).

## Host serial reader

Tiny systemd service on Athena (no cognition inside Docker):

Responsibilities:

- Open stable path `/dev/serial/by-id/usb-Arduino_Nano_ESP32_*` (never hard-code `/dev/ttyACM0`)
- Auto-reconnect on unplug/replug/reboot of the Nano
- Read line-oriented NDJSON; reject malformed/partial frames
- Stamp host `received_at` (UTC ISO8601)
- Atomically write `/run/orion-sensors/latest.json` (temp file + rename)
- Expose status: `ok` | `stale` | `error` | `missing`, plus optional `error` string
- Preserve last good frame across bad lines
- **No** normalization, EWMA, pressures, or cognition

On-disk shape written by the reader:

```json
{
  "status": "ok",
  "received_at": "2026-08-23T06:00:00.123Z",
  "device": "/dev/serial/by-id/usb-Arduino_Nano_ESP32_E8F60ABE5E98-if01",
  "frame": { "...": "CabinetSensorFrameV1" }
}
```

Supporting pieces:

- `RuntimeDirectory=orion-sensors` (or tmpfiles.d) for `/run/orion-sensors`
- udev rule so the service user can open the by-id device after replug
- Unit: restart on failure; reader under `scripts/` or `deploy/systemd/`

Bind `/run/orion-sensors` **read-only** into `orion-biometrics` via compose.

## Biometrics integration

### Sample

Extend `BiometricsSampleV1` with optional `sensors` (WIP shape, refined):

```text
sensors: {
  frame: <CabinetSensorFrameV1 dict>,
  received_at: <host ISO8601>,
  stale: bool
}
```

- `sensors` **absent** (not `{}`) if no valid snapshot has ever been read on this node.
- Biometrics sets `stale=true` when `now - received_at > CABINET_SENSOR_STALE_AFTER_SEC` (env; default ~5–10s at 1 Hz).
- When stale: do not publish cabinet measurement keys; do not emit fresh cabinet activity pressures (staleness is the `cabinet_sensor_staleness` channel/hint). Never zero-fill missing sensors.

### Measurements

Merge into `summary.measurements` with unit-in-name keys, only when fresh + vouchable:

| Key | Source |
|---|---|
| `cabinet_temp_c` | environment.temp_c |
| `cabinet_humidity_pct` | environment.humidity_pct |
| `cabinet_pressure_hpa` | environment.pressure_hpa |
| `cabinet_gas_resistance_ohm` | environment.gas_resistance_ohm |
| `cabinet_uv_raw` | uv.raw |
| `cabinet_als_raw` | uv.als_raw |
| `cabinet_magnetic_ut` | magnetic.magnitude_ut |
| `cabinet_pm1_ug_m3` | particulate.pm1_ug_m3 |
| `cabinet_pm25_ug_m3` | particulate.pm25_ug_m3 |
| `cabinet_pm10_ug_m3` | particulate.pm10_ug_m3 |
| `cabinet_lidar_mm` | lidar.distance_mm iff status==0 |
| `cabinet_vibration_g` | \|‖accel‖/g − 1\| from imu |

### Pressures (v1)

New keys only — never write into existing host pressure keys (`thermal`, `fan`, `power`, `mem`, `cpu`, `gpu`, …).

For climate, particulate, EM, UV, vibration, proximity: derive activity/anomaly in \[0,1\] from EWMA baseline, delta, volatility → anomaly. Raw values stay in measurements.

Pressure keys (also grammar hint keys and field channel names):

- `cabinet_climate_activity`
- `cabinet_particulate_activity`
- `cabinet_em_activity`
- `cabinet_uv_activity`
- `cabinet_vibration_activity`
- `cabinet_proximity_activity`

No absolute comfort or EPA AQI mapping in v1. Tracker state is in-process (cold start after biometrics restart is acceptable and documented). Unit-test the constant-input rest point.

Host `peak_pressure` / `constraint` stay computed from host pressures only (WIP already does this).

## Grammar and substrate

Extend `services/orion-biometrics/app/grammar_emit.py` with new atoms parallel to existing host pressure signals, e.g.:

- `cabinet_climate_activity_signal`
- `cabinet_particulate_activity_signal`
- `cabinet_em_activity_signal`
- `cabinet_uv_activity_signal`
- `cabinet_vibration_activity_signal`
- `cabinet_proximity_activity_signal`
- `cabinet_sensor_staleness_signal`

Wire edges from the telemetry sample atom → each new signal (same pattern as host pressures).

Extend `orion/substrate/biometrics_loop/grammar_extract.py` to copy atom salience into `pressure_hints` under the pressure key names above (plus `cabinet_sensor_staleness`).

Existing host grammar behavior must remain unchanged when `sensors` is absent.

## Field-digester

1. Add dedicated channels to `NODE_CHANNELS` in `services/orion-field-digester/app/tensor/channels.py`.
2. Add glossary entries in `config/field/field_channel_glossary.v1.yaml` (`physical_substrate` for activity channels; `sensor_trust_liveness` for staleness).
3. In `state_deltas.py`, for `target_kind == "node_biometrics"`, map each cabinet hint → `Perturbation(..., mode="replace")` on the matching channel for the delta’s node (`athena`).

Channels:

- `cabinet_climate_activity`
- `cabinet_particulate_activity`
- `cabinet_em_activity`
- `cabinet_uv_activity`
- `cabinet_vibration_activity`
- `cabinet_proximity_activity`
- `cabinet_sensor_staleness`

Do not map cabinet climate into `thermal_pressure` or any existing host channel.

Acceptance smoke must show channels follow the sensor and recover downward under replace mode (no add-saturation from multi-atom traces).

## Automation

Idempotent scripts (names may vary slightly; coverage required):

| Script | Purpose |
|---|---|
| `scripts/setup_athena_cabinet_sensors.sh` | Host deps, udev, runtime dir, systemd install/enable |
| `scripts/flash_athena_cabinet_nano.sh` | Discover by-id device; flash firmware (`arduino-cli` or documented equivalent) |
| `scripts/discover_athena_cabinet_serial.sh` | Print/verify stable by-id path |
| `scripts/smoke_athena_cabinet_serial.sh` | N valid frames; fail on timeout / only-malformed |
| `scripts/smoke_biometrics_cabinet_sensors.sh` | Biometrics sees fresh sensors + measurement keys |
| `scripts/smoke_biometrics_cabinet_grammar.sh` | Grammar / pressure_hints carry cabinet keys |
| Extend `scripts/smoke_field_digester_biometrics.sh` | New channels present, move, recover downward |

## Tests

Unit / focused:

- Frame schema accepts valid NDJSON; rejects malformed
- Reader logic: bad line does not clobber last good; atomic write semantics
- Stale sensors → no cabinet measurement keys
- Lidar `status != 0` → no `cabinet_lidar_mm`
- Pressures bounded \[0,1\]; constant input → calm activity rest point
- `state_deltas` emits `mode="replace"` for cabinet hints
- Host pressure / peak_pressure tests unchanged when cabinet present or absent

E2E: scripted smokes on Athena when hardware is attached.

## Acceptance checklist

1. Nano reconnect/reboot recovers without operator action.
2. `ttyACM*` number change does not matter (by-id only).
3. Malformed serial frame does not poison latest good sample.
4. Stale Nano represented as stale/absent, not zeros.
5. Raw BME/LTR/LIS/PMS/LiDAR/IMU values visible through biometrics.
6. Summary physical measurements retain native units in key names.
7. Normalized derived sensor pressures remain bounded 0..1.
8. Pressure hints survive grammar/substrate path.
9. Field-digester channel values follow the sensor and recover downward; no accumulate/saturate from duplicate grammar events.
10. Existing biometrics behavior and current field channels unchanged.
11. Focused unit tests plus end-to-end smoke coverage.

## Non-goals

- USB microphone capture on Athena (see `2026-08-24-athena-ambient-audio-levels-design.md` — host path, not this Nano patch)
- MAX9814 / Arduino analog audio
- Absolute comfort or AQI thresholds in v1
- Calibrated dBA
- Chemical species ID from BME680 gas resistance
- Direct Arduino → field-digester path
- Overloading host `thermal_pressure` / `fan_pressure` / CPU/GPU pressures
- Separate virtual lattice node for the cabinet
- Persisted EWMA baselines across biometrics restarts (v1 is in-process)

## Files likely to touch

- `firmware/athena-cabinet-nano/` (new)
- `scripts/*athena_cabinet*` / `deploy/systemd/` (new)
- `orion/schemas/telemetry/cabinet_sensor_frame.py` (WIP → align)
- `orion/telemetry/cabinet_sensors.py` (WIP → baseline-relative only)
- `orion/schemas/telemetry/biometrics.py`
- `orion/telemetry/biometrics_pipeline.py`
- `services/orion-biometrics/app/main.py` (snapshot load path)
- `services/orion-biometrics/app/grammar_emit.py`
- `services/orion-biometrics/docker-compose.yml` + `.env_example` / `settings.py`
- `orion/substrate/biometrics_loop/grammar_extract.py`
- `services/orion-field-digester/app/ingest/state_deltas.py`
- `services/orion-field-digester/app/tensor/channels.py`
- `config/field/field_channel_glossary.v1.yaml`
- `scripts/smoke_field_digester_biometrics.sh` + new smokes
- `tests/test_cabinet_sensors.py` (+ reader / state_deltas tests)
- `services/orion-biometrics/README.md` (Cabinet sensor node + USB mic note)

## Env / config

Expected new keys (exact names at implementation; parity required):

- `CABINET_SENSORS_PATH` — default `/run/orion-sensors/latest.json`
- `CABINET_SENSOR_STALE_AFTER_SEC` — freshness gate inside biometrics
- Optional enable flag if needed for nodes without hardware (default: attempt read; absent file ⇒ no `sensors` key)

`.env_example` updated; local `.env` synced via `python scripts/sync_local_env_from_example.py`.

## Risks

- **Decay vs replace cadence** on new channels — same known field-digester class as host biometrics; mitigate with replace mode + smoke that proves downward recovery.
- **Serial permissions** — without udev, reader fails closed; setup script must be idempotent and verified.
- **Cold EWMA** — first minutes after biometrics restart look calm; document, do not fake history.
- **I2C / UART wiring** — firmware must fail soft per sensor; flash/smoke scripts must make wiring failures obvious.

## Recommended next patch

1. Writing-plans: implementation plan from this spec.
2. Implement in this worktree: firmware + reader + biometrics + grammar + field + tests + smokes.
3. Live Athena smoke with the attached Nano.

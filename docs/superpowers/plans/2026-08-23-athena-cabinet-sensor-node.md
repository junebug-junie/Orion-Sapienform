# Athena Cabinet Sensor Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Nano ESP32 cabinet sensing on Athena through host serial snapshot → `orion-biometrics` → grammar → substrate → field-digester as new measurements/pressures/channels on node `athena`.

**Architecture:** Host systemd NDJSON reader writes `/run/orion-sensors/latest.json`; biometrics bind-mounts it into `BiometricsSampleV1.sensors`; baseline-relative `cabinet_*` pressures feed new grammar atoms and `mode="replace"` field channels. No Arduino→field shortcut; no host pressure overload; no absolute comfort/AQI thresholds in v1.

**Tech Stack:** Arduino Nano ESP32 firmware, Python systemd reader, existing Orion biometrics/grammar/field-digester contracts, pytest, bash smokes.

**Spec:** `docs/superpowers/specs/2026-08-23-athena-cabinet-sensor-node-design.md`

**Worktree:** `/mnt/scripts/Orion-Sapienform-cabinet-sensor-node` on `feat/cabinet-sensor-node`

## Global Constraints

- Path: biometrics → grammar → substrate → field-digester only
- Snapshot path: `/run/orion-sensors/latest.json`
- Serial: `/dev/serial/by-id/usb-Arduino_Nano_ESP32_*` only (never hard-code `ttyACM0`)
- Absent ≠ zero for measurements; stale ⇒ omit cabinet measurement keys
- Pressure keys (exact): `cabinet_climate_activity`, `cabinet_particulate_activity`, `cabinet_em_activity`, `cabinet_uv_activity`, `cabinet_vibration_activity`, `cabinet_proximity_activity`
- Staleness hint/channel: `cabinet_sensor_staleness`
- All cabinet field perturbations: `mode="replace"`
- Host `peak_pressure` / `constraint` ignore cabinet keys
- No Arduino audio; USB mic on Athena is document-only
- No absolute thermal/humidity/PM pressure normalizers in v1
- Commit from this worktree only; do not touch unrelated files
- Every new adapter degrades to absent/None — never raise on missing input

## File map

| Area | Files |
|---|---|
| Frame + normalize | `orion/schemas/telemetry/cabinet_sensor_frame.py`, `orion/telemetry/cabinet_sensors.py`, `orion/schemas/telemetry/biometrics.py`, `orion/telemetry/biometrics_pipeline.py`, `tests/test_cabinet_sensors.py` |
| Host reader | `scripts/orion_cabinet_sensor_reader.py`, `deploy/systemd/orion-cabinet-sensors.service`, `deploy/udev/99-orion-cabinet-nano.rules`, `scripts/setup_athena_cabinet_sensors.sh`, `scripts/discover_athena_cabinet_serial.sh`, `scripts/smoke_athena_cabinet_serial.sh`, `tests/test_cabinet_sensor_reader.py` |
| Firmware | `firmware/athena-cabinet-nano/` (+ `scripts/flash_athena_cabinet_nano.sh`) |
| Biometrics ingest | `services/orion-biometrics/app/main.py`, `settings.py`, `docker-compose.yml`, `.env.example` |
| Grammar | `services/orion-biometrics/app/grammar_emit.py`, `orion/substrate/biometrics_loop/grammar_extract.py`, focused tests |
| Field | `services/orion-field-digester/app/tensor/channels.py`, `app/ingest/state_deltas.py`, `config/field/field_channel_glossary.v1.yaml`, focused tests |
| Smokes / README | `scripts/smoke_biometrics_cabinet_*.sh`, extend `scripts/smoke_field_digester_biometrics.sh`, `services/orion-biometrics/README.md` |

---

### Task 1: Align frame + cabinet_sensors to baseline-relative v1

**Owns:** `orion/schemas/telemetry/cabinet_sensor_frame.py`, `orion/telemetry/cabinet_sensors.py`, `orion/schemas/telemetry/biometrics.py`, `orion/telemetry/biometrics_pipeline.py`, `tests/test_cabinet_sensors.py`

- [ ] Rewrite/align frame schema to spec (`schema: orion.sensor_frame.v1`, no audio)
- [ ] Remove absolute thermal/humidity/PM normalizers; all activity pressures baseline-relative
- [ ] Measurement keys with units per spec; lidar only if `status==0`
- [ ] Pipeline merge: cabinet measurements/pressures additive; peak_pressure host-only
- [ ] Tests: absent≠zero, stale omits, lidar gate, pressures∈[0,1], constant→calm, host peak unchanged
- [ ] Commit

**Acceptance:** `pytest tests/test_cabinet_sensors.py -q` green; no absolute climate/PM pressure functions remain.

---

### Task 2: Host serial reader + install/smoke scripts

**Owns:** reader script, systemd unit, udev, setup/discover/serial smoke, reader unit tests

- [ ] Reader: by-id open, reconnect, validate NDJSON, atomic write, preserve last good, status ok/stale/error/missing
- [ ] No normalization/cognition in reader
- [ ] systemd + udev + setup script idempotent
- [ ] Unit tests for malformed-line preservation + atomic write
- [ ] Commit

**Acceptance:** `pytest tests/test_cabinet_sensor_reader.py -q` green; `bash -n` on shell scripts.

---

### Task 3: Firmware sketch + flash helper

**Owns:** `firmware/athena-cabinet-nano/`, `scripts/flash_athena_cabinet_nano.sh`

- [ ] Sketch emits ~1 Hz NDJSON matching frame schema; omit failed sensors; BNO085 UART-RVC; no audio
- [ ] Soft-fail per sensor; flash script uses by-id + arduino-cli (or documents required tool)
- [ ] Commit (no hardware flash required for task gate; sketch must compile if toolchain present, else note)

**Acceptance:** Frame JSON examples in comments/README of firmware dir match schema; flash script discovers by-id.

---

### Task 4: Biometrics ingest of snapshot

**Depends on:** Task 1  
**Owns:** `services/orion-biometrics/app/main.py`, `settings.py`, `docker-compose.yml`, `.env.example` (+ sync local `.env`)

- [ ] Settings: `CABINET_SENSORS_PATH`, `CABINET_SENSOR_STALE_AFTER_SEC`
- [ ] Load snapshot each tick → `sample.sensors` or omit; mark stale
- [ ] ro bind `/run/orion-sensors`
- [ ] Commit

**Acceptance:** Unit/integration test or focused test that stale/missing file ⇒ no sensors / no zero fill; env example updated.

---

### Task 5: Grammar emit + substrate extract

**Depends on:** Task 1  
**Owns:** `grammar_emit.py`, `grammar_extract.py`, focused tests

- [ ] New atoms/roles for six activities + staleness
- [ ] Extract into `pressure_hints` under exact pressure keys
- [ ] Host atoms unchanged when sensors absent
- [ ] Commit

**Acceptance:** Focused grammar tests green.

---

### Task 6: Field channels + state_deltas + glossary

**Depends on:** Task 1 (key names)  
**Owns:** `channels.py`, `state_deltas.py`, `field_channel_glossary.v1.yaml`, focused tests

- [ ] Add 7 channels to `NODE_CHANNELS`
- [ ] Map hints → `Perturbation(mode="replace")` under `node_biometrics`
- [ ] Glossary entries
- [ ] Commit

**Acceptance:** Test asserts replace mode for cabinet hints; channels listed.

---

### Task 7: Smokes + README note

**Depends on:** Tasks 2–6  
**Owns:** smoke scripts, biometrics README cabinet section

- [ ] Smokes for serial, biometrics sensors, grammar, extend field-digester biometrics smoke
- [ ] README: path diagram + USB mic document-only
- [ ] Commit

**Acceptance:** Scripts `bash -n` clean; README documents non-goals.

---

## Parallelism

- Wave 1 (parallel): Tasks 1, 2, 3
- Wave 2 (parallel after Task 1): Tasks 4, 5, 6
- Wave 3: Task 7

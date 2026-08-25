# Athena ambient audio levels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continuous CMTECK USB mic RMS/peak levels on Athena flow into biometrics → grammar → field as `cabinet_ambient_audio_activity` (levels-only; no Whisper).

**Architecture:** Host systemd ALSA reader writes `/run/orion-audio/latest.json`; `orion-biometrics` loads sibling `ambient_audio` on the sample (not Nano `sensors`); EWMA activity on RMS; grammar + field digester `mode="replace"` on node `athena`.

**Tech Stack:** Python 3, ALSA/`arecord`, systemd, existing biometrics/cabinet EWMA patterns, field digester NODE_CHANNELS.

**Spec:** `docs/superpowers/specs/2026-08-24-athena-ambient-audio-levels-design.md`

**Worktree:** `/mnt/scripts/Orion-Sapienform-athena-ambient-audio-levels` on `feat/athena-ambient-audio-levels`

## Global Constraints

- Levels only — no Whisper, no STT bus traffic, no Hub UI
- Device bind: `plughw:CARD=CMTECK,DEV=0` (never hard-coded `hw:0,0`)
- Absent/stale → omit measurement keys (never zero-fill)
- Do not fold ambient into host `peak_pressure` / `constraint`
- Perturbation `mode="replace"` for activity and staleness
- Mirror cabinet Nano patterns (`orion_cabinet_sensor_reader.py`, `cabinet_snapshot.py`, `cabinet_sensors.py`)
- Env parity: update `.env_example` + run `python scripts/sync_local_env_from_example.py` when env keys change
- Commit from this worktree only; never stage `.env`
- Every new loader degrades to `None` / omit on absent input — never raise on missing file

## File map

| File | Responsibility |
|---|---|
| `orion/schemas/telemetry/ambient_audio.py` | Snapshot schema `orion.ambient_audio.v1` |
| `orion/schemas/telemetry/biometrics.py` | `BiometricsSampleV1.ambient_audio` optional field |
| `orion/telemetry/ambient_audio.py` | extract measurements + EWMA pressure tracker |
| `services/orion-biometrics/app/ambient_audio_snapshot.py` | Load `/run/orion-audio/latest.json` |
| `scripts/orion_ambient_audio_reader.py` | Host ALSA capture loop |
| `deploy/systemd/orion-ambient-audio.service` | systemd unit |
| `scripts/setup_athena_ambient_audio.sh` etc. | setup / discover / smoke |
| biometrics main/grammar/settings/compose | wire sample + atoms |
| grammar_extract + field digester + glossary | hints → channels |

---

### Task 1: Schema + telemetry helpers (measurements + EWMA)

**Files:**
- Create: `orion/schemas/telemetry/ambient_audio.py`
- Create: `orion/telemetry/ambient_audio.py`
- Modify: `orion/schemas/telemetry/biometrics.py` — add optional `ambient_audio: Optional[Dict[str, Any]] = None` with docstring mirroring `sensors`
- Create: `tests/test_ambient_audio.py`
- Test: `tests/test_ambient_audio.py`

**Interfaces:**
- Produces:
  - `AMBIENT_AUDIO_SCHEMA_V1 = "orion.ambient_audio.v1"`
  - `AmbientAudioSnapshotV1` (pydantic) with fields: schema, status, received_at, device, window_sec, sample_rate, channels, rms, peak; optional error
  - `extract_ambient_audio_measurements(ambient_audio: Optional[Dict]) -> Dict[str, float]` → keys `cabinet_ambient_rms`, `cabinet_ambient_peak` only when present and not stale
  - `AmbientAudioTracker` + `compute_ambient_audio_pressures(measurements, tracker) -> Dict[str, float]` → `cabinet_ambient_audio_activity` from RMS only via same `_ActivityChannel` / EwmaBand pattern as `orion/telemetry/cabinet_sensors.py`

- [ ] **Step 1: Write failing tests** in `tests/test_ambient_audio.py`:
  - valid snapshot model accepts known-good JSON shape
  - stale/absent → empty measurements
  - constant RMS for many ticks → activity rests at ~0.0
  - activity ∈ [0,1]

- [ ] **Step 2:** `pytest tests/test_ambient_audio.py -q` → FAIL (import missing)

- [ ] **Step 3:** Implement schema + `orion/telemetry/ambient_audio.py` by copying the EWMA rest-point pattern from `cabinet_sensors.py` (`_ActivityChannel.activity`).

- [ ] **Step 4:** `pytest tests/test_ambient_audio.py -q` → PASS

- [ ] **Step 5: Commit** `feat(telemetry): ambient audio snapshot schema and RMS activity pressure`

---

### Task 2: Host reader + systemd + setup/discover/smoke

**Files:**
- Create: `scripts/orion_ambient_audio_reader.py`
- Create: `deploy/systemd/orion-ambient-audio.service`
- Create: `scripts/setup_athena_ambient_audio.sh`
- Create: `scripts/discover_athena_ambient_audio.sh`
- Create: `scripts/smoke_athena_ambient_audio.sh`
- Create: `tests/test_ambient_audio_reader.py` (unit: levels from synthetic PCM; atomic write; failed capture keeps last good)

**Interfaces:**
- Consumes: none from Task 1 at runtime (writes raw JSON matching schema)
- Produces: atomic file at `ORION_AMBIENT_AUDIO_PATH` default `/run/orion-audio/latest.json`
- Env: `ORION_AMBIENT_AUDIO_PATH`, `ORION_AMBIENT_AUDIO_DEVICE` default `plughw:CARD=CMTECK,DEV=0`, window ~0.5s
- Capture via `arecord` subprocess (or equivalent); compute rms/peak like Whisper `STTEngine._measure_wav_levels`
- systemd: `RuntimeDirectory=orion-audio`, `User=athena`, `SupplementaryGroups=audio`, `@ORION_ROOT@` substitution like cabinet unit
- setup: install alsa-utils if needed, `usermod -aG audio`, install unit, enable service, one-shot verify

- [ ] **Step 1:** Failing unit tests for `compute_levels_from_pcm` / atomic write helpers extracted from the reader module

- [ ] **Step 2:** Implement reader + deploy + scripts (mirror `scripts/orion_cabinet_sensor_reader.py` structure and `setup_athena_cabinet_sensors.sh`)

- [ ] **Step 3:** `pytest tests/test_ambient_audio_reader.py -q` → PASS

- [ ] **Step 4: Commit** `feat(host): Athena ambient audio ALSA reader and setup`

---

### Task 3: Biometrics load + pipeline + grammar

**Files:**
- Create: `services/orion-biometrics/app/ambient_audio_snapshot.py` (mirror `cabinet_snapshot.py`)
- Modify: `services/orion-biometrics/app/main.py` — load ambient snapshot into `sample_data["ambient_audio"]` when not None
- Modify: `orion/telemetry/biometrics_pipeline.py` — after cabinet block, merge ambient measurements/pressures; do **not** include ambient keys in `_peak_pressure` inputs beyond what's already in `pressures` before peak calc — compute peak **before** adding ambient (same as cabinet today: cabinet is merged after peak — keep that order for ambient too)
- Modify: `services/orion-biometrics/app/grammar_emit.py` — add activity + staleness atoms parallel to cabinet
- Modify: `services/orion-biometrics/app/settings.py`, `.env_example`, `docker-compose.yml` — `AMBIENT_AUDIO_PATH`, `AMBIENT_AUDIO_STALE_AFTER_SEC=5.0`; ro-bind `/run/orion-audio`
- Create: `services/orion-biometrics/tests/test_ambient_audio_snapshot.py`
- Create: `services/orion-biometrics/tests/test_ambient_audio_grammar.py` (or extend cabinet grammar tests carefully)
- Create: `scripts/smoke_biometrics_ambient_audio.sh`
- Modify: `services/orion-biometrics/README.md` — replace USB-mic document-only note

**Interfaces:**
- `load_ambient_audio_snapshot(path, stale_after_sec, now=None) -> Optional[Dict]` shape for sample field
- Grammar roles: `cabinet_ambient_audio_activity_signal`, `cabinet_ambient_audio_staleness_signal`
- Staleness salience: 0.0 fresh, 1.0 stale (match Nano)

- [ ] **Step 1:** Failing snapshot + grammar tests

- [ ] **Step 2:** Wire loader/pipeline/grammar/env/compose

- [ ] **Step 3:** Run `pytest services/orion-biometrics/tests/test_ambient_audio_snapshot.py services/orion-biometrics/tests/test_ambient_audio_grammar.py tests/test_ambient_audio.py -q` + sync local env

- [ ] **Step 4: Commit** `feat(biometrics): fold ambient audio levels into sample/summary/grammar`

---

### Task 4: Substrate hints + field digester + glossary

**Files:**
- Modify: `orion/substrate/biometrics_loop/grammar_extract.py` — map new signal roles → hint keys
- Modify: `services/orion-field-digester/app/tensor/channels.py` — add `cabinet_ambient_audio_activity`, `cabinet_ambient_audio_staleness`
- Modify: `services/orion-field-digester/app/ingest/state_deltas.py` — replace-mode perturbations in the cabinet loop
- Modify: `services/orion-field-digester/app/digestion/decay.py` — add new channels to decay list if cabinet channels are listed there
- Modify: `config/field/field_channel_glossary.v1.yaml`
- Modify: `services/orion-field-digester/tests/test_field_node_biometrics_perturbations.py`
- Modify: `tests/test_field_channel_glossary.py` if it asserts channel names
- Re-lock metrics if required by repo gates (`config/metrics/metric_definitions.lock.json`)

- [ ] **Step 1:** Extend field digester tests for replace-mode ambient channels

- [ ] **Step 2:** Wire extract + channels + state_deltas + glossary (+ lock if needed)

- [ ] **Step 3:** `pytest services/orion-field-digester/tests/test_field_node_biometrics_perturbations.py tests/test_field_channel_glossary.py -q`

- [ ] **Step 4: Commit** `feat(field): cabinet ambient audio activity and staleness channels`

---

### Task 5: Spec status + cross-links + graphify

**Files:**
- Modify: `docs/superpowers/specs/2026-08-24-athena-ambient-audio-levels-design.md` — status → approved/implemented-in-progress
- Modify: `docs/superpowers/specs/2026-08-23-athena-cabinet-sensor-node-design.md` — USB mic non-goal → pointer to ambient audio design
- Run: `scripts/safe_graphify_update.sh` after code changes

- [ ] **Step 1:** Doc cross-links only (no new features)

- [ ] **Step 2: Commit** `docs: link Athena ambient audio levels design`

---

## Acceptance (branch done when)

1. Unit tests above green
2. Host smoke can run on Athena after `sudo scripts/setup_athena_ambient_audio.sh` (may need Juniper for sudo)
3. No Whisper / STT wiring
4. Peak_pressure tests still pass with ambient present

## Spec coverage checklist

| Spec item | Task |
|---|---|
| Host reader + `/run/orion-audio/latest.json` | 2 |
| Schema + sample.ambient_audio | 1, 3 |
| measurements rms/peak | 1, 3 |
| activity pressure from RMS | 1, 3 |
| staleness signal/channel | 3, 4 |
| grammar → substrate → field replace | 3, 4 |
| setup/discover/smoke | 2, 3 |
| Non-goals (no Whisper) | all |
| Metric quality gate documented | already in spec |

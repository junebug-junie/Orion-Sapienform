# Athena cabinet ambient audio (levels-only) — design

Date: 2026-08-24  
Status: implemented on `feat/athena-ambient-audio-levels` — awaiting merge  
Worktree: `/mnt/scripts/Orion-Sapienform-athena-ambient-audio-levels` on `feat/athena-ambient-audio-levels`  
Hardware observed: ALSA card `CMTECK` (`MV-SILICON CMTECK`), `/dev/snd/pcmC0D0c`, capture S16_LE stereo @ 44100/48000

## Arsonist summary

Give Athena a continuous **cabinet noise level** sense from the USB mic already plugged into the host: a tiny systemd ALSA reader publishes RMS/peak snapshots; `orion-biometrics` folds them into the existing biometrics → grammar → substrate → field-digester path as **new** measurements and one baseline-relative activity channel on node `athena`. No Whisper. No dBA calibration. No overload of host fan/thermal pressures. Correlation against internal signals is a later consumer of the channel, not this patch.

This closes the explicit non-goal left in `docs/superpowers/specs/2026-08-23-athena-cabinet-sensor-node-design.md` (“USB mic on Athena — document-only”).

## Decisions locked

| Topic | Choice |
|---|---|
| Scope | Levels only (RMS + peak); Whisper transcription deferred |
| Architecture | Host systemd ALSA reader → `/run/orion-audio/latest.json` → biometrics (ro bind) → grammar → substrate → field-digester |
| Sample seam | Sibling `BiometricsSampleV1.ambient_audio` — **not** inside Nano `sensors` |
| Field node | `athena` |
| Pressure (v1) | One channel: `cabinet_ambient_audio_activity` from RMS via EWMA baseline-relative [0,1] |
| Absolute loudness / dBA | Out of scope for v1 |
| Device bind | ALSA by card name (`CARD=CMTECK`), never hard-coded `hw:0,0` |
| Perturbation mode | `mode="replace"` |
| Staleness | Emit `cabinet_ambient_audio_staleness` when `ambient_audio` is present on the sample (same pattern as Nano `cabinet_sensor_staleness`) |
| Host peak_pressure | Unchanged — ambient audio does not enter host max |

## Current architecture (grounded)

Existing path (do not bypass):

```text
orion-biometrics sample/summary
  → app/grammar_emit.py (GrammarEventV1 atoms)
  → orion/substrate/biometrics_loop (grammar_extract → pressure_hints → StateDeltaV1)
  → orion-field-digester app/ingest/state_deltas.py (Perturbation, mode="replace")
  → NODE_CHANNELS on the node
```

Parallel physical path already live for the Nano:

```text
Nano NDJSON → orion_cabinet_sensor_reader → /run/orion-sensors/latest.json
  → BiometricsSampleV1.sensors → cabinet_* measurements / *_activity pressures
```

Whisper STT (`orion-whisper-tts` on `orion:stt:intake`) is request/reply for Hub voice — not an ambient loop. Reuse its level-measurement *idea* (peak/rms over PCM) in the host reader; do **not** route continuous cabinet audio through the STT worker in v1.

Hardware blockers observed on Athena at design time (must be fixed by setup script):

- `athena` user not in `audio` group → `/dev/snd/pcmC0D0c` not openable
- `alsa-utils` / `arecord` not installed

## End-to-end data flow

```text
CMTECK USB mic (ALSA)
  → host systemd ambient audio reader (~1 Hz)
      short window capture → RMS + peak
      atomic /run/orion-audio/latest.json
  → orion-biometrics (ro bind): sample.ambient_audio
  → summary.measurements (cabinet_ambient_rms, cabinet_ambient_peak)
  → summary.pressures (cabinet_ambient_audio_activity)
  → grammar atoms → substrate pressure_hints
  → field-digester Perturbation(mode="replace") on node athena
```

Hard rules:

1. Physical audio levels belong to the host reader + biometrics path. No Docker USB/ALSA passthrough for v1.
2. Malformed/failed captures never overwrite the last good snapshot.
3. Stale/missing mic → omit ambient measurement keys and omit fresh activity pressure — never zero-fill.
4. Existing host and Nano cabinet channels unchanged in meaning and producers.
5. Host `peak_pressure` / `constraint` remain host-only.

## Host ambient audio reader

Tiny systemd service on Athena (no cognition inside Docker) — parallel to `scripts/orion_cabinet_sensor_reader.py`.

Responsibilities:

- Open ALSA device by stable card identity, e.g. `plughw:CARD=CMTECK,DEV=0` (env-overridable)
- Capture a short window (~0.5 s default); request mono 16 kHz S16 via `plughw` (hardware is stereo 44.1/48 kHz)
- Compute `rms` (float) and `peak` (int16 abs max) over the window
- Stamp host `received_at` (UTC ISO8601)
- Atomically write `/run/orion-audio/latest.json` (temp + rename)
- Status: `ok` | `stale` | `error` | `missing`, plus optional `error` string
- Preserve last good frame across failed captures
- **No** EWMA, pressures, or cognition

On-disk shape:

```json
{
  "schema": "orion.ambient_audio.v1",
  "status": "ok",
  "received_at": "2026-08-25T05:00:00.123Z",
  "device": "plughw:CARD=CMTECK,DEV=0",
  "window_sec": 0.5,
  "sample_rate": 16000,
  "channels": 1,
  "rms": 412.3,
  "peak": 1820
}
```

Supporting pieces:

- `RuntimeDirectory=orion-audio` (or tmpfiles.d) for `/run/orion-audio`
- Service user in `audio` group; setup verifies a one-shot capture succeeds
- Unit: restart on failure; reader under `scripts/` + unit under `deploy/systemd/`

Bind `/run/orion-audio` **read-only** into `orion-biometrics` via compose.

## Biometrics integration

### Sample

Extend `BiometricsSampleV1` with optional `ambient_audio` (sibling to `sensors`):

```text
ambient_audio: {
  rms: float,
  peak: int,
  received_at: <host ISO8601>,
  stale: bool,
  device?: str,
  window_sec?: float
}
```

- `ambient_audio` **absent** (not `{}`) if no valid snapshot has ever been read on this node.
- Biometrics sets `stale=true` when status ∈ {stale, error, missing} **or** `now - received_at > AMBIENT_AUDIO_STALE_AFTER_SEC` (env; default ~3–5 s at ~1 Hz).
- When stale: do not publish ambient measurement keys; do not emit fresh `cabinet_ambient_audio_activity`. Always emit `cabinet_ambient_audio_staleness` when `ambient_audio` is present (fresh or stale).

### Measurements

| Key | Source |
|---|---|
| `cabinet_ambient_rms` | snapshot `rms` when fresh |
| `cabinet_ambient_peak` | snapshot `peak` when fresh |

### Pressure (v1)

New key only — never write into existing host or Nano pressure keys.

- `cabinet_ambient_audio_activity` — EWMA band + volatility on **RMS** → anomaly ∈ [0,1]
- Peak remains a measurement for debug/smokes; not a second field channel in v1
- Tracker state in-process (cold start after biometrics restart is calm — document, do not fake history)
- Unit-test constant-input rest point (same invariant as other cabinet activity channels)

Host `peak_pressure` / `constraint` stay computed from host pressures only.

## Grammar and substrate

Extend `services/orion-biometrics/app/grammar_emit.py`:

- `cabinet_ambient_audio_activity_signal` (when pressure present)
- `cabinet_ambient_audio_staleness_signal` (when `ambient_audio` present on sample)

Extend `orion/substrate/biometrics_loop/grammar_extract.py` to copy atom salience into `pressure_hints` under `cabinet_ambient_audio_activity` and `cabinet_ambient_audio_staleness`.

Existing host / Nano grammar behavior unchanged when `ambient_audio` is absent.

## Field-digester

1. Add `cabinet_ambient_audio_activity` and `cabinet_ambient_audio_staleness` to `NODE_CHANNELS`.
2. Glossary entries in `config/field/field_channel_glossary.v1.yaml` (`physical_substrate` for activity; `sensor_trust_liveness` for staleness).
3. In `state_deltas.py`, for `target_kind == "node_biometrics"`, map both hints → `Perturbation(..., mode="replace")` on node `athena`.

Do not map ambient audio into `fan_pressure`, `thermal_pressure`, or any existing host channel.

Acceptance smoke must show the channel follows RMS activity and recovers downward under replace mode.

## Automation

| Script | Purpose |
|---|---|
| `scripts/setup_athena_ambient_audio.sh` | alsa-utils, audio group, runtime dir, systemd install/enable, one-shot capture verify |
| `scripts/discover_athena_ambient_audio.sh` | Print/verify ALSA card + chosen plughw string |
| `scripts/smoke_athena_ambient_audio.sh` | N valid snapshots; fail on timeout / only-error |
| `scripts/smoke_biometrics_ambient_audio.sh` | Biometrics sees fresh ambient measurements + activity pressure |
| Extend field digester biometrics smoke | New channel present, moves, recovers downward |

## Tests

Unit / focused:

- Ambient snapshot schema accepts valid JSON; rejects malformed
- Reader logic: failed capture does not clobber last good; atomic write semantics
- Stale ambient → no `cabinet_ambient_*` measurement keys
- Pressure bounded [0,1]; constant RMS → calm activity rest point
- `state_deltas` emits `mode="replace"` for ambient audio hint
- Host peak_pressure tests unchanged when ambient present or absent

E2E: scripted smokes on Athena with the CMTECK attached.

## Acceptance checklist

1. Mic reconnect/replug recovers without operator action (card name bind).
2. Card index change does not matter (`CARD=CMTECK` only).
3. Failed capture does not poison last good sample.
4. Stale/missing represented as stale/absent, not zeros.
5. `cabinet_ambient_rms` / `cabinet_ambient_peak` visible through biometrics when fresh.
6. `cabinet_ambient_audio_activity` bounded 0..1 and rests near 0 on constant quiet.
7. Pressure hint survives grammar/substrate path.
8. Field channel follows activity and recovers downward; no accumulate/saturate.
9. Existing biometrics / Nano cabinet / host channels unchanged.
10. Focused unit tests + Athena smokes.

## Non-goals

- Whisper / STT / continuous transcription
- Event-driven clip → `orion:stt:intake` (v2 candidate)
- Calibrated dBA or absolute “too loud” thresholds
- Hub UI panels
- Correlation / mutual-information consumers vs fan/TTS/load (later)
- Docker device passthrough for the mic
- Folding audio into Nano `sensors` / Arduino MAX9814
- Feeding ambient activity into host `peak_pressure`

## Files likely to touch

- `scripts/orion_ambient_audio_reader.py` (new)
- `scripts/setup_athena_ambient_audio.sh`, `discover_*`, `smoke_*` (new)
- `deploy/systemd/` ambient audio unit (new)
- `orion/schemas/telemetry/` ambient snapshot model + `BiometricsSampleV1.ambient_audio`
- `orion/telemetry/` ambient measurements + EWMA pressure (extend cabinet helpers or thin sibling module)
- `services/orion-biometrics/app/` loader, main merge, grammar_emit, settings, compose, `.env_example`, README
- `orion/substrate/biometrics_loop/grammar_extract.py`
- `services/orion-field-digester/app/ingest/state_deltas.py`, `tensor/channels.py`, decay if needed
- `config/field/field_channel_glossary.v1.yaml`
- `config/metrics/metric_definitions.lock.json` if glossary lock requires it
- Tests + smokes as above
- `services/orion-biometrics/README.md` — replace “USB mic document-only” with pointer to this design
- Cross-link from cabinet sensor design / README non-goals

## Env / config

Expected new keys (exact names at implementation; parity required — mirror cabinet `ORION_CABINET_*` / `CABINET_*` split):

- Host reader: `ORION_AMBIENT_AUDIO_PATH` (default `/run/orion-audio/latest.json`), `ORION_AMBIENT_AUDIO_DEVICE` (default `plughw:CARD=CMTECK,DEV=0`), window/rate knobs as needed
- Biometrics: `AMBIENT_AUDIO_PATH` (default same path), `AMBIENT_AUDIO_STALE_AFTER_SEC` (default `5`)

`.env_example` updated; local `.env` synced via `python scripts/sync_local_env_from_example.py`.

## Metric quality gate (CLAUDE.md 0A)

| Gate | Finding |
|---|---|
| 1. Provenance | RMS/peak from host ALSA PCM window in `orion_ambient_audio_reader.py`; activity from EWMA volatility on `cabinet_ambient_rms` in biometrics telemetry helpers |
| 2. Independence | Not a transform of fan/CPU/thermal pressures; shares physical cabinet context with Nano channels but different transducer (USB mic vs I2C/IMU). Fan noise will *correlate* with ambient RMS in the world — that is the point, not redundancy of the instrument |
| 3. Theory anchor | Continuous acoustic energy as an exogenous/enclosure cue for later coupling to interoceptive load (perception-frontier “room as companion” / self↔world partition). v1 ships the scalar; coupling math is out of scope |
| 4. Live-data sanity | Setup + smoke must print real RMS/peak from CMTECK before treating the channel as live. Constant input rest point unit-tested. Fan floor expected — baseline-relative activity, not absolute |
| 5. Existing mechanism | Whisper STT already measures peak/rms for silence gating — reuse the *measurement idea*, not the bus STT path. Cabinet Nano path is the integration template |
| 6. Reversibility | New optional sample field + one pressure/channel; disable by stopping the systemd reader or omitting the bind path |

## Risks

- **Permissions** — without `audio` group + verified open, reader fails closed; setup must be idempotent and prove capture.
- **Exclusive ALSA open** — reader owns the mic in v1; a later Whisper clip path must share or pause cleanly.
- **Cold EWMA** — first minutes after biometrics restart look calm; document.
- **Decay vs replace** — mitigate with `mode="replace"` + downward-recovery smoke (same class as other cabinet channels).

## Recommended next patch

1. Writing-plans: implementation plan from this approved spec.
2. Worktree `feat/athena-ambient-audio-levels`: reader + biometrics + grammar + field + tests + smokes + setup.
3. Live Athena smoke with CMTECK attached (group + alsa-utils first).

## Deferred (v2+)

- Spike-gated short clip → `STTRequestPayload` on `orion:stt:intake` for “what was that?”
- Lagged correlation / mutual information between `cabinet_ambient_audio_activity` and host `fan_pressure` / TTS playback / load
- Hub debug surface for ambient levels
